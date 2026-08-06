#!/usr/bin/env python3
"""Portable zarr read-throughput benchmark (msir) — the canonical eval tool.

Self-contained (msir/numpy/zarr/stdlib), no project-relative paths. Run from
anywhere; write results to --output-dir. Benchmarks any zarr v3 store
addressable by msir:
  - bare local path           -> wrapped as file://<abspath> (msir needs the scheme)
  - file://...                -> used as-is
  - oss://<bucket>/<path>     -> passed to msir (Aliyun OSS native; creds/endpoint
                                 via OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET /
                                 OSS_ENDPOINT env)
  - s3://<bucket>/<path>      -> passed to msir (S3; creds via AWS_ACCESS_KEY_ID /
                                 AWS_SECRET_ACCESS_KEY, endpoint via AWS_ENDPOINT_URL)
  - https://<host>/<bucket>/<path> -> passed to msir (S3 + explicit endpoint;
                                 creds via AWS_* env)

A single --store auto-detects single vs container (no mode distinction):
  - single zarr root       : msir opens it -> benchmark it.
  - container of variants  : zarr-python lists subgroup children (the container
                             should be a zarr group, i.e. a zarr.json at its
                             root — gen scripts write this; local containers
                             without it fall back to a filesystem iterdir) ->
                             benchmark each child via msir.
Local and remote share the same logic (data hierarchy is identical). Remote
(oss:// / s3://) container listing needs s3fs; a remote SINGLE store needs only
msir. msir parses the scheme itself (oss:// = Aliyun OSS native; s3:// = S3;
https:// = S3 + URL endpoint) — this script only uses s3fs for container
discovery, with the scheme-appropriate credential/endpoint env vars.

ONE metric, close to real training: one epoch = N samples read exactly once,
in a random (shuffle, without replacement) order batched by read_block_size —
the DataLoader pattern. N defaults to the full dataset (--epoch-samples to
cap; a capped epoch reads a random subset).

A single epoch per store: when the dataset is large enough that it does not
fit the OS page cache, cache effects are negligible and the single-pass
throughput is representative of real training data loading. No cache-clearing
between trials is attempted — POSIX_FADV_DONTNEED only suggests reclaiming
clean data pages and cannot restore a truly cold cache (inode/dentry/extent
and device-level caches survive), so repeated trials would not be
independent anyway.

Usage:
  uv run python zarr_read_bench.py --store bench/zarr_chunks/data_samplechunk_sp32 --output-dir ./out
  uv run python zarr_read_bench.py --store bench/zarr_chunks/data_samplechunk_sp32/ss4 --output-dir ./out
  uv run python zarr_read_bench.py --store s3://default/cyc/datasets/astro-images-v2/LegacySurvey/train --output-dir ./out
  uv run python zarr_read_bench.py --store oss://my-bucket/path/to/root --output-dir ./out
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import msir
import numpy as np
import zarr
import zarr.storage
from tabulate import tabulate

CROP = 96
FULL = 160
BLOCK = 32  # default read_block_size
BYTES_PER_SAMPLE = 4 * CROP * CROP * 4 + 1 * CROP * CROP * 1  # flux f32 + mask bool

SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)


def resolve_zarr_url(p: str) -> str:
    """URL schemes (oss://, s3://, file://, https://) pass through; bare local
    paths are wrapped as file://+abspath (msir needs the scheme to pick the
    local FileSystemStore; else it reports 'group metadata is missing')."""
    if SCHEME_RE.match(p):
        return p
    return "file://" + str(Path(p).resolve())


def local_path_of(store: str) -> Path | None:
    if SCHEME_RE.match(store):
        if store.lower().startswith("file://"):
            return Path(store[len("file://") :])
        return None
    return Path(store)


def spatial_tiles(spatial: int, crop: int = CROP, full: int = FULL) -> int:
    """Number of spatial chunks covered by the center crop window."""
    start = (full - crop) // 2
    end = start + crop
    n = len({i // spatial for i in range(start, end)})
    return n * n


def _chunk_shape_from_dict(md) -> list:
    """Inner chunk shape from a v3 array metadata dict (handles sharding_indexed)."""
    for c in md.get("codecs", []):
        if c.get("name") == "sharding_indexed":
            cs = c.get("configuration", {}).get("chunk_shape")
            if cs:
                return cs
    return md.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")


def _inner_chunk_shape(zarr_json_path: Path):
    return _chunk_shape_from_dict(json.loads(zarr_json_path.read_text()))


def detect_chunk_shape(root: Path):
    """Best-effort (S_s, spatial) from a local flux zarr.json under root."""
    candidates = []
    f = root / "flux" / "zarr.json"
    if f.exists():
        candidates.append(f)
    if root.is_dir():
        for sub in root.iterdir():
            f = sub / "flux" / "zarr.json"
            if f.exists():
                candidates.append(f)
            for sub2 in sub.iterdir() if sub.is_dir() else []:
                f = sub2 / "flux" / "zarr.json"
                if f.exists():
                    candidates.append(f)
    for f in candidates:
        try:
            cs = _inner_chunk_shape(f)
            if cs and len(cs) >= 4:
                return int(cs[0]), int(cs[-1])
        except Exception:  # ruff: ignore[blind-except, try-except-continue]
            continue
    return None


def detect_chunk_shape_remote(store: str):
    """Best-effort (S_s, spatial) from a remote (oss:// / s3:// / https://) flux
    zarr.json via s3fs. Only reads the flux array's zarr.json metadata (chunk
    shape) for the decomp analytical column.
    """
    bucket, path = _remote_parts(store)
    fs = _s3fs_for_remote(store)
    base = f"{bucket}/{path}".rstrip("/")
    candidates = [f"{base}/00/flux/zarr.json", f"{base}/flux/zarr.json"]
    try:
        for e in fs.ls(base):
            name = e.rsplit("/", 1)[-1]
            if name and fs.isdir(e):
                candidates.append(f"{base}/{name}/flux/zarr.json")
    except Exception:  # ruff: ignore[blind-except, try-except-pass]
        pass
    for c in candidates:
        try:
            md = json.loads(fs.cat(c))  # type: ignore[arg-type]
            cs = _chunk_shape_from_dict(md)
            if cs and len(cs) >= 4:
                return int(cs[0]), int(cs[-1])
        except Exception:  # ruff: ignore[blind-except, try-except-continue]
            continue
    return None


def make_reader(
    store_url: str,
    block_size: int,
    crop: int = CROP,
    disable_ivar: bool = True,
    disable_mask: bool = False,
    max_chunk: int = 5000,
):
    return msir.AstroImageReader(
        zarr_root_path=store_url,
        crop_size=crop,
        disable_mask=disable_mask,
        disable_ivar=disable_ivar,
        max_chunk_size=max_chunk,
        read_block_size=block_size,
    )


def run_epoch(reader, n: int, block: int, epoch_n: int, seed: int) -> tuple[float, int]:
    """One epoch: read `epoch_n` samples exactly once, in shuffled order.

    msir `read_batch` 的参数是 **block 起始索引**（read_block_size > 1 时每个值
    代表一个 block 的起始全局索引，实际读取 indices.len() × block 个样本），
    不是样本索引。因此这里从 `collect_block_ids` 取所有对齐的 block 起始，
    无放回洗牌后按 block 逐个读取 —— 每个样本恰好读一次（DataLoader 模式），
    且不会产生越界 block 起始。样本按 `per_call` 个 block 分批（对齐旧版
    run_bN 的 128 样本/次）。返回 (秒数, 实际读取样本数 = n_blocks × block)。
    """
    ranges = reader.collect_block_ids(0, 1)
    starts = np.array([s for r in ranges for s in r], dtype=np.uintp)
    n_blocks = min(epoch_n // block, len(starts))
    if n_blocks == 0:
        return 0.0, 0
    rng = np.random.default_rng(seed)
    picks = starts[rng.permutation(len(starts))[:n_blocks]]
    t0 = time.perf_counter()
    per_call = max(1, 128 // block)
    for i in range(0, len(picks), per_call):
        reader.read_batch(picks[i : i + per_call])
    return time.perf_counter() - t0, n_blocks * block


def mb_s(samples, seconds, bps=BYTES_PER_SAMPLE):
    return samples * bps / seconds / 1e6 if seconds > 0 else float("inf")


def directory_size(p: Path | None) -> int:
    if not p or not p.is_dir():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def bench_store(store: str, label: str, block, epoch_samples, seed=7331, disable_ivar=True, disable_mask=False):
    store_url = resolve_zarr_url(store)
    lp = local_path_of(store)
    r = make_reader(store_url, block, disable_ivar=disable_ivar, disable_mask=disable_mask)
    n = r.total_samples
    out = r.read_batch(np.array([0], dtype=np.uintp))
    assert out is not None
    assert out[0].shape == (block, 4, CROP, CROP), out[0].shape
    epoch_n = min(epoch_samples, n) if epoch_samples else n
    # 单次 shuffled epoch：每样本恰好读一次（DataLoader 模式）。
    # 不做多 trial 中位数：缓存无法彻底清除（POSIX_FADV_DONTNEED 不覆盖
    # inode/dentry/extent/设备层缓存），重复 trial 之间不独立；数据集足够大时
    # 单次全量读取的缓存命中占比可忽略，足以代表真实训练吞吐。
    t, epoch_n_read = run_epoch(r, n, block, epoch_n, seed)
    # decomp 解析注解（原语义）：每 block 读的 chunk 解压次数 = (block // S_s) * tiles。
    det = (
        detect_chunk_shape(lp)
        if lp is not None
        else (
            detect_chunk_shape_remote(store) if store.startswith(("oss://", "s3://", "http://", "https://")) else None
        )
    )
    s_s = int(det[0]) if det else None
    spatial = int(det[1]) if det else None
    tiles = spatial_tiles(spatial) if spatial else None
    decomp_per_block = ((block // s_s) * tiles) if (s_s and tiles) else None
    return {
        "store": store,
        "label": label,
        "n": n,
        "epoch_samples": epoch_n_read,
        "read_block_size": block,
        "epoch_s": round(t, 4),
        "S_s": s_s,
        "spatial": spatial,
        "decomp_per_block": decomp_per_block,
        "disk_gb": round(directory_size(lp) / 1e9, 3) if lp else None,
    }


def _fmt(v):
    return "n/a" if v is None else str(v)


HEADERS = ["label", "n", "epoch_n", "epoch_s", "samp/s", "MB/s", "decomp/blk", "disk_GB"]
# label 左对齐, 数值列右对齐; 交给 tabulate 的 colalign 强制生效(不依赖类型推断,
# 因为单元格已预先格式化为字符串). 错误行把 ERROR 放在 epoch_s 槽, 消息放最后一列.
COLALIGN = ("left", "right", "right", "right", "right", "right", "right", "right")


def _table_row(fr: dict) -> list:
    """单行结果(字符串单元), 供 tabulate 渲染终端表格与 markdown 表格共用."""
    if "error" in fr:
        return [fr["label"], "", "", "ERROR", "", "", "", fr["error"]]
    return [
        fr["label"],
        str(fr["n"]),
        str(fr["epoch_n"]),
        f"{fr['med_s']:.3f}",
        f"{fr['samp_s']:.0f}",
        f"{fr['MB_s']:.1f}",
        fr["decomp"],
        fr["disk"],
    ]


def _derive_label(store: str) -> str:
    s = store.rstrip("/")
    name = s.rsplit("/", 1)[-1] if "/" in s else store
    return name or "store"


def _remote_parts(store: str):
    """Extract (bucket, path) from a remote object-store URL for s3fs discovery.

    New msir semantics (bucket in the authority):
      oss://<bucket>/<path>     bucket = authority (endpoint via OSS_ENDPOINT env)
      s3://<bucket>/<path>      bucket = authority (endpoint via AWS_ENDPOINT_URL env)
      https://<host>/<bucket>/<path>  bucket = first path segment (endpoint = host)

    msir parses the URL itself; this only extracts bucket/path for s3fs listing.
    NOTE: the old empty-host form oss:///<bucket>/<path> is no longer valid.
    """
    from urllib.parse import urlparse

    u = urlparse(store)
    if u.scheme in ("oss", "s3"):
        bucket, _, rest = u.netloc, "", u.path.lstrip("/")
        # userinfo (key:secret@) is allowed in the authority; strip it
        if "@" in bucket:
            bucket = bucket.rsplit("@", 1)[-1]
        if not bucket:
            raise ValueError(f"{u.scheme}:// URL needs a bucket in the authority: {store!r}")
        return bucket, rest
    if u.scheme in ("http", "https"):
        full = u.path.lstrip("/")
        bucket, _, rest = full.partition("/")
        if not bucket:
            raise ValueError(f"{u.scheme}:// URL needs <host>/<bucket>/<path>: {store!r}")
        return bucket, rest
    raise ValueError(f"unsupported remote scheme in {store!r}")


def _s3fs_for_remote(store: str):
    """s3fs discovery client; creds/endpoint come from the scheme-appropriate env.

    oss:// -> OSS_* env vars (Aliyun OSS native endpoint)
    s3:// and https:// -> AWS_* env vars (S3-compatible endpoint)
    """
    from urllib.parse import urlparse

    import s3fs

    u = urlparse(store)
    if u.scheme == "oss":
        key = os.environ.get("OSS_ACCESS_KEY_ID")
        secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
        endpoint = os.environ.get("OSS_ENDPOINT")
    else:  # s3:// or http(s)://
        key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        if u.scheme in ("http", "https"):
            endpoint = f"{u.scheme}://{u.netloc}"
    return s3fs.S3FileSystem(
        key=key,
        secret=secret,
        endpoint_url=endpoint,
        anon=False,
    )


def discover_stores(store, label, crop, disable_ivar, disable_mask, max_chunk):
    """Auto-detect: a single zarr root vs a container of variant roots.

    Local: msir opens single; else zarr-python/iterdir lists container children.
    Remote (oss:// / s3:// / http(s)://): msir parses the URL as-is and reads the
    scheme-appropriate env (oss:// -> OSS_*; s3:// / http(s):// -> AWS_*).
    Container discovery uses s3fs with the same env namespace. NOTE: the old
    empty-host form oss:///<bucket>/<path> is no longer valid — the bucket is
    now always in the authority: oss://<bucket>/<path>.
    Returns list of (store_url, label, local_path_or_None).
    """
    # remote object store: pass the URL to msir as-is (msir parses + reads env);
    # container discovery via s3fs (scheme-appropriate creds/endpoint).
    if SCHEME_RE.match(store) and not store.lower().startswith("file://"):
        try:  # single? (msir parses the URL and reads scheme-appropriate env)
            r = msir.AstroImageReader(
                zarr_root_path=store,
                crop_size=crop,
                disable_mask=disable_mask,
                disable_ivar=disable_ivar,
                max_chunk_size=max_chunk,
                read_block_size=32,
            )
            _ = r.total_samples  # forces group open + make_index
            return [(store, label or _derive_label(store), None)]
        except Exception:  # ruff: ignore[blind-except, try-except-pass]
            pass
        bucket, rest = _remote_parts(store)
        try:
            fs = _s3fs_for_remote(store)
            prefix = (f"{bucket}/{rest}".rstrip("/") + "/") if rest else f"{bucket}/"
            kids = []
            for e in sorted(fs.ls(prefix)):
                name = e.rsplit("/", 1)[-1]
                if name and not name.endswith("zarr.json") and fs.isdir(e):
                    kids.append((store.rstrip("/") + "/" + name, name, None))
            if kids:
                return kids
        except Exception as e:
            raise ValueError(
                f"could not open {store!r} as a single zarr root, and s3fs container listing "
                f"failed ({type(e).__name__}: {e}). Check the scheme-appropriate env vars "
                f"(oss:// -> OSS_ENDPOINT, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET; "
                f"s3:// -> AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)."
            ) from e
        raise ValueError(f"could not open {store!r} as a single zarr root, and no variant subdirs found.")
    # local / file:// / other remote
    url = resolve_zarr_url(store)
    try:  # single zarr root (msir opens + make_index succeeds)?
        r = msir.AstroImageReader(
            zarr_root_path=url,
            crop_size=crop,
            disable_mask=disable_mask,
            disable_ivar=disable_ivar,
            max_chunk_size=max_chunk,
            read_block_size=32,
        )
        _ = r.total_samples
        return [(url, label or _derive_label(store), local_path_of(store))]
    except Exception:  # ruff: ignore[blind-except, try-except-pass]
        pass
    kids = []
    lp = local_path_of(store)
    if lp is not None and lp.is_dir():
        try:  # zarr-python listing (container is a zarr group with zarr.json)
            g = zarr.open_group(zarr.storage.LocalStore(str(lp)), mode="r")
            for k in sorted(g.keys()):
                ch = lp / k
                if ch.is_dir():
                    kids.append(("file://" + str(ch.resolve()), k, ch))
        except Exception:  # ruff: ignore[blind-except, try-except-pass]
            pass
        if not kids:  # fallback: filesystem iterdir (no container zarr.json)
            for ch in sorted(lp.iterdir()):
                if ch.is_dir() and (ch / "zarr.json").exists():
                    kids.append(("file://" + str(ch.resolve()), ch.name, ch))
        if kids:
            return kids
    if SCHEME_RE.match(store) and not store.lower().startswith("file://"):
        try:
            g = zarr.open_group(store, mode="r")
            base = store.rstrip("/")
            for k in sorted(g.keys()):
                kids.append((f"{base}/{k}", k, None))
            if kids:
                return kids
        except Exception as e:
            raise ValueError(
                f"could not open {store!r} as a single zarr root, and remote container listing "
                f"failed ({type(e).__name__}: {e}). For remote reads use oss:// (msir supports "
                f"oss:// only, not s3://); install s3fs/ossfs and set creds env vars."
            ) from e
    raise ValueError(f"could not open {store!r} as a single zarr root, and no variant subdirs found.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Portable zarr read benchmark (msir, OSS-capable, auto-discovery).")
    ap.add_argument(
        "--store",
        required=True,
        help="zarr root URL/path OR a container of variant roots. Local path / file:// / "
        "oss://<bucket>/<path> (Aliyun OSS native; reads OSS_ACCESS_KEY_ID/"
        "OSS_ACCESS_KEY_SECRET/OSS_ENDPOINT env) / s3://<bucket>/<path> (S3; reads "
        "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_ENDPOINT_URL env) / "
        "https://<host>/<bucket>/<path> (S3 + explicit endpoint; AWS_* env). "
        "Auto-detects single vs container (container = zarr group with zarr.json at root).",
    )
    ap.add_argument("--label", help="label for single-store mode (default: derived).")
    ap.add_argument("--output-dir", default=".", help="where to write results (default: CWD).")
    ap.add_argument("--run-name", default="zarr_read_bench", help="results file basename.")
    ap.add_argument("--read-block-size", type=int, default=BLOCK)
    ap.add_argument(
        "--epoch-samples",
        type=int,
        default=0,
        help="samples per epoch; 0 = full dataset. A capped epoch reads a random subset of the dataset.",
    )
    ap.add_argument(
        "--disable-ivar", action="store_true", default=True, help="skip ivar (matches pretrain; default True)."
    )
    ap.add_argument("--read-ivar", dest="disable_ivar", action="store_false", help="also read ivar.")
    ap.add_argument(
        "--disable-mask",
        action="store_true",
        default=False,
        help="skip mask (default False: mask is read, matching pretrain).",
    )
    ap.add_argument("--max-chunk-size", type=int, default=5000)
    ap.add_argument("--msir-chunk-concurrent", type=int, default=4)
    ap.add_argument("--msir-codec-concurrent", type=int, default=8)
    args = ap.parse_args()

    msir.configure(
        chunk_concurrent_minimum=args.msir_chunk_concurrent, codec_concurrent_target=args.msir_codec_concurrent
    )

    block = args.read_block_size
    bps = 4 * CROP * CROP * 4 + (0 if args.disable_mask else 1 * CROP * CROP)

    discovered = discover_stores(
        args.store, args.label, CROP, args.disable_ivar, args.disable_mask, args.max_chunk_size
    )
    print(f"== zarr read benchmark (msir, read_block_size={block}, one metric: shuffled epoch) ==")
    print(f"  store={args.store}  -> {len(discovered)} store(s): {[d[1] for d in discovered]}")
    print(
        f"  output_dir={args.output_dir}  crop={CROP}  ivar={'off' if args.disable_ivar else 'on'}  "
        f"mask={'off' if args.disable_mask else 'on'}  "
        f"epoch_samples={args.epoch_samples or 'all'}"
    )

    rows = []
    for store, label, _lp in discovered:
        try:
            r = bench_store(
                store, label, block, args.epoch_samples, disable_ivar=args.disable_ivar, disable_mask=args.disable_mask
            )
        except Exception as e:  # ruff: ignore[blind-except]
            print(f"\n--- {label}  store={store} ---", flush=True)
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            rows.append({"store": store, "label": label, "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append(r)
        print(f"\n--- {label}  store={store} ---", flush=True)
        print(
            f"  n={r['n']}  epoch_samples={r['epoch_samples']}  "
            f"epoch_s={r['epoch_s']}s  S_s={r['S_s']}  spatial={r['spatial']}  "
            f"decomp/blk={_fmt(r['decomp_per_block'])}  disk={r['disk_gb']} GB",
            flush=True,
        )
        print(
            f"  {r['epoch_samples'] / r['epoch_s']:9.0f} samp/s  "  # type: ignore[operator]
            f"{mb_s(r['epoch_samples'], r['epoch_s'], bps):8.1f} MB/s",
            flush=True,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in rows:
        if "error" in r:
            flat.append({"label": r["label"], "error": r["error"], "samp_s": None})
            continue
        t = r["epoch_s"]
        flat.append({
            "label": r["label"],
            "n": r["n"],
            "epoch_n": r["epoch_samples"],
            "med_s": t,
            "samp_s": (r["epoch_samples"] / t if t > 0 else float("inf")),
            "MB_s": mb_s(r["epoch_samples"], t, bps),
            "decomp": _fmt(r["decomp_per_block"]),
            "disk": _fmt(r["disk_gb"]),
        })
    flat.sort(key=lambda x: (x.get("samp_s") is None, -(x["samp_s"] if x.get("samp_s") else 0)))
    print(f"\n\n===== RESULTS (msir, one metric: shuffled epoch, read_block_size={block}; sorted by samp/s desc) =====")
    print(tabulate([_table_row(fr) for fr in flat], headers=HEADERS, tablefmt="grid", colalign=COLALIGN))

    jpath = out_dir / f"{args.run_name}.json"
    mpath = out_dir / f"{args.run_name}.md"
    payload = {
        "backend": f"msir {getattr(msir, '__version__', '?')}",
        "read_block_size": block,
        "crop": CROP,
        "epoch_samples": args.epoch_samples,
        "bytes_per_sample": bps,
        "stores": rows,
    }
    jpath.write_text(json.dumps(payload, indent=2))
    lines = [
        f"# zarr read benchmark (msir, one metric: shuffled epoch, read_block_size={block})\n",
        (
            f"crop={CROP} epoch_samples={args.epoch_samples or 'all'}. "
            f"One shuffled epoch per store: every sample read exactly once in random order "
            f"(DataLoader pattern); dataset large enough for cache effects to be negligible. "
            f"Sorted by samp/s desc.\n"
        ),
        tabulate([_table_row(fr) for fr in flat], headers=HEADERS, tablefmt="github", colalign=COLALIGN),
    ]
    mpath.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {jpath} and {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
