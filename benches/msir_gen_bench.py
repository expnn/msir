#!/usr/bin/env python3
"""Portable zarr benchmark-data generator.

Generates variant zarr groups (differing in sample-dim chunk S_s, spatial chunk,
compression, and sharding) from a source zarr store, writing a container dir
that ``zarr_read_bench.py`` can auto-discover (the container is a zarr v3 group
with ``zarr.json`` at its root, and each variant is a subgroup with subset
``00`` holding flux/ivar/mask — matching the production layout).

Portable: ``--out-dir`` (default CWD) and ``--src`` required (local path,
``file://``, or ``oss://``). Source read uses zarr-python (preserves exact array
shapes incl. 3-D vs 4-D mask). For ``oss://`` source, zarr-python uses s3fs with
``AWS_*`` creds (mapped from ``OSS_*`` env if needed); reading full arrays over
OSS is slow — prefer a local source.

Variant matrix = cartesian product of ``--sample-chunks`` × ``--spatial`` ×
``--configs``. Each variant named ``s{S_s}_sp{spatial}_{config}``.

Usage::

    # sample-chunk matrix (default): s1_sp16_zstd .. s32_sp16_zstd
    uv run python zarr_gen_bench.py --src data/astro-images-zarr/LegacySurvey/train/00 --out-dir ./bench-data

    # spatial matrix (reproduces the spatial experiment's configs)
    uv run python zarr_gen_bench.py --src <src> --out-dir ./bench-data \\
        --sample-chunks 1 --spatial 16 32 80 160 --configs zstd nocomp noshard

    # from OSS source
    uv run python zarr_gen_bench.py --src oss:///default/cyc/.../LegacySurvey/train/00 --out-dir ./bench-data
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import zarr
import zarr.codecs
import zarr.storage
from tqdm import tqdm
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding

ZSTD = [zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")]
NOCOMP: list = []
CONTAINER_ZARR_JSON = json.dumps({"attributes": {}, "zarr_format": 3, "node_type": "group"})
CONFIGS = {"zstd", "nocomp", "noshard"}


def _oss_parts(store: str):
    """oss://[host]/<bucket>/<path> -> (bucket, path). [host] optional (env)."""
    from urllib.parse import urlparse

    full = urlparse(store).path.lstrip("/")
    bucket, _, rest = full.partition("/")
    if not bucket:
        raise ValueError(f"oss:// URL needs oss://[host]/<bucket>/<path>: {store!r}")
    return bucket, rest


def _map_oss_to_aws_env() -> None:
    """zarr-python/s3fs reads AWS_* env; map from OSS_* if AWS_* unset."""
    m = {
        "OSS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
        "OSS_ACCESS_KEY_SECRET": "AWS_SECRET_ACCESS_KEY",
        "OSS_ENDPOINT": "AWS_ENDPOINT_URL_S3",
    }
    for o, a in m.items():
        if os.environ.get(o) and not os.environ.get(a):
            os.environ[a] = os.environ[o]
    if not os.environ.get("AWS_REGION") and not os.environ.get("AWS_DEFAULT_REGION"):
        os.environ["AWS_REGION"] = "us-east-1"


def open_source(src: str):
    """Open the source zarr group; resolve root -> subset 00 if needed."""
    if src.startswith("oss://"):
        _map_oss_to_aws_env()
        bucket, path = _oss_parts(src)
        root = zarr.open_group(f"s3://{bucket}/{path}", mode="r")
    else:
        lp = Path(src.replace("file://", "") if src.startswith("file://") else src).resolve()
        root = zarr.open_group(zarr.storage.LocalStore(str(lp), read_only=True), mode="r")
    return root if "flux" in root else root["00"]


def load_source(src: str, n: int):
    g = open_source(src)
    flux = np.asarray(g["flux"][0:n])  # type: ignore[reportArgumentType, reportIndexIssue, reportCallIssue]
    ivar = np.asarray(g["ivar"][0:n]) if "ivar" in g else None  # type: ignore[reportArgumentType, reportIndexIssue, reportCallIssue]
    mask = np.asarray(g["mask"][0:n]) if "mask" in g else None  # type: ignore[reportArgumentType, reportIndexIssue, reportCallIssue]
    return flux, ivar, mask


def write_variant(
    out_dir: Path,
    name: str,
    s_s: int,
    spatial: int,
    config: str,
    shard_samples: int,
    flux_all: np.ndarray,
    ivar_all,
    mask_all,
) -> None:
    vdir = out_dir / name
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True, exist_ok=True)
    n, bands, h, w = flux_all.shape
    cke = DefaultChunkKeyEncoding(separator=".")
    compressors = NOCOMP if config == "nocomp" else ZSTD
    sharded = config != "noshard"
    flux_shards = (shard_samples, bands, h, w) if sharded else None
    store = zarr.storage.LocalStore(vdir, read_only=False)
    g = zarr.open_group(store=store, mode="a").require_group("00")
    zf = g.create_array(
        "flux",
        shape=(0, bands, h, w),
        chunks=(s_s, bands, spatial, spatial),
        shards=flux_shards,
        dtype="float32",
        chunk_key_encoding=cke,
        compressors=compressors,
    )
    if ivar_all is not None:
        zi = g.create_array(
            "ivar",
            shape=(0, bands, h, w),
            chunks=(s_s, bands, spatial, spatial),
            shards=flux_shards,
            dtype="float32",
            chunk_key_encoding=cke,
            compressors=compressors,
        )
    else:
        zi = None
    if mask_all is not None:
        if mask_all.ndim == 3:  # (n, H, W)
            mchunks = (s_s, spatial, spatial)
            mshards = (shard_samples, h, w) if sharded else None
            zm = g.create_array(
                "mask",
                shape=(0, h, w),
                chunks=mchunks,
                shards=mshards,
                dtype="bool",
                fill_value=True,
                chunk_key_encoding=cke,
                compressors=compressors,
            )
        else:  # (n, bands, H, W)
            zm = g.create_array(
                "mask",
                shape=(0, bands, h, w),
                chunks=(s_s, bands, spatial, spatial),
                shards=flux_shards,
                dtype="bool",
                fill_value=True,
                chunk_key_encoding=cke,
                compressors=compressors,
            )
    else:
        zm = None
    zf.resize((n, bands, h, w))
    if zi is not None:
        zi.resize((n, bands, h, w))
    if zm is not None:
        zm.resize(mask_all.shape)  # type: ignore[attr-defined]
    step = max(1, shard_samples)
    for s in tqdm(range((n + step - 1) // step), desc=f"write {name}", unit="shard"):
        i, j = s * step, min((s + 1) * step, n)
        zf[i:j] = flux_all[i:j]
        if zi is not None:
            zi[i:j] = ivar_all[i:j]  # type: ignore[index]
        if zm is not None:
            zm[i:j] = mask_all[i:j]  # type: ignore[index]


def directory_size(p) -> int:
    pp = Path(p)
    return sum(f.stat().st_size for f in pp.rglob("*") if f.is_file()) if pp.is_dir() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Portable zarr benchmark-data generator.")
    ap.add_argument(
        "--src",
        required=True,
        help="source zarr root (local path, file://, or oss://). "
        "A subset dir (e.g. .../train/00) or the train root (resolves 00).",
    )
    ap.add_argument(
        "--output-dir",
        default=".",
        help="output container dir (the zarr group of variants); default CWD. "
        "Variants and the container zarr.json are written directly here.",
    )
    ap.add_argument("--n-samples", type=int, default=4096, help="samples to copy from source.")
    ap.add_argument(
        "--sample-chunks",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32],
        help="sample-dim chunk sizes S_s (cartesian axis).",
    )
    ap.add_argument("--spatial", type=int, nargs="+", default=[16], help="spatial chunk sizes (cartesian axis).")
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["zstd"],
        choices=sorted(CONFIGS),
        help="zstd=sharded zstd; nocomp=sharded uncompressed; noshard=zstd no shard.",
    )
    ap.add_argument(
        "--shard-samples",
        type=int,
        default=1024,
        help="shard sample-dim size (for sharded configs; S_s must divide this).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"== gen variants: S_s={args.sample_chunks} spatial={args.spatial} "
        f"configs={args.configs} shard={args.shard_samples} =="
    )
    t0 = time.perf_counter()
    flux, ivar, mask = load_source(args.src, args.n_samples)
    n, bands, h, w = flux.shape
    print(
        f"  loaded {n} samples in {time.perf_counter() - t0:.1f}s "
        f"(flux {flux.nbytes / 1e9:.2f} GB; bands={bands} HxW={h}x{w})"
    )

    for sp in args.spatial:
        if h % sp != 0 or w % sp != 0:
            print(f"ERROR: spatial {sp} must divide image HxW={h}x{w}", file=sys.stderr)
            return 2
    if any(c != "noshard" for c in args.configs):
        bad = [s for s in args.sample_chunks if args.shard_samples % s != 0]
        if bad:
            print(
                f"ERROR: S_s {bad} must divide --shard-samples {args.shard_samples} (for sharded configs)",
                file=sys.stderr,
            )
            return 2

    sizes = {}
    for s_s in args.sample_chunks:
        for sp in args.spatial:
            for cfg in args.configs:
                name = f"s{s_s}_sp{sp}_{cfg}"
                t0 = time.perf_counter()
                write_variant(out_dir, name, s_s, sp, cfg, args.shard_samples, flux, ivar, mask)
                sizes[name] = directory_size(out_dir / name)
                print(f"  wrote {name}: {time.perf_counter() - t0:.1f}s  {sizes[name] / 1e9:.3f} GB")

    (out_dir / "zarr.json").write_text(CONTAINER_ZARR_JSON)
    print(f"  wrote container zarr.json (group) at {out_dir}")
    print(f"\n== summary ({len(sizes)} variants, sorted by size) ==")
    for name, s in sorted(sizes.items(), key=lambda kv: kv[1]):
        print(f"  {name:24s}  {s / 1e9:.3f} GB")
    print(f"\nBenchmark with: zarr_read_bench.py --store {out_dir} --output-dir <out>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
