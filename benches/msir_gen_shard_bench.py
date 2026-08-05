#!/usr/bin/env python3
"""Portable zarr benchmark-data generator with auto-computed sharding.

Generates variant zarr groups from a source zarr store. Unlike
``msir_gen_bench.py`` (which uses full-spatial shards), this script auto-
computes the sharding spatial dimension from the chunk spatial dimension:

    chunk shape  = (M, bands, X, X)
    shard shape  = (N, bands, max(32, X), max(32, X))

Only the sample dimension N of the shard is a free parameter
(``--shard-samples``); the spatial shard dimensions are derived as
``max(32, X)``.

Variant matrix = cartesian product of ``--chunk-samples`` (M) × ``--spatial``
(X) × ``--configs``. Each variant named ``s{M}_sp{X}_s{shard_sp}_{cfg}``
where ``shard_sp = max(32, X)`` is the auto-computed spatial shard size.

Usage::

    # defaults: M=[8,16,32] X=[16,32] N=1024 cfg=zstd
    uv run python msir_gen_shard_bench.py --src data/.../00 --output-dir ./bench-data

    # custom M and X
    uv run python msir_gen_shard_bench.py --src <src> --output-dir ./bench-data \\
        --chunk-samples 8 16 32 64 --spatial 16 32 --configs zstd nocomp noshard
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
    flux = np.asarray(g["flux"][0:n])
    ivar = np.asarray(g["ivar"][0:n]) if "ivar" in g else None
    mask = np.asarray(g["mask"][0:n]) if "mask" in g else None
    return flux, ivar, mask


def write_variant(
    out_dir: Path,
    name: str,
    M: int,
    X: int,
    config: str,
    shard_samples: int,
    shard_sp: int,
    flux_all: np.ndarray,
    ivar_all,
    mask_all,
) -> None:
    """Write one variant. Shard spatial = shard_sp = max(32, X)."""
    vdir = out_dir / name
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True, exist_ok=True)
    n, bands, H, W = flux_all.shape
    cke = DefaultChunkKeyEncoding(separator=".")
    compressors = NOCOMP if config == "nocomp" else ZSTD
    sharded = config != "noshard"
    # shard shape: (N, bands, shard_sp, shard_sp) — only N is free; spatial auto-computed
    flux_shards = (shard_samples, bands, shard_sp, shard_sp) if sharded else None
    store = zarr.storage.LocalStore(vdir, read_only=False)
    g = zarr.open_group(store=store, mode="a").require_group("00")
    g.create_array(
        "flux",
        shape=(0, bands, H, W),
        chunks=(M, bands, X, X),
        shards=flux_shards,
        dtype="float32",
        chunk_key_encoding=cke,
        compressors=compressors,
    )
    if ivar_all is not None:
        g.create_array(
            "ivar",
            shape=(0, bands, H, W),
            chunks=(M, bands, X, X),
            shards=flux_shards,
            dtype="float32",
            chunk_key_encoding=cke,
            compressors=compressors,
        )
    if mask_all is not None:
        if mask_all.ndim == 3:  # (n, H, W)
            mchunks = (M, X, X)
            mshards = (shard_samples, shard_sp, shard_sp) if sharded else None
            g.create_array(
                "mask",
                shape=(0, H, W),
                chunks=mchunks,
                shards=mshards,
                dtype="bool",
                fill_value=True,
                chunk_key_encoding=cke,
                compressors=compressors,
            )
        else:  # (n, bands, H, W)
            g.create_array(
                "mask",
                shape=(0, bands, H, W),
                chunks=(M, bands, X, X),
                shards=flux_shards,
                dtype="bool",
                fill_value=True,
                chunk_key_encoding=cke,
                compressors=compressors,
            )
    zf = g["flux"]
    zi = g["ivar"] if ivar_all is not None else None
    zm = g["mask"] if mask_all is not None else None
    zf.resize((n, bands, H, W))
    if zi is not None:
        zi.resize((n, bands, H, W))
    if zm is not None:
        zm.resize(mask_all.shape)
    step = max(1, shard_samples)
    for s in tqdm(range((n + step - 1) // step), desc=f"write {name}", unit="shard"):
        i, j = s * step, min((s + 1) * step, n)
        zf[i:j] = flux_all[i:j]
        if zi is not None:
            zi[i:j] = ivar_all[i:j]
        if zm is not None:
            zm[i:j] = mask_all[i:j]


def directory_size(p) -> int:
    pp = Path(p)
    return sum(f.stat().st_size for f in pp.rglob("*") if f.is_file()) if pp.is_dir() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Zarr benchmark-data generator with auto-computed sharding spatial.")
    ap.add_argument(
        "--src",
        required=True,
        help="source zarr root (local path, file://, or oss://). "
        "A subset dir (e.g. .../train/00) or the train root (resolves 00).",
    )
    ap.add_argument("--output-dir", default=".", help="output container dir (the zarr group of variants); default CWD.")
    ap.add_argument("--n-samples", type=int, default=4096, help="samples to copy from source.")
    ap.add_argument(
        "--chunk-samples",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="chunk sample-dim sizes M (cartesian axis). Shard sample-dim N must be divisible by each M.",
    )
    ap.add_argument(
        "--spatial",
        type=int,
        nargs="+",
        default=[16, 32],
        help="spatial chunk sizes X (cartesian axis). Shard spatial is auto-computed as max(32, X).",
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["zstd"],
        choices=sorted(CONFIGS),
        help="zstd=sharded zstd; nocomp=sharded uncompressed; noshard=zstd no shard.",
    )
    ap.add_argument(
        "--shard-samples", type=int, default=1024, help="shard sample-dim size N (only configurable shard dimension)."
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== gen variants: M={args.chunk_samples} X={args.spatial} configs={args.configs} N={args.shard_samples} ==")
    t0 = time.perf_counter()
    flux, ivar, mask = load_source(args.src, args.n_samples)
    n, bands, H, W = flux.shape
    print(
        f"  loaded {n} samples in {time.perf_counter() - t0:.1f}s "
        f"(flux {flux.nbytes / 1e9:.2f} GB; bands={bands} HxW={H}x{W})"
    )

    # Validate spatial: X must divide H, W; shard_sp=max(32,X) must divide H, W
    # and be a multiple of X (shard must contain whole chunks).
    for X in args.spatial:
        shard_sp = max(32, X)
        if H % X != 0 or W % X != 0:
            print(f"ERROR: chunk spatial X={X} must divide image HxW={H}x{W}", file=sys.stderr)
            return 2
        if H % shard_sp != 0 or W % shard_sp != 0:
            print(f"ERROR: shard spatial max(32,{X})={shard_sp} must divide image HxW={H}x{W}", file=sys.stderr)
            return 2
        if shard_sp % X != 0:
            print(f"ERROR: shard spatial {shard_sp} must be a multiple of chunk spatial {X}", file=sys.stderr)
            return 2

    # Validate sample: N must be divisible by M (for sharded configs)
    if any(c != "noshard" for c in args.configs):
        bad = [M for M in args.chunk_samples if args.shard_samples % M != 0]
        if bad:
            print(
                f"ERROR: M {bad} must divide --shard-samples {args.shard_samples} (for sharded configs)",
                file=sys.stderr,
            )
            return 2

    sizes = {}
    for M in args.chunk_samples:
        for X in args.spatial:
            shard_sp = max(32, X)
            for cfg in args.configs:
                name = f"s{M}_sp{X}_s{shard_sp}_{cfg}"
                t0 = time.perf_counter()
                write_variant(out_dir, name, M, X, cfg, args.shard_samples, shard_sp, flux, ivar, mask)
                sizes[name] = directory_size(out_dir / name)
                print(
                    f"  wrote {name}: {time.perf_counter() - t0:.1f}s  {sizes[name] / 1e9:.3f} GB (shard_sp={shard_sp})"
                )

    (out_dir / "zarr.json").write_text(CONTAINER_ZARR_JSON)
    print(f"  wrote container zarr.json (group) at {out_dir}")
    print(f"\n== summary ({len(sizes)} variants, sorted by size) ==")
    for name, s in sorted(sizes.items(), key=lambda kv: kv[1]):
        print(f"  {name:32s}  {s / 1e9:.3f} GB")
    print(f"\nBenchmark with: zarr_read_bench.py --store {out_dir} --output-dir <out>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
