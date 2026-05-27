#!/usr/bin/env python3
"""Build the full manifest_v5.parquet from manifest_v5.jsonl.

Direct JSONL → parquet conversion via pyarrow (no pandas materialization
of the full 4.48M-row table — uses pyarrow's streaming JSON reader and
writes parquet in one shot from the resulting Arrow table).

The output parquet has the same column structure as the smoke / dev / test
parquets — STRUCT-valued `quality_flags` and `transcripts`. The curator
serves all of these the same way via DuckDB.

Input:
  processed/manifests/manifest_v5.jsonl

Output:
  processed/parquets/manifest_v5.parquet  (atomic temp + rename)

Run with the base env (pyarrow ≥10):
  /media/cseti/datassd/conda/miniconda3/bin/python bin/build_manifest_v5_parquet.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
DEFAULT_INPUT = DATA_ROOT / "processed" / "manifests" / "manifest_v5.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "processed" / "parquets" / "manifest_v5.parquet"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--block-size", type=int, default=1 << 26,
                   help="JSON block size for streaming reader (default 64 MiB).")
    args = p.parse_args()

    if not args.input.exists():
        print(f"[error] missing input: {args.input}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build] reading {args.input}", file=sys.stderr)
    print(f"[build] size: {args.input.stat().st_size / 1024**2:.0f} MiB",
          file=sys.stderr)
    t0 = time.time()

    import pyarrow as pa
    import pyarrow.json as paj
    import pyarrow.parquet as pq

    read_opts = paj.ReadOptions(block_size=args.block_size)
    parse_opts = paj.ParseOptions(explicit_schema=None,
                                  unexpected_field_behavior="infer")
    table = paj.read_json(str(args.input),
                          read_options=read_opts,
                          parse_options=parse_opts)
    print(f"[build]   {table.num_rows:,} rows in Arrow table "
          f"({time.time() - t0:.1f}s)", file=sys.stderr)

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    print(f"[build] writing {tmp}", file=sys.stderr)
    pq.write_table(table, str(tmp), compression="zstd",
                   use_dictionary=True)
    os.replace(tmp, args.output)
    sz = args.output.stat().st_size / 1024**2
    print(f"\n=== manifest_v5.parquet summary ===")
    print(f"Rows:    {table.num_rows:,}")
    print(f"Cols:    {len(table.schema)}")
    print(f"Output:  {args.output}")
    print(f"Size:    {sz:.1f} MiB (zstd compressed)")
    print(f"Time:    {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
