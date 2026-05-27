#!/usr/bin/env python3
"""Build train.parquet — manifest_v5 minus test uids.

Read the held-out uid list emitted by `bin/build_test_v1.py` and emit a
parquet containing every other manifest_v5 row, marked with
`quality_flags.set_membership = "train"`.

Input:
  processed/manifests/manifest_v5.jsonl
  processed/parquets/test_work/test_uids.txt

Output:
  processed/parquets/train.parquet

Run with the base env (pandas + pyarrow):
  /media/cseti/datassd/conda/miniconda3/bin/python bin/build_train.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
DEFAULT_INPUT = DATA_ROOT / "processed" / "manifests" / "manifest_v5.jsonl"
PARQUET_DIR = DATA_ROOT / "processed" / "parquets"
DEFAULT_OUTPUT = PARQUET_DIR / "train.parquet"
DEFAULT_TEST_UIDS = PARQUET_DIR / "test_work" / "test_uids.txt"


def load_uid_set(path: Path) -> set[str]:
    if not path.exists():
        print(f"[train] WARN: test uid list missing ({path}); train will "
              f"include all manifest_v5 rows", file=sys.stderr)
        return set()
    with path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--test-uids", type=Path, default=DEFAULT_TEST_UIDS)
    args = p.parse_args()

    t0 = time.time()
    test_uids = load_uid_set(args.test_uids)
    print(f"[train] excluding {len(test_uids):,} test uids", file=sys.stderr)

    rows: list[dict] = []
    source_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "hours": 0.0})

    with args.input.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["utterance_id"] in test_uids:
                continue
            qf = dict(row.get("quality_flags") or {})
            qf["set_membership"] = "train"
            row["quality_flags"] = qf
            rows.append(row)
            sc = source_counts[row["source"]]
            sc["count"] += 1
            sc["hours"] += float(row.get("duration_sec") or 0.0) / 3600.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_pq = args.output.with_suffix(args.output.suffix + ".tmp")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(tmp_pq, index=False)
    os.replace(tmp_pq, args.output)

    total_h = sum(s["hours"] for s in source_counts.values())
    print(f"\n=== train summary ===")
    print(f"Total rows: {len(rows):,}")
    print(f"Total hours: {total_h:.2f}")
    print()
    print("By source:")
    for src in sorted(source_counts.keys()):
        sc = source_counts[src]
        print(f"  {src:28s} {sc['count']:>9,} clips, {sc['hours']:>9.2f} h")
    print()
    print(f"Parquet: {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024 / 1024:.1f} MiB")
    print(f"Time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
