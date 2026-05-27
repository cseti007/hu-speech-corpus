#!/usr/bin/env python3
"""Build test_v1.parquet — held-out evaluation set with ground-truth transcripts.

test_v1 composition (2026-05-26):
  A) voxpopuli_hu_labeled rows where quality_flags.hf_split == 'test'
     (~1,022 clips / 2.98 h, parliamentary register, ground-truth human labels)
  B) (planned) FLEURS-HU 12h read-speech benchmark — to be added once
     bin/download_fleurs_hu.py lands.

test_v2 will extend with consensus-validated clips after Phase 4 (clips where
source_caption matches all 3 ASR pillars byte-for-byte after normalization).

Input:
  processed/manifests/manifest_v5.jsonl  (must have hf_split derived for vp_labeled)

Output:
  processed/parquets/test.parquet                          (full v5 schema)
  processed/parquets/test_work/test_uids.txt               (uid list — consumed by build_train + build_dev for exclusion)

Run with the base env (pandas + pyarrow):
  /media/cseti/datassd/conda/miniconda3/bin/python bin/build_test_v1.py
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
DEFAULT_OUTPUT = PARQUET_DIR / "test.parquet"
DEFAULT_UID_LIST = PARQUET_DIR / "test_work" / "test_uids.txt"


def is_test_clip(row: dict) -> tuple[bool, str | None]:
    """Apply test selection criteria. Returns (selected, criterion_label).

    Add new criteria here as the test set grows."""
    src = row.get("source")
    qf = row.get("quality_flags") or {}

    # Criterion A: vp_labeled HF test split
    if src == "voxpopuli_hu_labeled" and qf.get("hf_split") == "test":
        return True, "vp_labeled_hf_test"

    # Criterion B (TODO): FLEURS-HU clips, when source key lands.
    # if src == "fleurs_hu":
    #     return True, "fleurs_hu"

    return False, None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--uid-list", type=Path, default=DEFAULT_UID_LIST)
    args = p.parse_args()

    t0 = time.time()
    rows: list[dict] = []
    criteria_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "hours": 0.0})

    print(f"[test_v1] reading {args.input}", file=sys.stderr)
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            selected, label = is_test_clip(row)
            if not selected:
                continue
            # Stash a `set_membership` marker so a clip's set is obvious in
            # any downstream tool reading the parquet.
            qf = dict(row.get("quality_flags") or {})
            qf["set_membership"] = "test"
            qf["test_criterion"] = label
            row["quality_flags"] = qf
            rows.append(row)
            criteria_counts[label] += 1
            sc = source_counts[row["source"]]
            sc["count"] += 1
            sc["hours"] += float(row.get("duration_sec") or 0.0) / 3600.0

    if not rows:
        print(f"[test_v1] WARN: no rows matched any test criterion", file=sys.stderr)
        return 1

    # Write parquet (atomic temp + rename)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.uid_list.parent.mkdir(parents=True, exist_ok=True)
    tmp_pq = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp_uids = args.uid_list.with_suffix(args.uid_list.suffix + ".tmp")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(tmp_pq, index=False)
    os.replace(tmp_pq, args.output)

    with tmp_uids.open("w", encoding="utf-8") as out_f:
        for r in rows:
            out_f.write(r["utterance_id"] + "\n")
    os.replace(tmp_uids, args.uid_list)

    print(f"\n=== test_v1 summary ===")
    print(f"Total rows: {len(rows):,}")
    total_h = sum(s["hours"] for s in source_counts.values())
    print(f"Total hours: {total_h:.2f}")
    print()
    print("By selection criterion:")
    for label, n in sorted(criteria_counts.items()):
        print(f"  {label:30s} {n:>6,}")
    print()
    print("By source:")
    for src in sorted(source_counts.keys()):
        sc = source_counts[src]
        print(f"  {src:28s} {sc['count']:>6,} clips, {sc['hours']:>7.2f} h")
    print()
    print(f"Parquet: {args.output}")
    print(f"UID list: {args.uid_list}")
    print(f"Time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
