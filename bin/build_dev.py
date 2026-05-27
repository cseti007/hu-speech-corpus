#!/usr/bin/env python3
"""Build dev.parquet — 100h stratified sample from train.parquet for
Phase 4 ASR consensus WER measurement.

Hour-target composition (decided 2026-05-26, voxpopuli-heavy because
voxpopuli is the most-problematic source — see memory `smoke-and-dev-sets`):

  voxpopuli_resegmented       55 h
  yodas2_hu000                12 h
  voxpopuli_hu_labeled        10 h  (HF train + HF dev — NEVER hf_split='test')
  common_voice_25_0_hu        10 h  (validated only)
  podcasts_hu_cc               5 h
  librivox_hu                  5 h
  ----------------------------- ----
  TOTAL                       97 h  (sources sum to 97; rounding budget for partial clips)

For each source we draw clips greedily (random order, fixed seed) until we
hit (or exceed) the hour target. Smaller sources may underfill if the
source itself has fewer hours than the target — that's reported in the
summary, not an error.

Fixed seed (default 42) for reproducibility. Re-running with the same
inputs produces the same dev set.

Input:
  processed/parquets/train.parquet  (must be built first by bin/build_train.py)

Output:
  processed/parquets/dev.parquet                 (full v5 schema, ~36k clips)
  processed/parquets/dev_work/dev_uids.txt       (uid list, useful downstream)

Run with the base env (pandas + pyarrow):
  /media/cseti/datassd/conda/miniconda3/bin/python bin/build_dev.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
PARQUET_DIR = DATA_ROOT / "processed" / "parquets"
DEFAULT_INPUT = PARQUET_DIR / "train.parquet"
DEFAULT_OUTPUT = PARQUET_DIR / "dev.parquet"
DEFAULT_UID_LIST = PARQUET_DIR / "dev_work" / "dev_uids.txt"

# Per-source hour targets. See module docstring.
HOUR_TARGETS = {
    "voxpopuli_resegmented": 55.0,
    "yodas2_hu000":          12.0,
    "voxpopuli_hu_labeled":  10.0,
    "common_voice_25_0_hu":  10.0,
    "podcasts_hu_cc":         5.0,
    "librivox_hu":            5.0,
}


def _eligible(row: dict) -> bool:
    """Per-source eligibility filter. Anything not eligible is silently
    excluded from the random pool (the random sampler only sees eligibles)."""
    src = row.get("source")
    qf = row.get("quality_flags") or {}
    if src == "common_voice_25_0_hu":
        # Only validated CV25 clips (others are insufficient-vote or rejected).
        return qf.get("cv25_status") == "validated"
    # vp_labeled: we already excluded hf_split='test' via build_train. Any
    # remaining vp_labeled row is eligible (train or dev HF split).
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--uid-list", type=Path, default=DEFAULT_UID_LIST)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    print(f"[dev] reading {args.input}", file=sys.stderr)
    import pandas as pd
    df = pd.read_parquet(args.input)
    print(f"[dev]   {len(df):,} rows in train", file=sys.stderr)

    # Group eligible row indices by source.
    by_src: dict[str, list[int]] = defaultdict(list)
    durations: dict[int, float] = {}
    for i, row in df.iterrows():
        row_d = row.to_dict()
        if not _eligible(row_d):
            continue
        src = row_d.get("source")
        if src not in HOUR_TARGETS:
            continue
        by_src[src].append(i)
        durations[i] = float(row_d.get("duration_sec") or 0.0)

    rng = random.Random(args.seed)
    selected_indices: list[int] = []
    per_src_stats: dict[str, dict] = {}
    print(f"\n[dev] sampling per source (seed={args.seed}):", file=sys.stderr)
    for src in HOUR_TARGETS:
        pool = by_src.get(src, [])
        if not pool:
            print(f"  {src:30s} no eligible clips in train", file=sys.stderr)
            per_src_stats[src] = {"target_h": HOUR_TARGETS[src],
                                   "picked": 0, "hours": 0.0,
                                   "pool_size": 0, "pool_hours": 0.0}
            continue
        pool_hours = sum(durations[i] for i in pool) / 3600.0
        rng.shuffle(pool)
        target_sec = HOUR_TARGETS[src] * 3600.0
        acc = 0.0
        picked: list[int] = []
        for i in pool:
            if acc >= target_sec:
                break
            picked.append(i)
            acc += durations[i]
        selected_indices.extend(picked)
        per_src_stats[src] = {
            "target_h": HOUR_TARGETS[src],
            "picked": len(picked),
            "hours": acc / 3600.0,
            "pool_size": len(pool),
            "pool_hours": pool_hours,
        }
        print(f"  {src:30s} {len(picked):>6,} clips, {acc/3600.0:>6.2f}h "
              f"(target {HOUR_TARGETS[src]:.0f}h, pool {len(pool):,} / "
              f"{pool_hours:.0f}h)", file=sys.stderr)

    # Stable order: by source (HOUR_TARGETS order), then by index.
    src_order = {s: i for i, s in enumerate(HOUR_TARGETS)}
    selected_indices.sort(key=lambda i: (src_order.get(df.at[i, "source"], 99), i))
    out_df = df.loc[selected_indices].copy()

    # Stamp set_membership = 'dev' into quality_flags.
    new_qf = []
    for qf in out_df["quality_flags"]:
        qf = dict(qf) if qf else {}
        qf["set_membership"] = "dev"
        new_qf.append(qf)
    out_df["quality_flags"] = new_qf

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.uid_list.parent.mkdir(parents=True, exist_ok=True)
    tmp_pq = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp_uids = args.uid_list.with_suffix(args.uid_list.suffix + ".tmp")
    out_df.to_parquet(tmp_pq, index=False)
    os.replace(tmp_pq, args.output)
    with tmp_uids.open("w", encoding="utf-8") as f:
        for uid in out_df["utterance_id"]:
            f.write(str(uid) + "\n")
    os.replace(tmp_uids, args.uid_list)

    total_h = sum(s["hours"] for s in per_src_stats.values())
    total_n = sum(s["picked"] for s in per_src_stats.values())
    print(f"\n=== dev summary ===")
    print(f"Total rows: {total_n:,}")
    print(f"Total hours: {total_h:.2f}")
    print(f"Parquet: {args.output}")
    print(f"UID list: {args.uid_list}")
    print(f"Time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
