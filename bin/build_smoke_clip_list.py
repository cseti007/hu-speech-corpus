#!/usr/bin/env python3
"""Build the smoke clip-list + mini manifest_v5 (~300 multi-source clips).

Stratified per-source sample. For sources with the Tier-1 quality sidecar:
40 normal + 10 outlier per source (outlier := is_clipped OR rms_dbfs < -45
OR silence_ratio > 0.6). If a source has < 10 outliers, fill the remainder
from the normal pool to hit the per-source target. For sources without
Tier-1 (currently `common_voice_25_0_hu`): 50 pure-random clips.

Fixed random seed (default 42) for reproducibility.

Input:
  processed/manifests/manifest_v5.jsonl
  processed/quality/tier1.jsonl  (rms_dbfs, peak_dbfs, is_clipped, silence_ratio)

Outputs (both atomic temp+rename):
  processed/parquets/smoke_work/clip_list.jsonl
    Audit log: one short row per selected clip with `selection_bucket`.

  processed/parquets/smoke_work/manifest.jsonl
    Mini manifest_v5: full v5 schema, 300 rows, with `smoke_bucket` added
    to `quality_flags`. This is what the production quality scripts
    (`quality_tier1.py`, `quality_tier2.py`,
    `audit_clip_language_purity_v2.py`) consume via their `--input` flag.

Run with the base env:
  /media/cseti/datassd/conda/miniconda3/bin/python bin/build_smoke_clip_list.py
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
INPUT_MANIFEST = DATA_ROOT / "processed" / "manifests" / "manifest_v5.jsonl"
INPUT_TIER1 = DATA_ROOT / "processed" / "quality" / "tier1.jsonl"
OUTPUT_DIR = DATA_ROOT / "processed" / "parquets" / "smoke_work"
OUTPUT_CLIP_LIST = OUTPUT_DIR / "clip_list.jsonl"
OUTPUT_MANIFEST = OUTPUT_DIR / "manifest.jsonl"

PER_SOURCE_TARGET = 50
OUTLIER_FRACTION = 0.2  # 10 of 50

SOURCES = [
    "voxpopuli_resegmented",
    "yodas2_hu000",
    "voxpopuli_hu_labeled",
    "librivox_hu",
    "podcasts_hu_cc",
    "common_voice_25_0_hu",
]

SOURCES_WITHOUT_TIER1 = {"common_voice_25_0_hu"}


def is_outlier(t1: dict) -> bool:
    """Tier-1 outlier rule: clipped OR very quiet OR mostly silent."""
    if t1.get("is_clipped"):
        return True
    rms = t1.get("rms_dbfs")
    if rms is not None and rms < -45:
        return True
    sil = t1.get("silence_ratio")
    if sil is not None and sil > 0.6:
        return True
    return False


def build_outlier_map(path: Path) -> dict[str, bool]:
    """Stream the Tier-1 sidecar and return {utterance_id: is_outlier_bool}."""
    out: dict[str, bool] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["utterance_id"]] = is_outlier(d)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=INPUT_MANIFEST,
                   help="Input manifest_v5.jsonl to sample from.")
    p.add_argument("--tier1", type=Path, default=INPUT_TIER1,
                   help="Tier-1 sidecar for outlier classification.")
    p.add_argument("--clip-list-output", type=Path, default=OUTPUT_CLIP_LIST,
                   help="Audit-log output: short rows with selection_bucket.")
    p.add_argument("--manifest-output", type=Path, default=OUTPUT_MANIFEST,
                   help="Mini manifest_v5 output: full v5 schema rows with "
                        "smoke_bucket added under quality_flags. This is what "
                        "production quality scripts consume via --input.")
    p.add_argument("--per-source", type=int, default=PER_SOURCE_TARGET)
    p.add_argument("--outlier-fraction", type=float, default=OUTLIER_FRACTION)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    print(f"[smoke] loading Tier-1 outlier map from {args.tier1}",
          file=sys.stderr)
    outlier_map = build_outlier_map(args.tier1)
    print(f"[smoke]   {len(outlier_map):,} Tier-1 rows "
          f"({time.time() - t0:.1f}s)", file=sys.stderr)

    # Pass 1: classify each manifest uid into per-source buckets.
    pool: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"normal": [], "outlier": [], "all": []}
    )
    print(f"[smoke] pass 1: scanning {args.manifest}", file=sys.stderr)
    n_seen = 0
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            src = d.get("source")
            if src not in SOURCES:
                continue
            uid = d["utterance_id"]
            if src in SOURCES_WITHOUT_TIER1:
                pool[src]["all"].append(uid)
            else:
                if outlier_map.get(uid, False):
                    pool[src]["outlier"].append(uid)
                else:
                    pool[src]["normal"].append(uid)
            n_seen += 1
            if n_seen % 1_000_000 == 0:
                print(f"  ... {n_seen:,} rows scanned",
                      file=sys.stderr, flush=True)
    print(f"[smoke] pass 1 done: {n_seen:,} rows ({time.time() - t0:.1f}s)",
          file=sys.stderr)
    del outlier_map  # free RAM before second pass

    print("[smoke] per-source pool sizes:", file=sys.stderr)
    for src in SOURCES:
        if src in SOURCES_WITHOUT_TIER1:
            print(f"  {src:30s} random_pool = {len(pool[src]['all']):>10,}",
                  file=sys.stderr)
        else:
            print(f"  {src:30s} normal = {len(pool[src]['normal']):>10,}, "
                  f"outlier = {len(pool[src]['outlier']):>6,}",
                  file=sys.stderr)

    # Sample uids per source.
    rng = random.Random(args.seed)
    target_outlier = int(round(args.per_source * args.outlier_fraction))
    target_normal = args.per_source - target_outlier
    selection_bucket: dict[str, str] = {}

    print(f"\n[smoke] sampling per source "
          f"(target {args.per_source}: {target_normal} normal + "
          f"{target_outlier} outlier; seed {args.seed})",
          file=sys.stderr)
    for src in SOURCES:
        if src in SOURCES_WITHOUT_TIER1:
            pool_all = pool[src]["all"]
            n = min(args.per_source, len(pool_all))
            picks = rng.sample(pool_all, n)
            for uid in picks:
                selection_bucket[uid] = "random"
            print(f"  {src:30s} {n:>3} random", file=sys.stderr)
        else:
            out_pool = pool[src]["outlier"]
            norm_pool = pool[src]["normal"]
            n_outlier = min(target_outlier, len(out_pool))
            # If fewer outliers than target, top up from normal pool.
            extra_normal = target_outlier - n_outlier
            n_normal = min(target_normal + extra_normal, len(norm_pool))
            out_picks = rng.sample(out_pool, n_outlier) if n_outlier else []
            norm_picks = rng.sample(norm_pool, n_normal) if n_normal else []
            for uid in out_picks:
                selection_bucket[uid] = "outlier"
            for uid in norm_picks:
                selection_bucket[uid] = "normal"
            total = n_outlier + n_normal
            print(f"  {src:30s} {n_outlier:>3} outlier + "
                  f"{n_normal:>3} normal = {total:>3}",
                  file=sys.stderr)

    del pool  # free RAM before second pass

    # Pass 2: re-scan manifest, collect full v5 rows for selected uids.
    print(f"\n[smoke] pass 2: re-scanning manifest to collect full v5 rows",
          file=sys.stderr)
    rows_by_src: dict[str, list[dict]] = defaultdict(list)
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            uid = d.get("utterance_id")
            bucket = selection_bucket.get(uid)
            if bucket is None:
                continue
            # Stash smoke_bucket into quality_flags for downstream tools.
            qf = dict(d.get("quality_flags") or {})
            qf["smoke_bucket"] = bucket
            d["quality_flags"] = qf
            rows_by_src[d["source"]].append(d)

    # Stable output order: by source (in SOURCES order), then bucket, then uid.
    bucket_order = {"outlier": 0, "normal": 1, "random": 2}
    args.clip_list_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    cl_tmp = args.clip_list_output.with_suffix(
        args.clip_list_output.suffix + ".tmp")
    ms_tmp = args.manifest_output.with_suffix(
        args.manifest_output.suffix + ".tmp")

    n_out = 0
    with cl_tmp.open("w", encoding="utf-8") as cl_f, \
         ms_tmp.open("w", encoding="utf-8") as ms_f:
        for src in SOURCES:
            rows = rows_by_src.get(src, [])
            rows.sort(key=lambda r: (
                bucket_order.get(r["quality_flags"]["smoke_bucket"], 9),
                r["utterance_id"]))
            for r in rows:
                bucket = r["quality_flags"]["smoke_bucket"]
                # Audit-log row (light)
                cl_row = {
                    "utterance_id": r["utterance_id"],
                    "source": r["source"],
                    "audio_path": r["audio_path"],
                    "duration_sec": r.get("duration_sec"),
                    "selection_bucket": bucket,
                }
                cl_f.write(json.dumps(cl_row, ensure_ascii=False) + "\n")
                # Full v5 row (what production scripts will iterate)
                ms_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1

    os.replace(cl_tmp, args.clip_list_output)
    os.replace(ms_tmp, args.manifest_output)

    print(f"\n=== smoke clip-list summary ===")
    print(f"Total selected: {n_out:,}")
    bucket_counts: dict[tuple[str, str], int] = defaultdict(int)
    total_dur: dict[str, float] = defaultdict(float)
    for src in SOURCES:
        for r in rows_by_src.get(src, []):
            bucket_counts[(src, r["quality_flags"]["smoke_bucket"])] += 1
            total_dur[src] += float(r.get("duration_sec") or 0.0)
    print()
    print(f"{'source':30s} {'bucket':>8s}  {'count':>5s}")
    for (src, bucket), count in sorted(bucket_counts.items()):
        print(f"  {src:28s} {bucket:>8s}  {count:>5}")
    print()
    print(f"{'source':30s} {'duration':>10s}")
    for src in SOURCES:
        print(f"  {src:28s} {total_dur[src]:>8.1f} s")
    print()
    print(f"Clip-list (audit):     {args.clip_list_output}")
    print(f"Mini manifest (input): {args.manifest_output}")
    print(f"Random seed: {args.seed}")
    print(f"Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
