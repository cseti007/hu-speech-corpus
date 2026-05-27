#!/usr/bin/env python3
"""Build smoke.parquet (or dev.parquet) by merging the mini manifest_v5
with the 4 per-metric sidecars (tier1, tier2_vad, tier2_dnsmos, lid_pass1).

Each metric's per-clip fields are merged into the manifest row's
`quality_flags` dict. Clips lacking a particular metric (e.g.
`voxpopuli_hu_labeled` rows are skipped by Tier-2 production scripts)
simply don't get those fields — no synthetic null padding.

Idempotent: overwrites the output parquet atomically (temp + rename).

Input (under `processed/parquets/<set>_work/`):
  manifest.jsonl       — mini manifest_v5 (300 rows for smoke)
  tier1.jsonl          — Tier-1 sidecar (rms_dbfs / peak_dbfs / is_clipped / silence_ratio)
  tier2_vad.jsonl      — VAD sidecar (vad_speech_ratio / vad_num_segments / vad_speech_sec)
  tier2_dnsmos.jsonl   — DNSMOS sidecar (dnsmos_sig / dnsmos_bak / dnsmos_ovrl)
  lid_pass1.jsonl      — LID Pass 1 sidecar (whole_clip_top1 / whole_clip_hu_prob / needs_pass2)

Output:
  processed/parquets/<set>.parquet  (e.g. smoke.parquet, dev.parquet)

Run with the base env (pandas + pyarrow):
  /media/cseti/datassd/conda/miniconda3/bin/python bin/build_set_parquet.py
  /media/cseti/datassd/conda/miniconda3/bin/python bin/build_set_parquet.py --set dev
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
PARQUET_DIR = DATA_ROOT / "processed" / "parquets"
SET_ROOTS = {
    "smoke": PARQUET_DIR / "smoke_work",
    "dev":   PARQUET_DIR / "dev_work",
}


def load_sidecar(path: Path) -> dict[str, dict]:
    """Read a JSONL sidecar into {utterance_id: {fields except utterance_id}}."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            uid = d["utterance_id"]
            out[uid] = {k: v for k, v in d.items() if k != "utterance_id"}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["smoke", "dev"], default="smoke")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Override mini-manifest path "
                        "(default: <set>_work/manifest.jsonl).")
    p.add_argument("--output", type=Path, default=None,
                   help="Override output parquet path "
                        "(default: processed/parquets/<set>.parquet).")
    args = p.parse_args()

    root = SET_ROOTS[args.set]
    manifest_path = args.manifest if args.manifest else (root / "manifest.jsonl")
    output_path = args.output if args.output else (PARQUET_DIR / f"{args.set}.parquet")

    if not manifest_path.exists():
        print(f"[error] manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    t0 = time.time()
    print(f"[build] loading sidecars from {root}", file=sys.stderr)
    sidecars = [
        ("tier1", load_sidecar(root / "tier1.jsonl")),
        ("vad", load_sidecar(root / "tier2_vad.jsonl")),
        ("dnsmos", load_sidecar(root / "tier2_dnsmos.jsonl")),
        ("lid", load_sidecar(root / "lid_pass1.jsonl")),
    ]
    for label, s in sidecars:
        print(f"[build]   {label}: {len(s):,} rows", file=sys.stderr)

    coverage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "tier1": 0, "vad": 0, "dnsmos": 0, "lid": 0})

    rows = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            uid = d["utterance_id"]
            src = d["source"]
            coverage[src]["total"] += 1
            qf = dict(d.get("quality_flags") or {})
            for label, sidecar in sidecars:
                if uid in sidecar:
                    qf.update(sidecar[uid])
                    coverage[src][label] += 1
            d["quality_flags"] = qf
            rows.append(d)

    # Write parquet (atomic temp + rename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, output_path)

    print(f"\n=== {args.set}.parquet summary ===")
    print(f"Total rows: {len(rows):,}")
    print(f"Output:     {output_path}")
    print(f"File size:  {output_path.stat().st_size / 1024:.1f} KiB")
    print()
    print(f"{'source':30s} {'total':>5s} {'tier1':>5s} {'vad':>4s} "
          f"{'dnsmos':>6s} {'lid':>4s}")
    for src in sorted(coverage.keys()):
        c = coverage[src]
        print(f"  {src:28s} {c['total']:>5} {c['tier1']:>5} {c['vad']:>4} "
              f"{c['dnsmos']:>6} {c['lid']:>4}")
    print()
    print(f"Time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
