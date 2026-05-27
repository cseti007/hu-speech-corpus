#!/usr/bin/env python3
"""Smoke-test bench for `normalize_hu()` (in bin/build_manifest_v5.py).

Reads the smoke set's mini manifest, applies the CURRENT `normalize_hu()`
to every row's source_caption, and prints side-by-side: raw → new norm →
old norm (for delta detection). Optionally writes a JSONL log for diff
review.

This is the iterative-development tool for the normalizer (per
CLAUDE.md Rule 8 — smoke-test scripts before scaling to full corpus).
Workflow:
  1. Edit `normalize_hu()` in bin/build_manifest_v5.py
  2. Run this script:
       /media/cseti/datassd/conda/miniconda3/bin/python bin/test_normalize_hu_on_smoke.py
  3. Spot-check the diff lines (rows where old != new)
  4. If results look right, cascade-rebuild the full corpus

Run with the base env (num2words optional but recommended).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
DEFAULT_SMOKE_MANIFEST = (
    DATA_ROOT / "processed" / "parquets" / "smoke_work" / "manifest.jsonl"
)
DEFAULT_OUT_LOG = (
    DATA_ROOT / "processed" / "parquets" / "smoke_work"
    / "normalize_hu_test.jsonl"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_SMOKE_MANIFEST,
                   help="Mini-manifest to read source_caption from "
                        "(default: smoke_work/manifest.jsonl).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_LOG,
                   help="JSONL log for diff review.")
    p.add_argument("--show-all", action="store_true",
                   help="Print every row (default: only diffs vs. current "
                        "stored normalized field).")
    p.add_argument("--max-print", type=int, default=20,
                   help="Cap on stdout lines (default 20). Full output "
                        "always lands in --out.")
    args = p.parse_args()

    # Import the canonical normalizer fresh.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_manifest_v5 import normalize_hu  # noqa: WPS433

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tr = d.get("transcripts") or {}
            raw = tr.get("source_caption")
            old_norm = tr.get("source_caption_normalized")
            new_norm = normalize_hu(raw)
            rows.append({
                "utterance_id": d["utterance_id"],
                "source": d.get("source"),
                "raw": raw,
                "old_norm": old_norm,
                "new_norm": new_norm,
                "changed": (old_norm != new_norm),
            })

    n_total = len(rows)
    n_with_caption = sum(1 for r in rows if r["raw"])
    n_changed = sum(1 for r in rows if r["changed"] and r["raw"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[norm-test] rows: {n_total} (with source_caption: {n_with_caption})")
    print(f"[norm-test] changed (new norm != stored norm): {n_changed}")
    print(f"[norm-test] log: {args.out}")
    print()

    to_show = rows if args.show_all else [r for r in rows
                                          if r["changed"] and r["raw"]]
    if not to_show:
        print("[norm-test] no diffs — normalizer output matches stored values.")
        return 0
    print(f"[norm-test] showing first {min(args.max_print, len(to_show))} "
          f"of {len(to_show)} diff rows:\n")
    for r in to_show[:args.max_print]:
        print(f"--- {r['utterance_id']}  ({r['source']})")
        print(f"  raw : {r['raw']}")
        print(f"  OLD : {r['old_norm']}")
        print(f"  NEW : {r['new_norm']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
