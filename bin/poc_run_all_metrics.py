#!/usr/bin/env python3
"""Run all Phase 3 quality metrics on a smoke/dev mini manifest_v5.

Thin orchestrator: subprocess-calls the production quality scripts with
the correct conda env per metric. Each underlying script is idempotent
(skip already-processed utterance_ids), so re-running this is safe.

Default input/output paths assume the smoke set; pass --set dev to use
the dev set instead.

Per-metric conda env (matches what works in production):
  tier1   → base env (numpy + scipy + soundfile)
  vad     → hu-speech-corpus env (torch + silero_vad)
  dnsmos  → audio_ds env (onnxruntime — hu-speech-corpus lacks this)
  lid     → audio_ds env (speechbrain + torch GPU)

Known coverage gaps per metric (production scripts not yet extended):
  - tier1: handles all 6 sources via 3 stages (yodas2 / vp_labeled / standalone).
  - vad + dnsmos: skip voxpopuli_hu_labeled (parquet-internal audio not handled).
  - lid: skips yodas2_hu000 AND voxpopuli_hu_labeled (segment-internal +
         parquet-internal not handled). Standalone-file sources only.

Run (base env is fine — this only orchestrates subprocesses):
  /media/cseti/datassd/conda/miniconda3/bin/python bin/poc_run_all_metrics.py
  /media/cseti/datassd/conda/miniconda3/bin/python bin/poc_run_all_metrics.py --set dev
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
BIN = PROJ_ROOT / "bin"

# Conda env python interpreters (project convention; see CLAUDE.md Rule 5).
PY_BASE = Path("/media/cseti/datassd/conda/miniconda3/bin/python")
PY_HU_SPEECH = Path("/media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python")
PY_AUDIO_DS = Path("/media/cseti/datassd/conda/miniconda3/envs/audio_ds/bin/python")

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
SET_ROOTS = {
    "smoke": DATA_ROOT / "processed" / "parquets" / "smoke_work",
    "dev":   DATA_ROOT / "processed" / "parquets" / "dev_work",
}


def run(cmd: list[str], label: str) -> tuple[int, float]:
    print(f"\n=== [{label}] running:", " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0
    print(f"=== [{label}] exit code: {rc}  ({dt:.1f}s)", flush=True)
    return rc, dt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["smoke", "dev"], default="smoke",
                   help="Which set to score (default: smoke).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Override manifest path (default: <set_root>/manifest.jsonl).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override output dir (default: <set_root>).")
    p.add_argument("--skip", type=str, default="",
                   help="Comma-separated metrics to skip "
                        "(any of: tier1, vad, dnsmos, lid).")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated metrics to run "
                        "(any of: tier1, vad, dnsmos, lid). "
                        "Wins over --skip if both given.")
    args = p.parse_args()

    root = SET_ROOTS[args.set]
    manifest = args.manifest if args.manifest else (root / "manifest.jsonl")
    out_dir = args.out_dir if args.out_dir else root
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    if not manifest.exists():
        print(f"[error] manifest not found: {manifest}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("tier1", PY_BASE, [
            str(BIN / "quality_tier1.py"),
            "--input", str(manifest),
            "--output", str(out_dir / "tier1.jsonl"),
            "--n_workers", "6",
        ]),
        ("vad", PY_HU_SPEECH, [
            str(BIN / "quality_tier2.py"),
            "--input", str(manifest),
            "--output", str(out_dir / "tier2_vad.jsonl"),
            "--metric", "vad",
            "--n_workers", "4",
        ]),
        ("dnsmos", PY_AUDIO_DS, [
            str(BIN / "quality_tier2.py"),
            "--input", str(manifest),
            "--output", str(out_dir / "tier2_dnsmos.jsonl"),
            "--metric", "dnsmos",
            "--n_workers", "2",
            "--batch_size", "10",
        ]),
        ("lid", PY_AUDIO_DS, [
            str(BIN / "audit_clip_language_purity_v2.py"),
            "--input", str(manifest),
            "--out", str(out_dir / "lid_pass1.jsonl"),
            "--source", "",
            "--stage", "1",
            "--batch-size", "8",
        ]),
    ]

    selected = []
    for label, py, cmd in metrics:
        if only and label not in only:
            continue
        if label in skip:
            continue
        selected.append((label, py, cmd))

    print(f"[orchestrator] set={args.set}")
    print(f"[orchestrator] manifest={manifest}")
    print(f"[orchestrator] out_dir={out_dir}")
    print(f"[orchestrator] running: {[m[0] for m in selected]}")

    t0 = time.time()
    failures: list[str] = []
    timings: list[tuple[str, float]] = []
    for label, py, cmd in selected:
        rc, dt = run([str(py)] + cmd, label)
        timings.append((label, dt))
        if rc != 0:
            failures.append(label)

    print(f"\n=== orchestrator summary ===")
    for label, dt in timings:
        print(f"  {label:8s}  {dt:6.1f}s")
    print(f"  total     {time.time() - t0:6.1f}s")
    if failures:
        print(f"\n[FAILURES] {failures}")
        return 1
    print(f"\nAll metrics OK. Sidecars in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
