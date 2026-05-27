#!/usr/bin/env python3
"""Validate `manifest.jsonl` against the original quality sidecars + stats.json.

Runs 5 checks:
  A. Schema sanity: every row is valid JSON and quality_flags (if present) is a dict
  B. Range sanity: numeric quality values within expected bounds; report outliers
  D. Cross-sidecar spot check: 200 random utterance_ids verified against the
     original sidecars (tier1 / vad / lid / dnsmos) — values must match
  E. Audio spot check: list 5 worst + 5 best clips by dnsmos_ovrl (so user
     can listen / inspect)
  F. Stats parity: recompute count + hours from manifest.jsonl, compare with
     stats.json's `manifest.total` section

Exit code 0 if all checks pass; 1 otherwise.

Run:
  /media/cseti/datassd/conda/miniconda3/bin/python bin/validate_manifest_quality.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
MANIFESTS_DIR = DATA_ROOT / "processed" / "manifests"
QUALITY_DIR = DATA_ROOT / "processed" / "quality"

MANIFEST_NAME = "manifest.jsonl"
STATS_NAME = "stats.json"

# (key, lo, hi) — numeric range bounds. Tuned to real-world edge cases:
# -240 dBFS is the digital-silence floor; DNSMOS regression occasionally
# outputs slightly below 1.0; peak dBFS can be slightly positive on clipped audio.
RANGE_CHECKS = [
    ("rms_dbfs", -250.0, 6.0),
    ("peak_dbfs", -250.0, 10.0),
    ("silence_ratio", 0.0, 1.01),
    ("vad_speech_ratio", 0.0, 1.5),         # silero can slightly exceed 1.0
    ("vad_num_segments", 0, 10000),
    ("lid_is_hu_prob", 0.0, 1.001),
    ("lid_top1_score", -1000.0, 1.001),     # log-prob, can be very negative
    ("dnsmos_sig", -1.0, 5.5),
    ("dnsmos_bak", -1.0, 5.5),
    ("dnsmos_ovrl", -1.0, 5.5),
]

SAMPLE_SIZE = 200


def fail(check: str, msg: str) -> None:
    print(f"  [FAIL {check}] {msg}", file=sys.stderr)


def passed(check: str, msg: str) -> None:
    print(f"  [ OK  {check}] {msg}", file=sys.stderr)


def check_a_schema(manifest: Path) -> tuple[bool, int]:
    """Validate every row is parseable + quality_flags is dict if present."""
    ok = True
    n_rows = 0
    n_bad = 0
    with manifest.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            n_rows += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                n_bad += 1
                if n_bad <= 3:
                    fail("A", f"{manifest.name}:{i} JSON error: {e}")
                continue
            qf = r.get("quality_flags")
            if qf is not None and not isinstance(qf, dict):
                n_bad += 1
                if n_bad <= 3:
                    fail("A", f"{manifest.name}:{i} quality_flags not dict")
    if n_bad > 0:
        fail("A", f"{manifest.name}: {n_bad} bad rows")
        ok = False
    else:
        passed("A", f"{manifest.name}: {n_rows:,} rows, all parseable")
    return ok, n_rows


def check_b_ranges(manifest: Path) -> bool:
    """Scan all quality_flags numeric values; report any out of range."""
    ok = True
    out_of_range: dict[str, int] = defaultdict(int)
    examples: dict[str, list] = defaultdict(list)
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qf = r.get("quality_flags") or {}
            for key, lo, hi in RANGE_CHECKS:
                if key not in qf:
                    continue
                v = qf[key]
                if v is None or not isinstance(v, (int, float)):
                    continue
                if v < lo or v > hi:
                    out_of_range[key] += 1
                    if len(examples[key]) < 3:
                        examples[key].append((r["utterance_id"], v))
    if out_of_range:
        for key, n in out_of_range.items():
            lo, hi = next((l, h) for k, l, h in RANGE_CHECKS if k == key)
            fail("B", f"{manifest.name}: {key} OOR ({lo}..{hi}): "
                      f"{n} clips, e.g. {examples[key]}")
        ok = False
    else:
        passed("B", f"{manifest.name}: all numeric quality values in expected ranges")
    return ok


def check_d_spot_vs_sidecars(manifest: Path, sidecar_data: dict,
                             rng: random.Random) -> bool:
    """Sample N rows; for each, look up its utterance_id in every sidecar
    that originally had it; verify the values match."""
    ok = True
    all_ids = []
    rows_by_id = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            all_ids.append(r["utterance_id"])
            rows_by_id[r["utterance_id"]] = r
    sample = rng.sample(all_ids, min(SAMPLE_SIZE, len(all_ids)))
    mismatches = 0
    checked_keys = 0
    for utt in sample:
        row = rows_by_id[utt]
        qf = row.get("quality_flags") or {}
        for tag, sd in sidecar_data.items():
            ref = sd.get(utt)
            if ref is None:
                continue
            for k, v in ref.items():
                if k not in qf:
                    mismatches += 1
                    if mismatches <= 3:
                        fail("D", f"{manifest.name}: {utt} missing {k} from {tag}")
                    continue
                if qf[k] != v:
                    # Float compare: allow tiny rounding diff
                    if isinstance(v, float) and isinstance(qf[k], float):
                        if abs(qf[k] - v) < 1e-6:
                            checked_keys += 1
                            continue
                    mismatches += 1
                    if mismatches <= 3:
                        fail("D", f"{manifest.name}: {utt}.{k} mismatch "
                                  f"({qf[k]} vs sidecar {v})")
                else:
                    checked_keys += 1
    if mismatches:
        fail("D", f"{manifest.name}: {mismatches} mismatches across {SAMPLE_SIZE} samples")
        ok = False
    else:
        passed("D", f"{manifest.name}: {SAMPLE_SIZE} samples × all sidecars, "
                    f"{checked_keys} key matches, 0 mismatches")
    return ok


def check_e_audio_spot(manifest: Path, n: int = 5) -> None:
    """List `n` worst + `n` best clips by dnsmos_ovrl."""
    rows = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qf = r.get("quality_flags") or {}
            ovrl = qf.get("dnsmos_ovrl")
            if isinstance(ovrl, (int, float)):
                rows.append((ovrl, r["utterance_id"], r.get("audio_path"),
                             r.get("source"), qf.get("vad_speech_ratio"),
                             qf.get("lid_top1") or qf.get("lid")))
    rows.sort()
    print(f"\n  [E spot {manifest.name}] worst {n} by dnsmos_ovrl:")
    for ovrl, utt, path, src, vsr, lid in rows[:n]:
        print(f"    OVRL={ovrl:.2f}  vad={vsr}  lid={lid}  src={src}")
        print(f"      {utt}  ->  {path}")
    print(f"  [E spot {manifest.name}] best {n} by dnsmos_ovrl:")
    for ovrl, utt, path, src, vsr, lid in rows[-n:]:
        print(f"    OVRL={ovrl:.2f}  vad={vsr}  lid={lid}  src={src}")
        print(f"      {utt}  ->  {path}")


def check_f_stats_parity(manifest: Path, n_rows: int) -> bool:
    """Recompute count + hours from manifest, compare with stats.json
    (`manifest.total` section)."""
    ok = True
    stats_path = MANIFESTS_DIR / STATS_NAME
    if not stats_path.exists():
        fail("F", f"{stats_path} missing — skipping parity check")
        return False
    stats_all = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats_all.get("schema_version") != 4 or "manifest" not in stats_all:
        fail("F", f"{stats_path.name} is not v4 (schema_version={stats_all.get('schema_version')})")
        return False
    expected = stats_all["manifest"]["total"]
    hours = 0.0
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = r.get("duration_sec")
            if isinstance(d, (int, float)):
                hours += d / 3600
    hours = round(hours, 2)
    exp_count = expected["count"]
    exp_hours = expected["hours"]
    if n_rows != exp_count:
        fail("F", f"{manifest.name}: count mismatch {n_rows:,} vs stats.json {exp_count:,}")
        ok = False
    if abs(hours - exp_hours) > 0.5:  # half-hour tolerance for rounding
        fail("F", f"{manifest.name}: hours mismatch {hours} vs stats.json {exp_hours}")
        ok = False
    if ok:
        passed("F", f"{manifest.name}: count={n_rows:,}, hours={hours} matches stats.json")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    manifest = MANIFESTS_DIR / MANIFEST_NAME
    if not manifest.exists():
        print(f"[error] {manifest} not found", file=sys.stderr)
        return 2

    # Load full sidecar data once for check D
    print("[setup] loading sidecars for cross-check (D)...", file=sys.stderr)
    sidecar_data: dict[str, dict] = {}
    SIDECAR_KEYS = {
        "tier1": ["rms_dbfs", "peak_dbfs", "is_clipped", "silence_ratio"],
        "vad": ["vad_speech_ratio", "vad_num_segments", "vad_speech_sec"],
        "lid": ["lid_top1", "lid_top1_label", "lid_top1_score", "lid_is_hu_prob"],
        "dnsmos": ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl"],
    }
    for tag, fname in [("tier1", "tier1.jsonl"), ("vad", "tier2_vad.jsonl"),
                       ("lid", "tier2_lid.jsonl"), ("dnsmos", "tier2_dnsmos.jsonl")]:
        path = QUALITY_DIR / fname
        if not path.exists():
            sidecar_data[tag] = {}
            continue
        idx: dict[str, dict] = {}
        keys = SIDECAR_KEYS[tag]
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                utt = r.get("utterance_id")
                if not utt:
                    continue
                e = {k: r[k] for k in keys if k in r}
                if e:
                    idx[utt] = e
        sidecar_data[tag] = idx
        print(f"  {tag}: {len(idx):,} rows", file=sys.stderr)

    print(f"\n=== validating {manifest.name} ===", file=sys.stderr)
    a, n_rows = check_a_schema(manifest)
    b = check_b_ranges(manifest)
    d = check_d_spot_vs_sidecars(manifest, sidecar_data, rng)
    check_e_audio_spot(manifest, n=5)
    f = check_f_stats_parity(manifest, n_rows)

    all_ok = a and b and d and f
    print()
    print("=== overall ===" + (" PASSED" if all_ok else " FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
