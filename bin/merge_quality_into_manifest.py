#!/usr/bin/env python3
"""Phase 3 finale: merge Tier-1 + Tier-2 quality sidecars into manifest.jsonl.

Reads:
  processed/quality/tier1.jsonl
  processed/quality/tier2_vad.jsonl
  processed/quality/tier2_lid.jsonl
  processed/quality/tier2_dnsmos.jsonl

For every row in
  processed/manifests/manifest.jsonl

inject available quality keys into `quality_flags`. Missing scores are
omitted (not nulled). The new manifest is written to `manifest.jsonl.tmp`
and atomically renamed over `manifest.jsonl` on success, so an interrupted
run never leaves a half-written manifest in place.

Also refreshes `stats.json`: the `manifest.total` and `manifest.by_source`
sections are recomputed from the new manifest; the `sessions` section is
preserved verbatim (sessions never carry quality scores).

Idempotent: re-running rewrites the manifest with the current sidecar values.

Run:
  /media/cseti/datassd/conda/miniconda3/bin/python bin/merge_quality_into_manifest.py
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

MANIFEST = "manifest.jsonl"
STATS_FILE = "stats.json"

# Sidecar locations as paths relative to DATA_ROOT. Most live under
# processed/quality/; the boundary-refined sidecar lives under
# processed/normalization/ because it produced new audio files alongside
# the metadata.
SIDECARS = {
    "tier1": "processed/quality/tier1.jsonl",
    "vad": "processed/quality/tier2_vad.jsonl",
    "lid": "processed/quality/tier2_lid.jsonl",
    "dnsmos": "processed/quality/tier2_dnsmos.jsonl",
    "lang_purity": "processed/quality/clip_language_purity.jsonl",
    "lang_purity_v2": "processed/quality/clip_language_purity_v2.jsonl",
    "boundary_refined": "processed/normalization/mosel_boundary_refined.jsonl",
}

# Keys we copy verbatim from each sidecar into quality_flags. Anything not
# present in a row is just skipped (e.g. error-only rows don't pollute the
# numeric keys).
SIDECAR_KEYS = {
    "tier1": ["rms_dbfs", "peak_dbfs", "is_clipped", "silence_ratio"],
    "vad": ["vad_speech_ratio", "vad_num_segments", "vad_speech_sec", "vad_error"],
    "lid": ["lid_top1", "lid_top1_label", "lid_top1_score", "lid_is_hu_prob",
            "lid_error"],
    "dnsmos": ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "dnsmos_error"],
    "lang_purity": ["whole_clip_top1", "whole_clip_hu_prob",
                    "first_window_top1", "foreign_prefix_sec",
                    "n_non_hu_windows", "whole_non_hu", "lid_audit_error"],
    # LID v2 Pass 1 (2026-05-26+): whole-clip VoxLingua107 on
    # voxpopuli_resegmented + (after smoke harness) all 4 standalone sources.
    "lang_purity_v2": ["whole_clip_top1", "whole_clip_hu_prob",
                       "needs_pass2", "lid_v2_error"],
    # boundary_refined keys that go into quality_flags. The top-level
    # `refined_audio_path` field is handled separately in the merge loop.
    "boundary_refined": ["refined", "new_start_sec", "new_end_sec",
                         "change_start_ms", "change_end_ms",
                         "n_vad_segments", "refine_error"],
}


def load_sidecar(path: Path, keys: list[str]) -> tuple[dict, int, int]:
    """Index utterance_id -> {key: value} from a sidecar JSONL."""
    index: dict[str, dict] = {}
    n_lines = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            utt = r.get("utterance_id")
            if not utt:
                continue
            entry = {k: r[k] for k in keys if k in r}
            if entry:
                index[utt] = entry
    return index, n_lines, len(index)


def _new_source_bucket() -> dict:
    """Per-source aggregator matching the v4 stats schema produced by
    bin/unify_manifests.py. Keys must stay in sync with that script."""
    return {
        "count": 0,
        "hours": 0.0,
        "with_text": 0,
        "too_short": 0,
        "too_long": 0,
        "halluc_flagged": 0,
        "lid_not_hu": 0,
        "with_tier1": 0,
        "with_vad": 0,
        "with_lid": 0,
        "with_dnsmos": 0,
        "is_clipped": 0,
        "silence_high": 0,
        "vad_speech_ratio_lt_0_3": 0,
        "vad_speech_ratio_ge_0_7": 0,
        "dnsmos_ovrl_ge_3": 0,
        "dnsmos_ovrl_lt_2": 0,
        "_dnsmos_sum": 0.0,
        "_dnsmos_n": 0,
    }


def _finalise_bucket(bucket: dict) -> dict:
    n = bucket.pop("_dnsmos_n")
    s = bucket.pop("_dnsmos_sum")
    out = {k: v for k, v in bucket.items()}
    out["hours"] = round(out["hours"], 2)
    if n > 0:
        out["dnsmos_ovrl_mean"] = round(s / n, 3)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DATA_ROOT)
    p.add_argument("--input", type=Path, default=None,
                   help="Manifest JSONL to merge into (default: "
                        "<root>/processed/manifests/manifest.jsonl). "
                        "Pass manifest_v5.jsonl for v5.")
    p.add_argument("--stats", type=Path, default=None,
                   help="Stats JSON output (default: stats.json next to manifest). "
                        "Pass stats_v5.json for v5.")
    args = p.parse_args()

    manifests_dir = args.root / "processed" / "manifests"
    manifest_path = args.input if args.input else (manifests_dir / MANIFEST)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    stats_path = args.stats if args.stats else (manifests_dir / STATS_FILE)

    if not manifest_path.exists():
        print(f"[error] missing {manifest_path}", file=sys.stderr)
        return 2

    print("[load] reading sidecars...", file=sys.stderr)
    sidecar_data: dict[str, dict] = {}
    # Also load the boundary_refined top-level audio paths separately;
    # they don't belong inside quality_flags.
    refined_audio_paths: dict[str, str] = {}
    for tag, relpath in SIDECARS.items():
        path = args.root / relpath
        if not path.exists():
            print(f"  [WARN] missing: {path}", file=sys.stderr)
            sidecar_data[tag] = {}
            continue
        t0 = time.time()
        index, n_lines, n_keep = load_sidecar(path, SIDECAR_KEYS[tag])
        print(f"  {tag:16s} {n_keep:>10,} rows from {n_lines:>10,} lines "
              f"({time.time()-t0:.1f}s)", file=sys.stderr)
        sidecar_data[tag] = index
        # Pick up `refined_audio_path` separately for the boundary sidecar.
        if tag == "boundary_refined":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    utt = r.get("utterance_id")
                    p = r.get("refined_audio_path")
                    if utt and p:
                        refined_audio_paths[utt] = p

    print(f"[merge] {manifest_path.name} -> {tmp_path.name} (atomic swap on success)",
          file=sys.stderr)
    t0 = time.time()
    n_rows = 0
    n_with_any = 0

    per_src: dict[str, dict] = defaultdict(_new_source_bucket)

    with manifest_path.open(encoding="utf-8") as fin, \
            tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            n_rows += 1
            utt = r["utterance_id"]
            src = r.get("source", "unknown")
            qf = r.get("quality_flags") or {}
            added = False

            bucket = per_src[src]
            bucket["count"] += 1
            bucket["hours"] += float(r.get("duration_sec") or 0.0) / 3600.0
            if r.get("transcripts"):
                bucket["with_text"] += 1

            if utt in sidecar_data["tier1"]:
                qf.update(sidecar_data["tier1"][utt])
                bucket["with_tier1"] += 1
                added = True
            if utt in sidecar_data["vad"]:
                qf.update(sidecar_data["vad"][utt])
                bucket["with_vad"] += 1
                added = True
            if utt in sidecar_data["lid"]:
                qf.update(sidecar_data["lid"][utt])
                bucket["with_lid"] += 1
                added = True
            if utt in sidecar_data["dnsmos"]:
                qf.update(sidecar_data["dnsmos"][utt])
                bucket["with_dnsmos"] += 1
                added = True
            if utt in sidecar_data["lang_purity"]:
                qf.update(sidecar_data["lang_purity"][utt])
                added = True
            if utt in sidecar_data["lang_purity_v2"]:
                qf.update(sidecar_data["lang_purity_v2"][utt])
                added = True
            if utt in sidecar_data["boundary_refined"]:
                qf.update(sidecar_data["boundary_refined"][utt])
                added = True
            # Top-level field for the refined audio path (separate from
            # quality_flags; the curator can use this as the audio source
            # when present).
            if utt in refined_audio_paths:
                r["refined_audio_path"] = refined_audio_paths[utt]

            # Counters derived from the final merged quality_flags (so a
            # re-merge picks up corrections from new sidecar values).
            if qf.get("too_short"):
                bucket["too_short"] += 1
            if qf.get("too_long"):
                bucket["too_long"] += 1
            if qf.get("any_hallucination_flag"):
                bucket["halluc_flagged"] += 1
            if qf.get("is_clipped"):
                bucket["is_clipped"] += 1
            sr = qf.get("silence_ratio")
            if isinstance(sr, (int, float)) and sr >= 0.5:
                bucket["silence_high"] += 1
            vsr = qf.get("vad_speech_ratio")
            if isinstance(vsr, (int, float)):
                if vsr < 0.3:
                    bucket["vad_speech_ratio_lt_0_3"] += 1
                if vsr >= 0.7:
                    bucket["vad_speech_ratio_ge_0_7"] += 1
            lid_top1 = qf.get("lid_top1") or qf.get("lid")
            if lid_top1 is not None and lid_top1 != "hu":
                bucket["lid_not_hu"] += 1
            ovrl = qf.get("dnsmos_ovrl")
            if isinstance(ovrl, (int, float)):
                bucket["_dnsmos_sum"] += float(ovrl)
                bucket["_dnsmos_n"] += 1
                if ovrl >= 3.0:
                    bucket["dnsmos_ovrl_ge_3"] += 1
                if ovrl < 2.0:
                    bucket["dnsmos_ovrl_lt_2"] += 1

            if added:
                r["quality_flags"] = qf
                n_with_any += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

            if n_rows % 500_000 == 0:
                print(f"  ... {n_rows:,} rows", file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    print(f"  {n_rows:,} rows ({n_with_any:,} with >=1 quality score), "
          f"{elapsed:.1f}s", file=sys.stderr)

    # Finalise per-source buckets.
    by_source = {src: _finalise_bucket(b) for src, b in per_src.items()}
    manifest_total = {
        "count": sum(b["count"] for b in by_source.values()),
        "hours": round(sum(b["hours"] for b in by_source.values()), 2),
        "with_text": sum(b["with_text"] for b in by_source.values()),
    }
    manifest_total["audio_only"] = manifest_total["count"] - manifest_total["with_text"]

    # Atomic swap (POSIX rename): either the old or the new manifest is
    # visible at all times.
    os.replace(tmp_path, manifest_path)
    print(f"[swap] {tmp_path.name} -> {manifest_path.name}", file=sys.stderr)

    # Refresh stats.json: update manifest.* sections, preserve sessions.
    if stats_path.exists():
        existing = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        existing = {"schema_version": 4, "manifest": {}, "sessions": {}}
    if existing.get("schema_version") != 4:
        print(f"[warn] existing {stats_path.name} is not v4; overwriting with v4 schema",
              file=sys.stderr)
    new_stats = {
        "schema_version": 4,
        "manifest": {
            "total": manifest_total,
            "by_source": by_source,
        },
        "sessions": existing.get("sessions", {}),
    }
    stats_path.write_text(json.dumps(new_stats, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"[done] refreshed {stats_path.name} "
          f"(manifest.total: {manifest_total['count']:,} rows, "
          f"{manifest_total['hours']:.2f} h)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
