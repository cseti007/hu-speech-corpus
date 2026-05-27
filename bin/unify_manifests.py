#!/usr/bin/env python3
"""Unify the v3 quality-merged manifests into a single v4 manifest.jsonl.

Reads from processed/manifests/:
  - train_transcribed_with_quality.jsonl       (utterance rows, human text)
  - train_pseudo_transcribed_with_quality.jsonl (utterance rows, Whisper pseudo text)
  - train_untranscribed_chunks_with_quality.jsonl (chunked rows, audio-only)
  - train_untranscribed.jsonl                  (session-level long-form, audio-only)

Writes into processed/manifests/:
  - manifest.jsonl           training-ready rows from the three "_with_quality" files
                             concatenated in the order [transcribed, pseudo, chunks]
  - manifest_sessions.jsonl  session-level rows (parent metadata, no quality scores)
  - stats.json               v4 schema: aggregate counts and per-source breakdown for
                             both manifest.jsonl and manifest_sessions.jsonl

The categorisation (transcribed / pseudo / untranscribed) is intentionally NOT stored
as a field on the rows. It is derivable: `bool(row["transcripts"])` distinguishes
audio-only from text-bearing rows, and `transcripts.keys()` identifies the provider.

Idempotent. Re-running overwrites the outputs.

Sanity checks (compared against the legacy stats.json + stats_with_quality.json):
  - Row count of manifest.jsonl equals sum of transcribed + pseudo + chunks legacy counts
  - Sum of duration_sec in manifest.jsonl matches the legacy hours within 0.05h tolerance
  - Row count of manifest_sessions.jsonl matches legacy untranscribed.count
  - Per-source counts match the legacy by_source counts

Usage:
  /media/cseti/datassd/conda/miniconda3/bin/python -u bin/unify_manifests.py
  /media/cseti/datassd/conda/miniconda3/bin/python -u bin/unify_manifests.py --root /custom/data/root
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")

# (input filename, label for reporting). Order is preserved in the output.
QUALITY_INPUTS = [
    ("train_transcribed_with_quality.jsonl", "transcribed"),
    ("train_pseudo_transcribed_with_quality.jsonl", "pseudo_transcribed"),
    ("train_untranscribed_chunks_with_quality.jsonl", "untranscribed_chunks"),
]

SESSIONS_INPUT = "train_untranscribed.jsonl"

OUTPUT_MANIFEST = "manifest.jsonl"
OUTPUT_SESSIONS = "manifest_sessions.jsonl"
OUTPUT_STATS = "stats.json"

# Legacy stats files (used only for validation; remain in place after this step).
LEGACY_STATS = "stats.json.legacy_v3"
LEGACY_QUALITY_STATS = "stats_with_quality.json.legacy_v3"


def _new_source_bucket() -> dict:
    """Per-source aggregator. Mirrors the keys used in the legacy stats_with_quality.json,
    so the v4 stats stay compatible with existing dashboards / STATS.md."""
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
        "_dnsmos_ovrl_sum": 0.0,
        "_dnsmos_ovrl_n": 0,
    }


def _aggregate_row(bucket: dict, row: dict) -> None:
    """Update `bucket` with stats from a single manifest row."""
    bucket["count"] += 1
    bucket["hours"] += float(row.get("duration_sec") or 0.0) / 3600.0
    if row.get("transcripts"):
        bucket["with_text"] += 1

    q = row.get("quality_flags") or {}

    if q.get("too_short"):
        bucket["too_short"] += 1
    if q.get("too_long"):
        bucket["too_long"] += 1

    # Hallucination flag: pseudo rows have hallucination_flags dict, plus a
    # convenience `any_hallucination_flag` aggregated into quality_flags.
    if q.get("any_hallucination_flag"):
        bucket["halluc_flagged"] += 1

    # LID: two field naming conventions exist in v3 manifests.
    #   - yodas2 / chunks: lid_top1, lid_is_hu_prob
    #   - pseudo: lid, lid_is_hu
    lid_top1 = q.get("lid_top1")
    lid_simple = q.get("lid")
    if lid_top1 is not None:
        bucket["with_lid"] += 1
        if lid_top1 != "hu":
            bucket["lid_not_hu"] += 1
    elif lid_simple is not None:
        bucket["with_lid"] += 1
        if lid_simple != "hu":
            bucket["lid_not_hu"] += 1

    # Tier-1 / Tier-2 presence flags.
    if "rms_dbfs" in q:
        bucket["with_tier1"] += 1
    if "vad_speech_ratio" in q:
        bucket["with_vad"] += 1
    if "dnsmos_ovrl" in q:
        bucket["with_dnsmos"] += 1

    if q.get("is_clipped"):
        bucket["is_clipped"] += 1

    silence_ratio = q.get("silence_ratio")
    if silence_ratio is not None and silence_ratio >= 0.5:
        bucket["silence_high"] += 1

    vad = q.get("vad_speech_ratio")
    if vad is not None:
        if vad < 0.3:
            bucket["vad_speech_ratio_lt_0_3"] += 1
        if vad >= 0.7:
            bucket["vad_speech_ratio_ge_0_7"] += 1

    dnsmos = q.get("dnsmos_ovrl")
    if dnsmos is not None:
        if dnsmos >= 3.0:
            bucket["dnsmos_ovrl_ge_3"] += 1
        if dnsmos < 2.0:
            bucket["dnsmos_ovrl_lt_2"] += 1
        bucket["_dnsmos_ovrl_sum"] += float(dnsmos)
        bucket["_dnsmos_ovrl_n"] += 1


def _finalise_bucket(bucket: dict) -> dict:
    """Round floats and compute derived means; drop private accumulators."""
    n = bucket.pop("_dnsmos_ovrl_n")
    s = bucket.pop("_dnsmos_ovrl_sum")
    out = {k: v for k, v in bucket.items()}
    out["hours"] = round(out["hours"], 2)
    if n > 0:
        out["dnsmos_ovrl_mean"] = round(s / n, 3)
    return out


def stream_concat(inputs: list[Path], output: Path) -> tuple[int, dict]:
    """Read each input JSONL line-by-line, write to `output`, and aggregate per-source
    stats into a dict. Returns (total_rows, per_source_buckets)."""
    by_source: dict[str, dict] = defaultdict(_new_source_bucket)
    total = 0
    with output.open("w", encoding="utf-8") as out_f:
        for in_path in inputs:
            print(f"[concat] reading {in_path.name}", file=sys.stderr, flush=True)
            with in_path.open(encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    out_f.write(line)
                    row = json.loads(line)
                    src = row.get("source") or "_unknown"
                    _aggregate_row(by_source[src], row)
                    total += 1
                    if total % 500_000 == 0:
                        print(f"  ... {total:,} rows", file=sys.stderr, flush=True)
    return total, {k: _finalise_bucket(v) for k, v in by_source.items()}


def stream_sessions(input_path: Path, output: Path) -> tuple[int, dict]:
    """Sessions have no quality scores; track only count + hours per source."""
    by_source: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "hours": 0.0}
    )
    total = 0
    with output.open("w", encoding="utf-8") as out_f:
        with input_path.open(encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                out_f.write(line)
                row = json.loads(line)
                src = row.get("source") or "_unknown"
                by_source[src]["count"] += 1
                by_source[src]["hours"] += float(row.get("duration_sec") or 0.0) / 3600.0
                total += 1
    out_by_source = {
        k: {"count": v["count"], "hours": round(v["hours"], 2)}
        for k, v in by_source.items()
    }
    return total, out_by_source


def aggregate_total(by_source: dict[str, dict]) -> dict:
    """Sum the per-source buckets into a single total dict."""
    total = {"count": 0, "hours": 0.0, "with_text": 0}
    for src_stats in by_source.values():
        total["count"] += src_stats.get("count", 0)
        total["hours"] += src_stats.get("hours", 0.0)
        total["with_text"] += src_stats.get("with_text", 0)
    total["hours"] = round(total["hours"], 2)
    total["audio_only"] = total["count"] - total["with_text"]
    return total


def validate_against_legacy(
    manifests_dir: Path,
    manifest_total_rows: int,
    manifest_total_hours: float,
    manifest_by_source: dict,
    sessions_total_rows: int,
    sessions_by_source: dict,
) -> list[str]:
    """Compare new stats against the legacy stats.json + stats_with_quality.json.
    Returns a list of human-readable mismatch messages (empty list = all good)."""
    issues: list[str] = []
    # Prefer the renamed legacy backup (present on re-runs); fall back to stats.json
    # on first run (still in v3 schema before we overwrite it).
    legacy_stats_path = manifests_dir / LEGACY_STATS
    if not legacy_stats_path.exists():
        legacy_stats_path = manifests_dir / "stats.json"
    if not legacy_stats_path.exists():
        issues.append(f"legacy stats not found (looked for {LEGACY_STATS} and stats.json)")
        return issues

    legacy = json.loads(legacy_stats_path.read_text(encoding="utf-8"))
    if legacy.get("schema_version") == 4:
        issues.append(
            f"{legacy_stats_path.name} is already v4 and no legacy backup exists; "
            "cannot validate against original counts"
        )
        return issues

    # Sum expected counts and hours for the unified manifest:
    expected_count = (
        legacy["transcribed"]["total"]["count"]
        + legacy["pseudo_transcribed"]["total"]["count"]
        + legacy["untranscribed_chunks"]["total"]["count"]
    )
    expected_hours = (
        legacy["transcribed"]["total"]["hours"]
        + legacy["pseudo_transcribed"]["total"]["hours"]
        + legacy["untranscribed_chunks"]["total"]["hours"]
    )

    if manifest_total_rows != expected_count:
        issues.append(
            f"manifest.jsonl row count mismatch: got {manifest_total_rows:,}, "
            f"expected {expected_count:,}"
        )
    if abs(manifest_total_hours - expected_hours) > 0.05:
        issues.append(
            f"manifest.jsonl hours mismatch: got {manifest_total_hours:.2f}, "
            f"expected {expected_hours:.2f}"
        )

    if sessions_total_rows != legacy["untranscribed"]["total"]["count"]:
        issues.append(
            f"manifest_sessions.jsonl row count mismatch: got {sessions_total_rows:,}, "
            f"expected {legacy['untranscribed']['total']['count']:,}"
        )

    # Per-source row count check (manifest.jsonl):
    legacy_per_source: dict[str, int] = {}
    for cat in ("transcribed", "pseudo_transcribed", "untranscribed_chunks"):
        for src, s in legacy[cat]["by_source"].items():
            legacy_per_source[src] = legacy_per_source.get(src, 0) + s["count"]
    for src, expected in legacy_per_source.items():
        got = manifest_by_source.get(src, {}).get("count", 0)
        if got != expected:
            issues.append(
                f"manifest.jsonl source '{src}': got {got:,}, expected {expected:,}"
            )

    # Per-source row count check (sessions):
    for src, s in legacy["untranscribed"]["by_source"].items():
        got = sessions_by_source.get(src, {}).get("count", 0)
        if got != s["count"]:
            issues.append(
                f"manifest_sessions.jsonl source '{src}': got {got:,}, expected {s['count']:,}"
            )

    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="Data root (default: %(default)s)")
    args = ap.parse_args()

    manifests_dir = args.root / "processed" / "manifests"
    if not manifests_dir.is_dir():
        print(f"[error] manifests dir not found: {manifests_dir}", file=sys.stderr)
        return 2

    # Verify all inputs exist before we start writing anything.
    quality_inputs: list[Path] = []
    for fname, _label in QUALITY_INPUTS:
        p = manifests_dir / fname
        if not p.exists():
            print(f"[error] missing input: {p}", file=sys.stderr)
            return 2
        quality_inputs.append(p)
    sessions_input = manifests_dir / SESSIONS_INPUT
    if not sessions_input.exists():
        print(f"[error] missing input: {sessions_input}", file=sys.stderr)
        return 2

    manifest_out = manifests_dir / OUTPUT_MANIFEST
    sessions_out = manifests_dir / OUTPUT_SESSIONS
    stats_out = manifests_dir / OUTPUT_STATS

    print(f"[unify] manifests_dir = {manifests_dir}")
    print(f"[unify] writing {manifest_out.name} + {sessions_out.name} + {stats_out.name}")

    # Pass 1: build manifest.jsonl
    manifest_rows, manifest_by_source = stream_concat(quality_inputs, manifest_out)
    print(f"[unify] manifest.jsonl rows: {manifest_rows:,}")

    # Pass 2: build manifest_sessions.jsonl
    sessions_rows, sessions_by_source = stream_sessions(sessions_input, sessions_out)
    print(f"[unify] manifest_sessions.jsonl rows: {sessions_rows:,}")

    # Aggregate totals
    manifest_total = aggregate_total(manifest_by_source)
    sessions_total = {
        "count": sum(s["count"] for s in sessions_by_source.values()),
        "hours": round(sum(s["hours"] for s in sessions_by_source.values()), 2),
    }

    # Validate against legacy stats (legacy stats.json is still in place).
    issues = validate_against_legacy(
        manifests_dir,
        manifest_total["count"], manifest_total["hours"], manifest_by_source,
        sessions_total["count"], sessions_by_source,
    )

    # Build v4 stats.json content.
    new_stats = {
        "schema_version": 4,
        "manifest": {
            "total": manifest_total,
            "by_source": manifest_by_source,
        },
        "sessions": {
            "total": sessions_total,
            "by_source": sessions_by_source,
        },
    }

    # Preserve the legacy stats files before overwriting (one-time backup).
    legacy_backup = manifests_dir / LEGACY_STATS
    legacy_qbackup = manifests_dir / LEGACY_QUALITY_STATS
    if not legacy_backup.exists():
        (manifests_dir / "stats.json").rename(legacy_backup)
        print(f"[unify] preserved legacy stats.json -> {legacy_backup.name}")
    if (manifests_dir / "stats_with_quality.json").exists() and not legacy_qbackup.exists():
        (manifests_dir / "stats_with_quality.json").rename(legacy_qbackup)
        print(f"[unify] preserved legacy stats_with_quality.json -> {legacy_qbackup.name}")

    stats_out.write_text(json.dumps(new_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[unify] wrote {stats_out.name}")

    # Report.
    print()
    print("=" * 70)
    print("UNIFICATION SUMMARY")
    print("=" * 70)
    print(f"  manifest.jsonl:           {manifest_total['count']:>11,} rows  "
          f"{manifest_total['hours']:>10.2f} h")
    print(f"    with_text:              {manifest_total['with_text']:>11,} rows")
    print(f"    audio_only:             {manifest_total['audio_only']:>11,} rows")
    print(f"  manifest_sessions.jsonl:  {sessions_total['count']:>11,} rows  "
          f"{sessions_total['hours']:>10.2f} h")
    print()
    print("Per-source (manifest.jsonl):")
    for src in sorted(manifest_by_source.keys()):
        s = manifest_by_source[src]
        print(f"  {src:<30} {s['count']:>11,} rows  {s['hours']:>10.2f} h  "
              f"with_text={s['with_text']:>11,}")
    print()
    print("Per-source (manifest_sessions.jsonl):")
    for src in sorted(sessions_by_source.keys()):
        s = sessions_by_source[src]
        print(f"  {src:<30} {s['count']:>11,} rows  {s['hours']:>10.2f} h")
    print()
    if issues:
        print("[!] VALIDATION ISSUES:")
        for msg in issues:
            print(f"    - {msg}")
        print()
        print("[!] outputs were still written, but the numbers do not match legacy stats.")
        return 1
    print("[+] validation passed: all row counts and totals match legacy stats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
