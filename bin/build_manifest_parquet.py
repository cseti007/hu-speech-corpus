#!/usr/bin/env python3
"""Convert manifest.jsonl into a single Parquet file for fast filtering / querying.

Output: processed/manifests/manifest.parquet (~500 MB, snappy-compressed)

The Parquet file flattens the two nested dicts (`transcripts` and
`quality_flags`) into top-level columns so DuckDB / pandas / HF datasets can
query them with plain SQL or column access, without having to JSON-decode
each row. Both the original dicts are also stored verbatim as JSON strings
(`transcripts_json`, `quality_flags_json`) so a roundtrip back to v4 JSONL is
lossless.

Flattening rules:
  transcripts.source_caption                 -> text_source_caption
  transcripts.whisper_large_v3_pseudo        -> text_whisper_large_v3_pseudo
  quality_flags.<key>                        -> qf_<key>
  quality_flags.lid_top1 or .lid (whichever  -> qf_lid (unified language code)
    is present)                                qf_lid_is_hu (bool)

Derived columns added for the curator UI:
  has_text          bool   - True if any value in transcripts dict is non-empty
  text_preview      str    - first 80 chars of the chosen transcript (for the
                             table view), or '' for audio-only rows

Streaming write via pyarrow.parquet.ParquetWriter: batches of 100k rows are
converted to pa.Table and appended, so peak RAM stays under ~1 GB on the full
3.2M-row manifest.

Idempotent: re-running overwrites the output.

Run (uses the dedicated env where pyarrow lives):
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python -u bin/build_manifest_parquet.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
INPUT_NAME = "manifest.jsonl"
OUTPUT_NAME = "manifest.parquet"

BATCH_SIZE = 100_000
TEXT_PREVIEW_CHARS = 80

# Known transcript keys we surface as columns. Anything else is preserved
# inside `transcripts_json` but not promoted to a top-level column.
TRANSCRIPT_KEYS = (
    "source_caption",
    "whisper_large_v3_pseudo",
)

# Known quality_flags keys we surface as columns. Each becomes `qf_<key>`.
# Unknown keys (e.g. source-specific extras like `merged_from` on yodas2)
# remain in `quality_flags_json` only.
QUALITY_FLAG_KEYS = (
    "too_short",
    "too_long",
    "rms_dbfs",
    "peak_dbfs",
    "is_clipped",
    "silence_ratio",
    "vad_speech_ratio",
    "vad_num_segments",
    "vad_speech_sec",
    "dnsmos_sig",
    "dnsmos_bak",
    "dnsmos_ovrl",
    "any_hallucination_flag",
    # Phase 2.6 language purity (from bin/audit_clip_language_purity.py)
    "foreign_prefix_sec",
    "whole_non_hu",
    "whole_clip_top1",
    "whole_clip_hu_prob",
    "first_window_top1",
    "n_non_hu_windows",
    # Phase 2.6 boundary refinement (from bin/refine_mosel_boundaries.py)
    "refined",
    "change_start_ms",
    "change_end_ms",
    "new_start_sec",
    "new_end_sec",
    "n_vad_segments",
)

# LID is messy because pseudo rows use `lid` / `lid_is_hu` while yodas2 /
# chunks use `lid_top1` / `lid_top1_score` / `lid_is_hu_prob`. Unify into
# qf_lid (string code) and qf_lid_is_hu_prob (float 0..1) for the curator UI.


def _coalesce(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _text_preview(transcripts: dict) -> str:
    """Pick the first non-empty transcript value, truncate to TEXT_PREVIEW_CHARS."""
    for key in TRANSCRIPT_KEYS:
        v = transcripts.get(key)
        if v:
            s = str(v).strip().replace("\n", " ")
            if len(s) > TEXT_PREVIEW_CHARS:
                s = s[: TEXT_PREVIEW_CHARS - 1] + "…"
            return s
    # Fallback: any non-key transcript that happens to be there
    for v in transcripts.values():
        if v:
            s = str(v).strip().replace("\n", " ")
            if len(s) > TEXT_PREVIEW_CHARS:
                s = s[: TEXT_PREVIEW_CHARS - 1] + "…"
            return s
    return ""


def flatten_row(row: dict) -> dict:
    """Convert one manifest.jsonl row dict into the flat Parquet schema dict."""
    transcripts = row.get("transcripts") or {}
    quality_flags = row.get("quality_flags") or {}
    hallucination_flags = row.get("hallucination_flags")
    pairwise_wer = row.get("pairwise_wer")

    out: dict[str, Any] = {
        # Identification
        "utterance_id": row.get("utterance_id"),
        "source": row.get("source"),
        "source_item_id": row.get("source_item_id"),
        "parent_session_id": row.get("parent_session_id"),

        # Audio
        "audio_path": row.get("audio_path"),
        "audio_format": row.get("audio_format"),
        "parquet_row_index": row.get("parquet_row_index"),
        "relative_audio_path": row.get("relative_audio_path"),
        "sample_rate": row.get("sample_rate"),
        "channels": row.get("channels"),
        "codec": row.get("codec"),
        "duration_sec": row.get("duration_sec"),
        "segment_start_sec": row.get("segment_start_sec"),
        "segment_end_sec": row.get("segment_end_sec"),
        "parent_audio_path": row.get("parent_audio_path"),

        # Transcripts (flattened + preview)
        "has_text": bool(any(transcripts.values())) if transcripts else False,
        "text_source_caption": transcripts.get("source_caption"),
        "text_whisper_large_v3_pseudo": transcripts.get("whisper_large_v3_pseudo"),
        "text_preview": _text_preview(transcripts),
        "text_consensus": row.get("text_consensus"),
        "consensus_method": row.get("consensus_method"),
        "confidence_level": row.get("confidence_level"),
        "transcripts_json": json.dumps(transcripts, ensure_ascii=False, sort_keys=True),

        # Speaker / domain
        "speaker_id": row.get("speaker_id"),
        "domain": row.get("domain"),
        "register": row.get("register"),
        "language": row.get("language"),

        # License / provenance
        "license": row.get("license"),
        "license_url": row.get("license_url"),
        "attribution": row.get("attribution"),

        # Split / segmentation
        "split": row.get("split"),
        "segmentation_status": row.get("segmentation_status"),

        # Phase 2.6 boundary refinement: top-level path to the cleaned OGG
        "refined_audio_path": row.get("refined_audio_path"),
    }

    # Quality flags — promote known keys to qf_* columns
    for key in QUALITY_FLAG_KEYS:
        out[f"qf_{key}"] = quality_flags.get(key)

    # LID unification: two field schemes coexist (lid_top1 vs lid).
    out["qf_lid"] = _coalesce(quality_flags.get("lid_top1"), quality_flags.get("lid"))
    out["qf_lid_label"] = quality_flags.get("lid_top1_label")
    out["qf_lid_score"] = quality_flags.get("lid_top1_score")
    # lid_is_hu_prob (float 0..1) is the canonical signal; lid_is_hu is its
    # boolean shorthand on pseudo rows. Promote whichever is present to a
    # float in qf_lid_is_hu_prob so the UI can range-filter on a single col.
    lid_is_hu_prob = quality_flags.get("lid_is_hu_prob")
    if lid_is_hu_prob is None:
        lid_is_hu = quality_flags.get("lid_is_hu")
        if lid_is_hu is True:
            lid_is_hu_prob = 1.0
        elif lid_is_hu is False:
            lid_is_hu_prob = 0.0
    out["qf_lid_is_hu_prob"] = lid_is_hu_prob

    # Preserve the full nested dicts as JSON strings for lossless roundtrip
    out["quality_flags_json"] = json.dumps(quality_flags, ensure_ascii=False, sort_keys=True)
    out["hallucination_flags_json"] = (
        json.dumps(hallucination_flags, ensure_ascii=False, sort_keys=True)
        if hallucination_flags is not None else None
    )
    out["pairwise_wer_json"] = (
        json.dumps(pairwise_wer, ensure_ascii=False, sort_keys=True)
        if pairwise_wer is not None else None
    )

    return out


def build_schema() -> pa.Schema:
    """Explicit pyarrow schema so empty / missing values pick up the right dtype
    (pandas inference can be wrong on sparse columns like qf_dnsmos_*)."""
    return pa.schema([
        ("utterance_id", pa.string()),
        ("source", pa.string()),
        ("source_item_id", pa.string()),
        ("parent_session_id", pa.string()),
        ("audio_path", pa.string()),
        ("audio_format", pa.string()),
        ("parquet_row_index", pa.int64()),
        ("relative_audio_path", pa.string()),
        ("sample_rate", pa.int64()),
        ("channels", pa.int64()),
        ("codec", pa.string()),
        ("duration_sec", pa.float64()),
        ("segment_start_sec", pa.float64()),
        ("segment_end_sec", pa.float64()),
        ("parent_audio_path", pa.string()),
        ("has_text", pa.bool_()),
        ("text_source_caption", pa.string()),
        ("text_whisper_large_v3_pseudo", pa.string()),
        ("text_preview", pa.string()),
        ("text_consensus", pa.string()),
        ("consensus_method", pa.string()),
        ("confidence_level", pa.string()),
        ("transcripts_json", pa.string()),
        ("speaker_id", pa.string()),
        ("domain", pa.string()),
        ("register", pa.string()),
        ("language", pa.string()),
        ("license", pa.string()),
        ("license_url", pa.string()),
        ("attribution", pa.string()),
        ("split", pa.string()),
        ("segmentation_status", pa.string()),
        ("qf_too_short", pa.bool_()),
        ("qf_too_long", pa.bool_()),
        ("qf_rms_dbfs", pa.float64()),
        ("qf_peak_dbfs", pa.float64()),
        ("qf_is_clipped", pa.bool_()),
        ("qf_silence_ratio", pa.float64()),
        ("qf_vad_speech_ratio", pa.float64()),
        ("qf_vad_num_segments", pa.int64()),
        ("qf_vad_speech_sec", pa.float64()),
        ("qf_dnsmos_sig", pa.float64()),
        ("qf_dnsmos_bak", pa.float64()),
        ("qf_dnsmos_ovrl", pa.float64()),
        ("qf_any_hallucination_flag", pa.bool_()),
        ("qf_lid", pa.string()),
        ("qf_lid_label", pa.string()),
        ("qf_lid_score", pa.float64()),
        ("qf_lid_is_hu_prob", pa.float64()),
        # Phase 2.6 language purity
        ("qf_foreign_prefix_sec", pa.float64()),
        ("qf_whole_non_hu", pa.bool_()),
        ("qf_whole_clip_top1", pa.string()),
        ("qf_whole_clip_hu_prob", pa.float64()),
        ("qf_first_window_top1", pa.string()),
        ("qf_n_non_hu_windows", pa.int64()),
        # Phase 2.6 boundary refinement
        ("refined_audio_path", pa.string()),
        ("qf_refined", pa.bool_()),
        ("qf_change_start_ms", pa.float64()),
        ("qf_change_end_ms", pa.float64()),
        ("qf_new_start_sec", pa.float64()),
        ("qf_new_end_sec", pa.float64()),
        ("qf_n_vad_segments", pa.int64()),
        ("quality_flags_json", pa.string()),
        ("hallucination_flags_json", pa.string()),
        ("pairwise_wer_json", pa.string()),
    ])


def stream_rows(path: Path) -> Iterator[dict]:
    """Yield one flat dict per line of the input JSONL."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield flatten_row(json.loads(line))


def batched(it: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="Data root (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N rows (for testing).")
    args = ap.parse_args()

    manifests_dir = args.root / "processed" / "manifests"
    in_path = manifests_dir / INPUT_NAME
    out_path = manifests_dir / OUTPUT_NAME
    tmp_path = manifests_dir / (OUTPUT_NAME + ".tmp")

    if not in_path.exists():
        print(f"[error] input not found: {in_path}", file=sys.stderr)
        return 2

    schema = build_schema()
    print(f"[parquet] {in_path.name} -> {tmp_path.name}", file=sys.stderr)
    print(f"[parquet] schema: {len(schema.names)} columns", file=sys.stderr)
    t0 = time.time()
    n_total = 0
    n_batches = 0

    with pq.ParquetWriter(tmp_path, schema, compression="snappy") as writer:
        rows = stream_rows(in_path)
        if args.limit is not None:
            def take(it, n):
                for i, x in enumerate(it):
                    if i >= n:
                        return
                    yield x
            rows = take(rows, args.limit)
        for batch in batched(rows, BATCH_SIZE):
            table = pa.Table.from_pylist(batch, schema=schema)
            writer.write_table(table)
            n_total += len(batch)
            n_batches += 1
            elapsed = time.time() - t0
            rate = n_total / elapsed if elapsed > 0 else 0
            print(f"  ... {n_total:>9,} rows ({n_batches} batches, {rate:,.0f} rows/s)",
                  file=sys.stderr, flush=True)

    # Atomic swap
    tmp_path.replace(out_path)
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[done] wrote {n_total:,} rows to {out_path.name} "
          f"({size_mb:.1f} MB, {elapsed:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
