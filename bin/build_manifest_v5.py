#!/usr/bin/env python3
"""Build a fresh lean manifest v5 from existing sources + voxpopuli_resegmented.

Drops vestigial fields, drops the structurally-broken VoxPopuli mosel
windows, and introduces `voxpopuli_resegmented` (the new Silero-VAD
chunks). Adds an explicit `source_url` field for online provenance.

Schema v5 (lean) vs v4 (current):

Removed fields (mostly null or derivable):
  - parquet_row_index (only voxpopuli_hu_labeled had it; null elsewhere)
  - text_consensus, consensus_method, confidence_level, pairwise_wer
    (Phase 4 placeholders, all null)
  - speaker_id (sparsely populated)
  - segmentation_status (always "segmented_file" now)
  - relative_audio_path (derivable from audio_path)
  - parent_audio_path (heterogeneous: URL for yodas2, path for others;
    replaced by `source_url` + `parent_session_id`)
  - hallucination_flags top-level (kept inside quality_flags only)

Added field:
  - source_url (str | None) — online provenance URL, derived per-source:
      yodas2_hu000          → https://www.youtube.com/watch?v={item}
      librivox_hu           → https://archive.org/details/{item}
      podcasts_hu_cc        → https://archive.org/details/{item}
      voxpopuli_hu_labeled  → https://huggingface.co/datasets/facebook/voxpopuli
      voxpopuli_resegmented → https://dl.fbaipublicfiles.com/voxpopuli/audios/hu_{year}.tar

Sources kept: yodas2_hu000, librivox_hu, podcasts_hu_cc,
              voxpopuli_hu_labeled, voxpopuli_resegmented (new)
Sources dropped: mosel_hu_voxpopuli, mosel_hu_ytc, voxpopuli_unlabeled_gap
                 (their data lives on disk; can be deleted later)

Input:
  processed/manifests/manifest.jsonl     (current v4 manifest)
  processed/normalization/voxpopuli_resegmented.jsonl
                                          (new Silero-VAD chunks)

Output:
  processed/manifests/manifest_v5.jsonl  (lean schema, atomic swap)
  processed/manifests/stats_v5.json      (per-source counts + hours)

Note: voxpopuli_resegmented rows have no `quality_flags` yet — Phase 3
metrics (Tier-1, VAD, DNSMOS, LID) need to be re-run on the new chunks
before the merge step. This script just emits the manifest skeleton.

Run with the base env:
  /media/cseti/datassd/conda/miniconda3/bin/python bin/build_manifest_v5.py
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
MANIFESTS_DIR = DATA_ROOT / "processed" / "manifests"

INPUT_MANIFEST = MANIFESTS_DIR / "manifest.jsonl"
RESEGMENTED_SIDECAR = (
    DATA_ROOT / "processed" / "normalization" / "voxpopuli_resegmented.jsonl"
)
YODAS2_CHUNKED_SIDECAR = (
    DATA_ROOT / "processed" / "normalization" / "yodas2_chunked.jsonl"
)
VP_LABELED_EXTRACTED_SIDECAR = (
    DATA_ROOT / "processed" / "normalization" / "voxpopuli_hu_labeled_extracted.jsonl"
)
OUTPUT_MANIFEST = MANIFESTS_DIR / "manifest_v5.jsonl"
OUTPUT_STATS = MANIFESTS_DIR / "stats_v5.json"

# Sources that survive in v5.
SOURCES_KEEP = {
    "yodas2_hu000",
    "librivox_hu",
    "podcasts_hu_cc",
    "voxpopuli_hu_labeled",
}
NEW_SOURCE = "voxpopuli_resegmented"
CV25_SOURCE = "common_voice_25_0_hu"
CV25_DATA_ROOT = (
    DATA_ROOT / "raw" / CV25_SOURCE / "cv-corpus-25.0-2026-03-09" / "hu"
)

# Sources removed from v5 (data files stay on disk).
SOURCES_DROP = {
    "mosel_hu_voxpopuli",
    "mosel_hu_ytc",
    "voxpopuli_unlabeled_gap",
}

# Fields kept in the lean v5 row.
KEEP_FIELDS = [
    # Identification
    "utterance_id", "source", "source_item_id", "parent_session_id",
    # Audio
    "audio_path", "audio_format", "sample_rate", "channels", "codec",
    "duration_sec", "segment_start_sec", "segment_end_sec",
    "refined_audio_path",
    # Parquet-internal audio decode locator (only populated for
    # voxpopuli_hu_labeled rows; null elsewhere). Restored 2026-05-26
    # because quality_tier1.py + future re-decoders need it to seek into
    # the parquet shard. Dropping it in the initial v5 lean was premature.
    "parquet_row_index",
    # Transcripts
    "transcripts",
    # Metadata
    "language", "license", "license_url", "attribution",
    "domain", "register", "split",
    # Quality flags inline
    "quality_flags",
]


def derive_source_url(source: str, item: str | None,
                      parent_session: str | None) -> str | None:
    """Per-source online URL where the original audio is reachable."""
    if not item:
        item = parent_session or ""
    if source == "yodas2_hu000":
        return f"https://www.youtube.com/watch?v={item}"
    if source == "librivox_hu":
        return f"https://archive.org/details/{item}"
    if source == "podcasts_hu_cc":
        return f"https://archive.org/details/{item}"
    if source == "voxpopuli_hu_labeled":
        return "https://huggingface.co/datasets/facebook/voxpopuli"
    if source == NEW_SOURCE:
        # parent_session_id format: "20090112-0900-PLENARY-10_hu" → year 2009
        year = (parent_session or item)[:4]
        if year.isdigit():
            return f"https://dl.fbaipublicfiles.com/voxpopuli/audios/hu_{year}.tar"
    if source == CV25_SOURCE:
        return "https://commonvoice.mozilla.org/datasets"
    return None


def _vp_labeled_hf_split(audio_path: str | None) -> str | None:
    """Derive HF train/dev/test split from the parquet shard filename.

    `audio_path` for vp_labeled rows points at one of the 6 HF shards:
      test-00000-of-00001.parquet         -> 'test'
      validation-00000-of-00001.parquet   -> 'dev'  (HF calls it 'validation')
      train-NNNNN-of-00004.parquet        -> 'train'
    """
    if not audio_path:
        return None
    fname = audio_path.rsplit("/", 1)[-1].lower()
    if fname.startswith("test-"):
        return "test"
    if fname.startswith("validation-"):
        return "dev"
    if fname.startswith("train-"):
        return "train"
    return None


def lean_row(row: dict,
             yodas2_chunks: dict[str, str] | None = None,
             vp_labeled_extracted: dict[str, str] | None = None) -> dict:
    """Project a v4 row down to the v5 lean schema + add source_url.

    `yodas2_chunks`: optional {utterance_id: chunk_audio_path} map. If the row
    is yodas2 and its uid is in the map, override audio_path to the chunked
    OGG and null out segment_start_sec / segment_end_sec (chunk is a
    standalone file, no longer a virtual segment in a parent WAV).

    `vp_labeled_extracted`: optional {utterance_id: chunk_audio_path} map.
    Same idea for vp_labeled — the audio is extracted out of the parquet
    shard into a standalone OGG. Derive `hf_split` BEFORE overriding the
    audio_path (split lives in the shard filename, which we lose otherwise)."""
    out = {k: row.get(k) for k in KEEP_FIELDS}

    # Always derive hf_split for vp_labeled from the ORIGINAL parquet shard
    # path (this lookup must happen before any audio_path override).
    if out["source"] == "voxpopuli_hu_labeled":
        hf_split = _vp_labeled_hf_split(out.get("audio_path"))
        if hf_split is not None:
            qf = dict(out.get("quality_flags") or {})
            qf["hf_split"] = hf_split
            out["quality_flags"] = qf

    if (yodas2_chunks is not None
            and out["source"] == "yodas2_hu000"
            and out["utterance_id"] in yodas2_chunks):
        out["audio_path"] = yodas2_chunks[out["utterance_id"]]
        out["audio_format"] = "ogg"
        out["codec"] = "ogg"
        out["sample_rate"] = 16000
        out["channels"] = 1
        out["segment_start_sec"] = None
        out["segment_end_sec"] = None

    if (vp_labeled_extracted is not None
            and out["source"] == "voxpopuli_hu_labeled"
            and out["utterance_id"] in vp_labeled_extracted):
        out["audio_path"] = vp_labeled_extracted[out["utterance_id"]]
        out["audio_format"] = "ogg"
        out["codec"] = "ogg"
        out["sample_rate"] = 16000
        out["channels"] = 1
        # parquet_row_index can stay (still valid in the original shard, used
        # by the legacy /audio_parquet curator endpoint and as audit trail).

    out["source_url"] = derive_source_url(
        out["source"], out.get("source_item_id"), out.get("parent_session_id")
    )
    return out


def load_yodas2_chunks(path: Path) -> dict[str, str]:
    """Load yodas2_chunked.jsonl into {utterance_id: chunk_audio_path}."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["utterance_id"]] = r["audio_path"]
    return out


def load_vp_labeled_extracted(path: Path) -> dict[str, str]:
    """Load voxpopuli_hu_labeled_extracted.jsonl into
    {utterance_id: chunk_audio_path}. Same shape as yodas2_chunked."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["utterance_id"]] = r["audio_path"]
    return out


def emit_resegmented_row(rec: dict) -> dict:
    """Convert a `voxpopuli_resegmented.jsonl` sidecar entry to a v5 row.

    The sidecar fields are:
      utterance_id, parent_session_id, chunk_idx, audio_path, audio_format,
      sample_rate, channels, duration_sec, segment_start_sec, segment_end_sec,
      padding_sec

    Phase 3 metrics will be filled in later (via merge_quality_into_manifest.py
    once tier1/vad/dnsmos/lang_purity have been re-run on these clips)."""
    parent = rec.get("parent_session_id")
    # Strip the "_hu" suffix to get the calendar session_item_id
    source_item_id = parent  # parent itself acts as the item id for voxpopuli
    row = {
        "utterance_id": rec["utterance_id"],
        "source": NEW_SOURCE,
        "source_item_id": source_item_id,
        "parent_session_id": parent,
        "audio_path": rec["audio_path"],
        "audio_format": rec.get("audio_format", "ogg"),
        "sample_rate": rec.get("sample_rate", 16000),
        "channels": rec.get("channels", 1),
        "codec": "ogg",
        "duration_sec": rec["duration_sec"],
        "segment_start_sec": rec["segment_start_sec"],
        "segment_end_sec": rec["segment_end_sec"],
        "refined_audio_path": None,
        "parquet_row_index": None,
        "transcripts": {},  # no text yet; Phase 4 consensus will populate
        "language": "hu",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "VoxPopuli HU unlabeled (Facebook AI) — resegmented with Silero VAD",
        "domain": "parliament",
        "register": "formal",
        "split": "train",
        # Empty quality_flags — Phase 3 re-run pending; include the padding
        # metadata as a starting signal so downstream tools can see it.
        "quality_flags": {
            "padding_sec": rec.get("padding_sec"),
        },
    }
    row["source_url"] = derive_source_url(NEW_SOURCE, source_item_id, parent)
    return row


def _cv25_read_tsv_map(path: Path, key_col: str,
                      value_cols: list[str]) -> dict[str, dict[str, str]]:
    """Read a CV25 TSV, return {key_col_value: {col: value}} for value_cols."""
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if key_col not in header:
            return out
        key_idx = header.index(key_col)
        val_idxs = {c: header.index(c) for c in value_cols if c in header}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= key_idx:
                continue
            key = parts[key_idx]
            out[key] = {
                c: (parts[i] if i < len(parts) else "")
                for c, i in val_idxs.items()
            }
    return out


def _cv25_read_tsv_keys(path: Path, key_col: str = "path") -> set[str]:
    """Read a CV25 TSV, return the set of values in `key_col`."""
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        key_idx = header.index(key_col) if key_col in header else 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > key_idx:
                keys.add(parts[key_idx])
    return keys


def iter_cv25_rows(cv25_root: Path):
    """Yield v5 manifest rows for Common Voice 25.0 HU.

    Reads `clip_durations.tsv` to enumerate all clips (~117k), and merges
    the validated/invalidated/other/reported status + train/dev/test split
    assignment + per-clip metadata (sentence, votes, demographics)."""
    clips_dir = cv25_root / "clips"
    durations_path = cv25_root / "clip_durations.tsv"
    if not clips_dir.is_dir() or not durations_path.exists():
        return

    durations: dict[str, int] = {}
    with durations_path.open(encoding="utf-8") as f:
        f.readline()  # header: clip\tduration[ms]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].isdigit():
                durations[parts[0]] = int(parts[1])

    meta_cols = [
        "sentence", "client_id", "up_votes", "down_votes",
        "age", "gender", "accents",
    ]
    validated = _cv25_read_tsv_map(cv25_root / "validated.tsv", "path", meta_cols)
    invalidated = _cv25_read_tsv_map(cv25_root / "invalidated.tsv", "path", meta_cols)
    other = _cv25_read_tsv_map(cv25_root / "other.tsv", "path", meta_cols)
    reported_set = _cv25_read_tsv_keys(cv25_root / "reported.tsv", "path")

    train_set = _cv25_read_tsv_keys(cv25_root / "train.tsv", "path")
    dev_set = _cv25_read_tsv_keys(cv25_root / "dev.tsv", "path")
    test_set = _cv25_read_tsv_keys(cv25_root / "test.tsv", "path")

    def cv25_split_of(clip: str) -> str | None:
        if clip in train_set:
            return "train"
        if clip in dev_set:
            return "dev"
        if clip in test_set:
            return "test"
        return None

    def status_of(clip: str) -> str:
        if clip in validated:
            return "validated"
        if clip in invalidated:
            return "invalidated"
        if clip in other:
            return "other"
        return "unknown"

    def meta_of(clip: str) -> dict[str, str]:
        for src in (validated, invalidated, other):
            if clip in src:
                return src[clip]
        return {}

    def _int(s: str) -> int:
        try:
            return int(s or "0")
        except ValueError:
            return 0

    for clip_filename, duration_ms in durations.items():
        item_stem = clip_filename.rsplit(".", 1)[0]
        meta = meta_of(clip_filename)
        sentence = (meta.get("sentence") or "").strip()
        row = {
            "utterance_id": f"{CV25_SOURCE}/{item_stem}",
            "source": CV25_SOURCE,
            "source_item_id": item_stem,
            "parent_session_id": None,
            "audio_path": str(clips_dir / clip_filename),
            "audio_format": "mp3",
            "sample_rate": 48000,
            "channels": 1,
            "codec": "mp3",
            "duration_sec": duration_ms / 1000.0,
            "segment_start_sec": None,
            "segment_end_sec": None,
            "refined_audio_path": None,
            "parquet_row_index": None,
            "transcripts": ({"source_caption": sentence} if sentence else {}),
            "language": "hu",
            "license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "attribution": "Mozilla Common Voice 25.0 HU (CC0-1.0)",
            "domain": "read_speech",
            "register": "read",
            "split": "train",
            "quality_flags": {
                "cv25_status": status_of(clip_filename),
                "cv25_split": cv25_split_of(clip_filename),
                "cv25_reported": clip_filename in reported_set,
                "up_votes": _int(meta.get("up_votes", "")),
                "down_votes": _int(meta.get("down_votes", "")),
            },
            "source_url": derive_source_url(CV25_SOURCE, item_stem, None),
        }
        yield row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=INPUT_MANIFEST,
                   help="v4 manifest input (default manifest.jsonl)")
    p.add_argument("--resegmented", type=Path, default=RESEGMENTED_SIDECAR,
                   help="voxpopuli resegmented sidecar (default "
                        "voxpopuli_resegmented.jsonl)")
    p.add_argument("--yodas2-chunked", type=Path, default=YODAS2_CHUNKED_SIDECAR,
                   help="yodas2_chunked.jsonl sidecar (output of "
                        "bin/chunk_yodas2.py). If present, yodas2 audio_path "
                        "is rewritten to point at chunked OGG files instead "
                        "of parent WAVs with segment offsets.")
    p.add_argument("--vp-labeled-extracted", type=Path,
                   default=VP_LABELED_EXTRACTED_SIDECAR,
                   help="voxpopuli_hu_labeled_extracted.jsonl sidecar (output "
                        "of bin/extract_vp_labeled.py). If present, vp_labeled "
                        "audio_path is rewritten to the standalone OGG file.")
    p.add_argument("--cv25-root", type=Path, default=CV25_DATA_ROOT,
                   help="Common Voice 25.0 HU dataset root "
                        "(contains validated.tsv, clip_durations.tsv, clips/)")
    p.add_argument("--output", type=Path, default=OUTPUT_MANIFEST,
                   help="output v5 manifest (default manifest_v5.jsonl)")
    p.add_argument("--stats", type=Path, default=OUTPUT_STATS)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".jsonl.tmp")

    print(f"[build] v4 -> v5", file=sys.stderr)
    print(f"  input:       {args.input}", file=sys.stderr)
    print(f"  resegmented: {args.resegmented}", file=sys.stderr)
    print(f"  yodas2 chk:  {args.yodas2_chunked}", file=sys.stderr)
    print(f"  output:      {args.output}", file=sys.stderr)

    yodas2_chunks = load_yodas2_chunks(args.yodas2_chunked)
    if yodas2_chunks:
        print(f"  yodas2 chunks loaded: {len(yodas2_chunks):,}",
              file=sys.stderr)
    else:
        print(f"  yodas2 chunks: none found (yodas2 rows will keep their "
              f"v4 parent-WAV audio_path)", file=sys.stderr)

    vp_labeled_extracted = load_vp_labeled_extracted(args.vp_labeled_extracted)
    if vp_labeled_extracted:
        print(f"  vp_labeled extracted: {len(vp_labeled_extracted):,}",
              file=sys.stderr)
    else:
        print(f"  vp_labeled extracted: none found (vp_labeled rows will keep "
              f"parquet-shard audio_path)", file=sys.stderr)

    by_source: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "hours": 0.0, "with_text": 0}
    )
    t0 = time.time()
    n_kept = 0
    n_dropped = 0

    with tmp.open("w", encoding="utf-8") as out_f:
        # Pass 1: filter and project the existing v4 manifest.
        if args.input.exists():
            print(f"[pass1] filtering existing manifest...", file=sys.stderr)
            with args.input.open(encoding="utf-8") as in_f:
                for line in in_f:
                    row = json.loads(line)
                    src = row.get("source")
                    if src in SOURCES_DROP:
                        n_dropped += 1
                        continue
                    if src not in SOURCES_KEEP:
                        # Unknown source — keep but warn
                        print(f"  [warn] unknown source: {src}", file=sys.stderr)
                    lean = lean_row(row, yodas2_chunks, vp_labeled_extracted)
                    out_f.write(json.dumps(lean, ensure_ascii=False) + "\n")
                    n_kept += 1
                    b = by_source[lean["source"]]
                    b["count"] += 1
                    b["hours"] += float(lean.get("duration_sec") or 0.0) / 3600.0
                    if lean.get("transcripts"):
                        b["with_text"] += 1
                    if n_kept % 500_000 == 0:
                        print(f"  ... {n_kept:,} kept, {n_dropped:,} dropped",
                              file=sys.stderr, flush=True)
            print(f"[pass1] kept {n_kept:,}, dropped {n_dropped:,}",
                  file=sys.stderr)
        else:
            print(f"[pass1] input missing, skipping", file=sys.stderr)

        # Pass 2: emit voxpopuli_resegmented rows from the new sidecar.
        n_new = 0
        if args.resegmented.exists():
            print(f"[pass2] emitting voxpopuli_resegmented rows...",
                  file=sys.stderr)
            with args.resegmented.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    row = emit_resegmented_row(rec)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_new += 1
                    b = by_source[NEW_SOURCE]
                    b["count"] += 1
                    b["hours"] += float(row["duration_sec"] or 0.0) / 3600.0
                    if n_new % 200_000 == 0:
                        print(f"  ... {n_new:,} resegmented rows",
                              file=sys.stderr, flush=True)
            print(f"[pass2] emitted {n_new:,} voxpopuli_resegmented rows",
                  file=sys.stderr)
        else:
            print(f"[pass2] resegmented sidecar missing, skipping",
                  file=sys.stderr)

        # Pass 3: emit common_voice_25_0_hu rows from the raw CV25 layout.
        n_cv25 = 0
        if args.cv25_root.is_dir():
            print(f"[pass3] emitting common_voice_25_0_hu rows...",
                  file=sys.stderr)
            for row in iter_cv25_rows(args.cv25_root):
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_cv25 += 1
                b = by_source[CV25_SOURCE]
                b["count"] += 1
                b["hours"] += float(row.get("duration_sec") or 0.0) / 3600.0
                if row.get("transcripts"):
                    b["with_text"] += 1
                if n_cv25 % 20_000 == 0:
                    print(f"  ... {n_cv25:,} cv25 rows",
                          file=sys.stderr, flush=True)
            print(f"[pass3] emitted {n_cv25:,} common_voice_25_0_hu rows",
                  file=sys.stderr)
        else:
            print(f"[pass3] cv25 root missing, skipping", file=sys.stderr)

    # Atomic rename
    os.replace(tmp, args.output)

    # Stats
    total = {
        "count": sum(b["count"] for b in by_source.values()),
        "hours": round(sum(b["hours"] for b in by_source.values()), 2),
        "with_text": sum(b["with_text"] for b in by_source.values()),
    }
    stats = {
        "schema_version": 5,
        "manifest": {
            "total": total,
            "by_source": {k: {"count": v["count"],
                              "hours": round(v["hours"], 2),
                              "with_text": v["with_text"]}
                          for k, v in by_source.items()},
        },
    }
    args.stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    elapsed = time.time() - t0
    print()
    print(f"=== v5 manifest summary ===")
    print(f"Total rows:   {total['count']:,}")
    print(f"Total hours:  {total['hours']:.2f}")
    print(f"With text:    {total['with_text']:,}")
    print()
    for src in sorted(by_source.keys()):
        b = by_source[src]
        print(f"  {src:28s} {b['count']:>10,}  {b['hours']:>10.2f} h")
    print()
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {args.output}")
    print(f"Stats: {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
