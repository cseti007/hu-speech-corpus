#!/usr/bin/env python3
"""Build the unified JSONL manifest (v4 schema) from downloaded sources.

Outputs into processed/manifests/:
  - manifest.jsonl           training-ready rows: transcribed (human text) +
                             pseudo_transcribed (MOSEL Whisper text) +
                             untranscribed_chunks (long-form, chunked audio).
                             Category is derivable per row from `transcripts`
                             (empty = audio-only) and `transcripts.keys()`
                             (which provider supplied the text).
  - manifest_sessions.jsonl  session-level long-form parents (LibriVox chapters,
                             podcast episodes, VoxPopuli unlabeled sessions).
                             Linked to chunk rows via parent_audio_path.
  - stats.json               v4 schema: `manifest.{total,by_source}` and
                             `sessions.{total,by_source}`.

Quality scores (Tier-1 / Tier-2) are NOT written by this script. Run
bin/merge_quality_into_manifest.py afterwards to merge sidecars into
quality_flags and refresh the quality counters in stats.json.

Per-row schema (see notes/MANIFEST_SCHEMA.md for full description):
  Identification: utterance_id, source, source_item_id, parent_session_id
  Audio: audio_path (nullable), audio_format, parquet_row_index (nullable),
         relative_audio_path, sample_rate, channels, codec, duration_sec,
         segment_start_sec, segment_end_sec, parent_audio_path
  Transcripts: transcripts (dict), text_consensus, consensus_method,
               confidence_level, pairwise_wer, hallucination_flags
  Speaker/domain: speaker_id, domain, register, language
  License/provenance: license, license_url, attribution
  Split/quality: split, segmentation_status, quality_flags
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

import yaml
from tqdm import tqdm

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "sources.yaml"
DEFAULT_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")

SHORT_THRESHOLD_SEC = 1.0
LONG_THRESHOLD_SEC = 30.0

# Phase 2.5 normalization sidecars (paths relative to root)
NORM_DIR_REL = "processed/normalization"
YODAS2_MERGED_SIDECAR_REL = f"{NORM_DIR_REL}/yodas2_merged.jsonl"
VP_LABELED_DUR_SIDECAR_REL = f"{NORM_DIR_REL}/voxpopuli_labeled_durations.jsonl"
CHUNKS_LIBRIVOX_SIDECAR_REL = f"{NORM_DIR_REL}/chunks_librivox_hu.jsonl"
CHUNKS_PODCASTS_SIDECAR_REL = f"{NORM_DIR_REL}/chunks_podcasts_hu_cc.jsonl"
CHUNKS_VP_GAP_SIDECAR_REL = f"{NORM_DIR_REL}/chunks_voxpopuli_unlabeled_gap.jsonl"

# Clip duration acceptance window (3-30s, applied post-normalization).
CLIP_MIN_DUR_SEC = 3.0
CLIP_MAX_DUR_SEC = 30.0

LICENSE_URL_TO_SPDX = {
    "creativecommons.org/licenses/by/4.0": "CC-BY-4.0",
    "creativecommons.org/licenses/by/3.0": "CC-BY-3.0",
    "creativecommons.org/licenses/by-sa/4.0": "CC-BY-SA-4.0",
    "creativecommons.org/licenses/by-sa/3.0": "CC-BY-SA-3.0",
    "creativecommons.org/publicdomain/mark/1.0": "PD",
    "creativecommons.org/publicdomain/zero/1.0": "CC0-1.0",
    "creativecommons.org/licenses/publicdomain": "PD",
}


def license_from_url(license_url: str | None, fallback: str) -> str:
    if not license_url:
        return fallback
    url_lower = license_url.lower().rstrip("/")
    for stem, spdx in LICENSE_URL_TO_SPDX.items():
        if stem in url_lower:
            return spdx
    return fallback


@dataclass
class Row:
    utterance_id: str
    source: str
    source_item_id: str | None
    parent_session_id: str | None

    audio_path: str | None
    audio_format: str | None
    parquet_row_index: int | None
    relative_audio_path: str | None
    sample_rate: int | None
    channels: int | None
    codec: str | None
    duration_sec: float | None
    segment_start_sec: float | None
    segment_end_sec: float | None
    parent_audio_path: str | None

    transcripts: dict[str, str]
    text_consensus: str | None
    consensus_method: str | None
    confidence_level: str
    pairwise_wer: dict[str, float] | None
    hallucination_flags: dict[str, Any] | None

    speaker_id: str | None
    domain: str
    register: str
    language: str

    license: str
    license_url: str | None
    attribution: str | None

    split: str
    segmentation_status: str
    quality_flags: dict[str, Any]


def ffprobe_info(path: Path) -> dict:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-select_streams", "a:0",
             "-show_entries", "format=duration",
             "-show_entries", "stream=sample_rate,channels,codec_name",
             "-of", "default=noprint_wrappers=1",
             str(path)],
            timeout=30,
        ).decode().strip().splitlines()
        kv = {}
        for line in out:
            if "=" in line:
                k, _, v = line.partition("=")
                kv[k.strip()] = v.strip()
        return {
            "codec": kv.get("codec_name", ""),
            "sample_rate": int(kv["sample_rate"]) if kv.get("sample_rate", "").isdigit() else 0,
            "channels": int(kv["channels"]) if kv.get("channels", "").isdigit() else 0,
            "duration_sec": float(kv["duration"]) if kv.get("duration") else 0.0,
        }
    except (subprocess.SubprocessError, ValueError, KeyError):
        return {}


def quality_flags_basic(duration_sec: float | None, is_segmented: bool) -> dict:
    if duration_sec is None:
        return {"too_short": False, "too_long": False}
    return {
        "too_short": is_segmented and duration_sec > 0 and duration_sec < SHORT_THRESHOLD_SEC,
        "too_long": is_segmented and duration_sec >= LONG_THRESHOLD_SEC,
    }


def parse_yodas_utterance_id(uid: str) -> tuple[str | None, float | None, float | None]:
    """YODAS uid: <video_id>-<seg>-<start_csec>-<end_csec> (csec = centiseconds)."""
    parts = uid.rsplit("-", 3)
    if len(parts) != 4:
        return None, None, None
    video_id, _seg, start, end = parts
    try:
        return video_id, int(start) / 100.0, int(end) / 100.0
    except ValueError:
        return video_id, None, None


# === YODAS v1 (already segmented) ===

def yodas_v1_rows(root: Path) -> Iterator[Row]:
    base = root / "raw" / "yodas_hu000" / "data" / "hu000"
    text_dir = base / "text"
    dur_dir = base / "duration"
    audio_dir = base / "audio"

    text_map: dict[str, str] = {}
    for f in sorted(text_dir.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                uid, _, transcript = line.partition(" ")
                text_map[uid] = transcript.strip()

    dur_map: dict[str, float] = {}
    for f in sorted(dur_dir.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                uid, _, secs = line.partition(" ")
                try:
                    dur_map[uid] = float(secs)
                except ValueError:
                    pass

    wavs = sorted(audio_dir.rglob("*.wav"))
    for wav in tqdm(wavs, desc="yodas_hu000", unit="utt"):
        uid = wav.stem
        if uid not in text_map:
            continue
        video_id, seg_start, seg_end = parse_yodas_utterance_id(uid)
        duration_sec = dur_map.get(uid, 0.0)
        if duration_sec <= 0 and seg_start is not None and seg_end is not None:
            duration_sec = seg_end - seg_start
        yield Row(
            utterance_id=uid,
            source="yodas_hu000",
            source_item_id=video_id,
            parent_session_id=None,
            audio_path=str(wav),
            audio_format="wav",
            parquet_row_index=None,
            relative_audio_path=str(wav.relative_to(root)),
            sample_rate=16000,
            channels=1,
            codec="wav",
            duration_sec=duration_sec,
            segment_start_sec=seg_start,
            segment_end_sec=seg_end,
            parent_audio_path=(
                f"https://www.youtube.com/watch?v={video_id}" if video_id else None
            ),
            transcripts={"source_caption": text_map[uid]},
            text_consensus=None,
            consensus_method=None,
            confidence_level="UNKNOWN",
            pairwise_wer=None,
            hallucination_flags=None,
            speaker_id=video_id,
            domain="youtube",
            register="unknown",
            language="hu",
            license="CC-BY-3.0",
            license_url="https://creativecommons.org/licenses/by/3.0/",
            attribution=(
                f"YouTube video {video_id} (CC-BY, YODAS hu000 manual caption)"
                if video_id else "YODAS hu000 (CC-BY)"
            ),
            split="train",
            segmentation_status="segmented_file",
            quality_flags=quality_flags_basic(duration_sec, is_segmented=True),
        )


# === YODAS2 — merged 3-30s clips (Phase 2.5 normalization) ===


def _yodas2_video_wav_index(audio_dir: Path) -> dict[str, Path]:
    """Build dict: video_id -> path to its WAV file. YODAS2 extracts all WAVs
    flat into `audio/` (the `.tar.extracted` entries are 0-byte sentinels,
    not subdirectories)."""
    return {wav.stem: wav for wav in audio_dir.glob("*.wav")}


def yodas2_rows(root: Path) -> Iterator[Row]:
    """Emit YODAS2 rows from the merged sidecar (Phase 2.5 normalization).

    The sidecar `processed/normalization/yodas2_merged.jsonl` contains
    pre-merged 3-30s clips. The audio_path here is the parent video WAV, with
    segment_start_sec / segment_end_sec defining the clip window inside it.
    """
    sidecar = root / YODAS2_MERGED_SIDECAR_REL
    if not sidecar.exists():
        print(f"[yodas2_hu000] sidecar missing: {sidecar}", file=sys.stderr)
        return

    audio_dir = root / "raw" / "yodas2_hu000" / "data" / "hu000" / "audio"
    wav_index = _yodas2_video_wav_index(audio_dir)

    n_emitted = 0
    n_dropped_no_wav = 0
    with sidecar.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="yodas2_hu000 (merged)", unit="clip"):
            rec = json.loads(line)
            vid = rec["audio_id"]
            wav = wav_index.get(vid)
            if wav is None:
                n_dropped_no_wav += 1
                continue
            rel = str(wav.relative_to(root))
            dur = rec["duration_sec"]
            yield Row(
                utterance_id=rec["merged_utt_id"],
                source="yodas2_hu000",
                source_item_id=vid,
                parent_session_id=None,
                audio_path=str(wav),
                audio_format="wav",
                parquet_row_index=None,
                relative_audio_path=rel,
                sample_rate=24000,
                channels=1,
                codec="wav",
                duration_sec=dur,
                segment_start_sec=rec["start_sec"],
                segment_end_sec=rec["end_sec"],
                parent_audio_path=f"https://www.youtube.com/watch?v={vid}",
                transcripts={"source_caption": rec["text"]},
                text_consensus=None,
                consensus_method=None,
                confidence_level="UNKNOWN",
                pairwise_wer=None,
                hallucination_flags=None,
                speaker_id=vid,
                domain="youtube",
                register="unknown",
                language="hu",
                license="CC-BY-3.0",
                license_url="https://creativecommons.org/licenses/by/3.0/",
                attribution=f"YouTube video {vid} (CC-BY, YODAS2 hu000)",
                split="train",
                segmentation_status="merged",
                quality_flags={
                    **quality_flags_basic(dur, is_segmented=True),
                    "merged_from": rec["merged_from"],
                    "video_duration_sec": rec["video_duration_sec"],
                },
            )
            n_emitted += 1

    print(f"[yodas2_hu000] emitted {n_emitted:,} merged clips "
          f"(dropped {n_dropped_no_wav} missing-wav)", file=sys.stderr)


# === Archive.org sources (LibriVox, podcasts) ===

ARCHIVE_SRC_DOMAIN = {
    "librivox_hu": ("audiobook", "narrated"),
    "podcasts_hu_cc": ("podcast", "spontaneous"),
}


def archive_org_rows(root: Path, src_key: str, src_cfg: dict,
                    domain: str, register: str) -> Iterator[Row]:
    base = root / src_cfg["path"]
    src_license_fallback = (src_cfg.get("license", "?") or "?").split("(")[0].strip()
    audio_exts = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus"}
    items_by_ident = {it["identifier"]: it for it in src_cfg.get("items", [])}

    files = sorted(p for p in base.rglob("*") if p.suffix.lower() in audio_exts)
    for p in tqdm(files, desc=src_key, unit="file"):
        info = ffprobe_info(p)
        if not info or info.get("duration_sec", 0) <= 0:
            continue
        item_ident = p.parent.name
        item_meta = items_by_ident.get(item_ident, {})
        license_url = item_meta.get("license_url", "")
        title = item_meta.get("title", item_ident)
        yield Row(
            utterance_id=f"{item_ident}/{p.stem}",
            source=src_key,
            source_item_id=item_ident,
            parent_session_id=None,
            audio_path=str(p),
            audio_format=p.suffix.lstrip(".").lower(),
            parquet_row_index=None,
            relative_audio_path=str(p.relative_to(root)),
            sample_rate=info["sample_rate"],
            channels=info["channels"],
            codec=info["codec"],
            duration_sec=info["duration_sec"],
            segment_start_sec=None,
            segment_end_sec=None,
            parent_audio_path=None,
            transcripts={},
            text_consensus=None,
            consensus_method=None,
            confidence_level="UNKNOWN",
            pairwise_wer=None,
            hallucination_flags=None,
            speaker_id=item_ident,
            domain=domain,
            register=register,
            language="hu",
            license=license_from_url(license_url, src_license_fallback),
            license_url=license_url or None,
            attribution=f"{title} (archive.org: {item_ident})",
            split="train",
            segmentation_status="session_level",
            quality_flags=quality_flags_basic(info["duration_sec"], is_segmented=False),
        )


# === VoxPopuli HU labeled (parquet) ===

def _load_voxpopuli_labeled_durations(root: Path) -> dict[str, float]:
    """Load duration_sec by audio_id from the Phase 2.5 sidecar."""
    sidecar = root / VP_LABELED_DUR_SIDECAR_REL
    if not sidecar.exists():
        return {}
    out = {}
    with sidecar.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["audio_id"]] = r["duration_sec"]
    return out


def voxpopuli_labeled_rows(root: Path) -> Iterator[Row]:
    import pyarrow.parquet as pq
    base = root / "raw" / "voxpopuli_hu_labeled" / "hu"
    durations = _load_voxpopuli_labeled_durations(root)
    if durations:
        print(f"[voxpopuli_hu_labeled] loaded {len(durations):,} durations from sidecar",
              file=sys.stderr)

    n_dropped_short = 0
    n_dropped_long = 0
    n_emitted = 0
    for parquet_path in sorted(base.glob("*.parquet")):
        split_name = "train" if "train" in parquet_path.name else (
            "test" if "test" in parquet_path.name else "validation"
        )
        rel = str(parquet_path.relative_to(root))
        table = pq.read_table(parquet_path,
                              columns=["audio_id", "raw_text", "normalized_text",
                                       "gender", "speaker_id", "is_gold_transcript"])
        df = table.to_pandas()
        desc = f"voxpopuli_labeled/{parquet_path.stem}"
        for idx, r in tqdm(df.iterrows(), total=len(df), desc=desc, unit="utt"):
            uid = str(r["audio_id"])
            duration = durations.get(uid)
            # Filter to 3-30s acceptance window (Phase 2.5 normalization rule).
            if duration is not None:
                if duration < CLIP_MIN_DUR_SEC:
                    n_dropped_short += 1
                    continue
                if duration > CLIP_MAX_DUR_SEC:
                    n_dropped_long += 1
                    continue
            parent_session = uid.split("-hu")[0] + "_hu" if "-hu" in uid else None
            yield Row(
                utterance_id=f"voxpopuli_hu_labeled/{uid}",
                source="voxpopuli_hu_labeled",
                source_item_id=str(r.get("speaker_id", "")),
                parent_session_id=parent_session,
                audio_path=str(parquet_path),
                audio_format="parquet",
                parquet_row_index=int(idx),
                relative_audio_path=rel,
                sample_rate=16000,
                channels=1,
                codec="parquet_internal",
                duration_sec=duration,
                segment_start_sec=None,
                segment_end_sec=None,
                parent_audio_path=None,
                transcripts={
                    "source_caption": str(r["raw_text"]),
                    "source_caption_normalized": str(r["normalized_text"]),
                },
                text_consensus=None,
                consensus_method=None,
                confidence_level="HIGH" if bool(r["is_gold_transcript"]) else "UNKNOWN",
                pairwise_wer=None,
                hallucination_flags=None,
                speaker_id=str(r.get("speaker_id", "")),
                domain="parliament",
                register="formal",
                language="hu",
                license="CC0-1.0",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                attribution="European Parliament VoxPopuli HU (Facebook, CC0)",
                split=split_name,
                segmentation_status="segmented_file",
                quality_flags={"too_short": False, "too_long": False,
                               "is_gold_transcript": bool(r["is_gold_transcript"]),
                               "gender": str(r.get("gender", ""))},
            )
            n_emitted += 1

    if n_dropped_short or n_dropped_long:
        print(f"[voxpopuli_hu_labeled] emitted {n_emitted:,}, dropped "
              f"{n_dropped_short} <3s + {n_dropped_long} >30s outliers",
              file=sys.stderr)


# === VoxPopuli HU unlabeled (session-level ogg) ===

def voxpopuli_unlabeled_rows(root: Path) -> Iterator[Row]:
    base = root / "raw" / "voxpopuli_hu_unlabeled" / "raw_audios" / "hu"
    if not base.is_dir():
        return
    ogg_files = sorted(base.rglob("*.ogg"))
    for p in tqdm(ogg_files, desc="voxpopuli_hu_unlabeled", unit="session"):
        info = ffprobe_info(p)
        if not info or info.get("duration_sec", 0) <= 0:
            continue
        session_id = p.stem  # e.g. "20110407-0900-PLENARY-9_hu"
        year = p.parent.name
        yield Row(
            utterance_id=f"voxpopuli_hu_unlabeled/{session_id}",
            source="voxpopuli_hu_unlabeled",
            source_item_id=session_id,
            parent_session_id=session_id,
            audio_path=str(p),
            audio_format="ogg",
            parquet_row_index=None,
            relative_audio_path=str(p.relative_to(root)),
            sample_rate=info["sample_rate"],
            channels=info["channels"],
            codec=info["codec"],
            duration_sec=info["duration_sec"],
            segment_start_sec=None,
            segment_end_sec=None,
            parent_audio_path=None,
            transcripts={},
            text_consensus=None,
            consensus_method=None,
            confidence_level="UNKNOWN",
            pairwise_wer=None,
            hallucination_flags=None,
            speaker_id=None,
            domain="parliament",
            register="formal",
            language="hu",
            license="CC0-1.0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            attribution=f"European Parliament {year} HU (VoxPopuli unlabeled, CC0)",
            split="train",
            segmentation_status="session_level",
            quality_flags={"too_short": False, "too_long": False, "year": year},
        )


# === Untranscribed chunks (Phase 2.5: VAD-chunked long-form audio) ===

# Source-specific metadata for chunk-level untranscribed rows.
CHUNKS_SOURCE_META = {
    "librivox_hu": {
        "sidecar_rel": CHUNKS_LIBRIVOX_SIDECAR_REL,
        "domain": "audiobook",
        "register": "narrated",
        "license": "PD",
        "license_url": "https://creativecommons.org/publicdomain/mark/1.0/",
        "attribution_template": "LibriVox HU chapter (PD, chunked from {parent})",
    },
    "podcasts_hu_cc": {
        "sidecar_rel": CHUNKS_PODCASTS_SIDECAR_REL,
        "domain": "podcast",
        "register": "spontaneous",
        "license": "CC-BY",  # mixed CC-BY/CC0/PD; per-item resolution preserved in raw config
        "license_url": None,
        "attribution_template": "HU podcast (mixed CC, chunked from {parent})",
    },
    "voxpopuli_unlabeled_gap": {
        "sidecar_rel": CHUNKS_VP_GAP_SIDECAR_REL,
        "domain": "parliament",
        "register": "formal",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution_template": "European Parliament HU gap-chunk (VoxPopuli unlabeled, CC0, chunked from {parent})",
    },
}


def untranscribed_chunks_rows(root: Path) -> Iterator[Row]:
    """Emit chunk-level rows from Phase 2.5 VAD-chunking sidecars.

    Three sources feed this manifest:
    - librivox_hu chunks (3-30s VAD slices of audiobook chapters)
    - podcasts_hu_cc chunks (3-30s VAD slices of podcast episodes)
    - voxpopuli_unlabeled_gap chunks (3-30s VAD slices of the regions of
      VoxPopuli HU sessions NOT covered by MOSEL alignment)
    """
    n_emitted_by_source: dict[str, int] = {}
    for source_key, meta in CHUNKS_SOURCE_META.items():
        sidecar = root / meta["sidecar_rel"]
        if not sidecar.exists():
            print(f"[{source_key}] sidecar missing: {sidecar}", file=sys.stderr)
            continue
        n_source = 0
        with sidecar.open(encoding="utf-8") as f:
            for line in tqdm(f, desc=f"{source_key} chunks", unit="chunk"):
                rec = json.loads(line)
                dur = rec["duration_sec"]
                # Sanity: should already be 3-30s by Phase 2.5 construction.
                if dur < CLIP_MIN_DUR_SEC or dur > CLIP_MAX_DUR_SEC:
                    continue
                audio_path = rec["audio_path"]
                parent_file = rec["parent_file_path"]
                # Unique id and parent_session_id for tracking
                if source_key == "voxpopuli_unlabeled_gap":
                    sid = rec["session_id"]
                    item_id = f"{sid}_{rec['chunk_index']:06d}"
                    parent_session = sid
                else:
                    fid = rec["file_id"]
                    item_id = f"{fid}_{rec['chunk_index']:06d}"
                    parent_session = fid
                rel = str(Path(audio_path).relative_to(root))
                attribution = meta["attribution_template"].format(parent=Path(parent_file).name)
                yield Row(
                    utterance_id=f"{source_key}/{item_id}",
                    source=source_key,
                    source_item_id=item_id,
                    parent_session_id=parent_session,
                    audio_path=audio_path,
                    audio_format="ogg",
                    parquet_row_index=None,
                    relative_audio_path=rel,
                    sample_rate=16000,
                    channels=1,
                    codec="vorbis",
                    duration_sec=dur,
                    segment_start_sec=rec["start_sec"],
                    segment_end_sec=rec["end_sec"],
                    parent_audio_path=parent_file,
                    transcripts={},
                    text_consensus=None,
                    consensus_method=None,
                    confidence_level="UNKNOWN",
                    pairwise_wer=None,
                    hallucination_flags=None,
                    speaker_id=parent_session,
                    domain=meta["domain"],
                    register=meta["register"],
                    language="hu",
                    license=meta["license"],
                    license_url=meta["license_url"],
                    attribution=attribution,
                    split="train",
                    segmentation_status="chunked",
                    quality_flags={
                        "too_short": False,
                        "too_long": False,
                        "chunk_index": rec["chunk_index"],
                        "parent_file_duration_sec": rec["parent_file_duration_sec"],
                    },
                )
                n_source += 1
        n_emitted_by_source[source_key] = n_source
        print(f"[{source_key}] emitted {n_source:,} chunked rows", file=sys.stderr)


# === MOSEL pseudo-labels (audio not yet segmented) ===

def _load_voxpopuli_hu_alignment(root: Path) -> dict[str, tuple[float, float]]:
    """Load (start, end) seconds keyed by MOSEL utterance ID from the official
    VoxPopuli unlabelled_v2 alignment TSV. Returns empty dict if not found.
    """
    import gzip
    candidates = [
        root / "raw" / "voxpopuli_hu_unlabeled" / "annotations" / "unlabelled_v2.tsv.gz",
        root / "raw" / "voxpopuli_hu_unlabeled" / "unlabelled_data" / "unlabelled_v2.tsv.gz",
    ]
    tsv = next((p for p in candidates if p.exists()), None)
    if tsv is None:
        return {}
    lookup: dict[str, tuple[float, float]] = {}
    with gzip.open(tsv, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in tqdm(reader, desc="loading voxpopuli alignment", unit="row"):
            event_id = r.get("event_id", "")
            if not event_id.endswith("_hu"):
                continue
            seg_no = r.get("segment_no", "")
            try:
                start = float(r["start"])
                end = float(r["end"])
            except (ValueError, KeyError):
                continue
            lookup[f"{event_id}_{seg_no}"] = (start, end)
    return lookup


def mosel_pseudo_transcribed_rows(root: Path) -> Iterator[Row]:
    base = root / "raw" / "mosel_hu" / "transcripts" / "hu"
    csv.field_size_limit(sys.maxsize)

    # If the VoxPopuli HU session segmentation has completed (sentinel present),
    # we can fill in audio_path + duration for every voxpopuli-derived MOSEL row.
    # See bin/segment_voxpopuli.py and notes/MANIFEST_SCHEMA.md.
    vp_seg_root = (root / "raw" / "voxpopuli_hu_unlabeled" /
                   "unlabelled_data" / "hu")
    vp_seg_complete = (root / "raw" / "voxpopuli_hu_unlabeled" /
                       ".segmentation_complete").exists()
    vp_alignment = _load_voxpopuli_hu_alignment(root) if vp_seg_complete else {}
    if vp_alignment:
        print(f"[mosel] loaded {len(vp_alignment):,} voxpopuli HU alignments",
              file=sys.stderr)

    def yield_tsv(path: Path, source_key: str, attribution: str) -> Iterator[Row]:
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in tqdm(reader, desc=source_key, unit="utt"):
                uid = row.get("id", "")
                if not uid:
                    continue
                # Parent session ID is the "uid without trailing _<segnum>"
                # voxpopuli.tsv: "20160118-0900-PLENARY-10_hu_0" → "20160118-0900-PLENARY-10_hu"
                # ytc.tsv: "VoDy7yMW8tU-425964-240" → ytc YouTube id
                parent_session = uid.rsplit("_", 1)[0] if "_hu_" in uid else uid.split("-")[0]

                hall = {
                    "repeated_ngrams": row.get("hall_repeated_ngrams", "False") == "True",
                    "long_word": row.get("hall_long_word", "False") == "True",
                    "frequent_single_word": row.get("hall_frequent_single_word", "False") == "True",
                }
                lid = row.get("lid", "")
                duration_sec = None
                seg_start = None
                if "duration" in row and row["duration"]:
                    try:
                        duration_sec = float(row["duration"])
                    except ValueError:
                        pass
                if "offset" in row and row["offset"]:
                    try:
                        seg_start = float(row["offset"])
                    except ValueError:
                        pass
                seg_end = (seg_start + duration_sec
                          if seg_start is not None and duration_sec is not None else None)

                # Fill audio_path + override duration/start/end from the
                # official VoxPopuli alignment for voxpopuli-derived rows
                # when segmentation has completed.
                seg_audio_path = None
                seg_relative_path = None
                seg_status = "pending_segmentation"
                if source_key == "mosel_hu_voxpopuli" and vp_seg_complete:
                    year = uid[:4]
                    candidate = vp_seg_root / year / f"{uid}.ogg"
                    seg_audio_path = str(candidate)
                    seg_relative_path = str(candidate.relative_to(root))
                    seg_status = "segmented_file"
                    align = vp_alignment.get(uid)
                    if align is not None:
                        seg_start = align[0]
                        seg_end = align[1]
                        duration_sec = seg_end - seg_start

                # Quality flag: lid-mismatch (not Hungarian)
                qf = {"too_short": False, "too_long": False,
                      "any_hallucination_flag": any(hall.values()),
                      "lid": lid,
                      "lid_is_hu": lid == "hu"}
                yield Row(
                    utterance_id=f"mosel/{uid}",
                    source=source_key,
                    source_item_id=parent_session,
                    parent_session_id=parent_session,
                    audio_path=seg_audio_path,
                    audio_format="ogg" if seg_audio_path else None,
                    parquet_row_index=None,
                    relative_audio_path=seg_relative_path,
                    sample_rate=16000,
                    channels=1,
                    codec="ogg",
                    duration_sec=duration_sec,
                    segment_start_sec=seg_start,
                    segment_end_sec=seg_end,
                    parent_audio_path=(
                        f"raw/voxpopuli_hu_unlabeled/raw_audios/hu/*/{parent_session}.ogg"
                        if "_hu" in parent_session else None
                    ),
                    transcripts={"whisper_large_v3_pseudo": row.get("text", "").strip()},
                    text_consensus=None,
                    consensus_method=None,
                    confidence_level="LOW",  # Whisper-pseudo with halluc flags = LOW
                    pairwise_wer=None,
                    hallucination_flags=hall,
                    speaker_id=None,
                    domain="parliament" if source_key == "mosel_hu_voxpopuli" else "youtube_commons",
                    register="formal" if source_key == "mosel_hu_voxpopuli" else "unknown",
                    language="hu",
                    license="CC-BY-4.0",
                    license_url="https://creativecommons.org/licenses/by/4.0/",
                    attribution=attribution,
                    split="train",
                    segmentation_status=seg_status,
                    quality_flags=qf,
                )

    vp = base / "voxpopuli.tsv"
    ytc = base / "ytc.tsv"
    if vp.is_file():
        yield from yield_tsv(vp, "mosel_hu_voxpopuli",
                             "MOSEL HU Whisper-pseudo-labels for VoxPopuli HU (FBK, CC-BY)")
    if ytc.is_file():
        yield from yield_tsv(ytc, "mosel_hu_ytc",
                             "MOSEL HU Whisper-pseudo-labels for YouTube Commons HU (FBK, CC-BY)")


# === Manifest writers ===

def write_jsonl(rows_iter: Iterator[Row], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for r in rows_iter:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def stats_for_manifest(path: Path) -> dict:
    """Per-source counters for `manifest.jsonl` (post-build, pre-quality-merge).

    Mirrors the field set of bin/merge_quality_into_manifest.py and
    bin/unify_manifests.py so that the v4 `manifest.by_source` shape stays
    stable across the build -> merge_quality pipeline. Quality-coverage
    counters (with_tier1, with_vad, ...) are 0 at this stage; they get
    populated by merge_quality."""
    by_source: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "hours": 0.0, "with_text": 0,
                 "too_short": 0, "too_long": 0,
                 "halluc_flagged": 0, "lid_not_hu": 0}
    )
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = by_source[r["source"]]
            s["count"] += 1
            if r["duration_sec"]:
                s["hours"] += r["duration_sec"] / 3600
            if any(r["transcripts"].values()):
                s["with_text"] += 1
            qf = r.get("quality_flags") or {}
            if qf.get("too_short"):
                s["too_short"] += 1
            if qf.get("too_long"):
                s["too_long"] += 1
            if qf.get("any_hallucination_flag"):
                s["halluc_flagged"] += 1
            lid_top1 = qf.get("lid_top1") or qf.get("lid")
            if lid_top1 is not None and lid_top1 != "hu":
                s["lid_not_hu"] += 1
    total = {
        "count": sum(s["count"] for s in by_source.values()),
        "hours": round(sum(s["hours"] for s in by_source.values()), 2),
        "with_text": sum(s["with_text"] for s in by_source.values()),
    }
    total["audio_only"] = total["count"] - total["with_text"]
    return {"total": total,
            "by_source": {k: {**v, "hours": round(v["hours"], 2)}
                          for k, v in by_source.items()}}


def stats_for_sessions(path: Path) -> dict:
    """Per-source count + hours for `manifest_sessions.jsonl`. Sessions never
    carry quality scores or text, so the bucket only tracks count and hours."""
    by_source: dict[str, dict] = defaultdict(lambda: {"count": 0, "hours": 0.0})
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            s = by_source[r["source"]]
            s["count"] += 1
            if r["duration_sec"]:
                s["hours"] += r["duration_sec"] / 3600
    total = {
        "count": sum(s["count"] for s in by_source.values()),
        "hours": round(sum(s["hours"] for s in by_source.values()), 2),
    }
    return {"total": total,
            "by_source": {k: {**v, "hours": round(v["hours"], 2)}
                          for k, v in by_source.items()}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--skip-mosel", action="store_true",
                   help="Skip the very large MOSEL TSV iteration (saves ~5 min)")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    sources = cfg.get("sources", {})
    manifests_dir = args.root / "processed" / "manifests"

    # Row generators per logical category (kept as separate functions for
    # clarity; they all feed into the single manifest.jsonl below, except
    # session-level untranscribed which goes to manifest_sessions.jsonl).
    def transcribed_rows() -> Iterator[Row]:
        # YODAS2 supersedes YODAS v1 (v1 is a video-level subset of v2 with
        # near-identical utterance sets — see notes/YODAS_v1_v2_analysis.md).
        # Emit v1 only as a fallback when v2 isn't downloaded.
        yodas2_dir = args.root / "raw" / "yodas2_hu000"
        has_yodas2 = (yodas2_dir / ".download_complete").exists()
        if has_yodas2:
            yield from yodas2_rows(args.root)
        elif sources.get("yodas_hu000", {}).get("status", "").startswith("downloaded"):
            print("[transcribed] yodas2 missing, falling back to yodas v1", file=sys.stderr)
            yield from yodas_v1_rows(args.root)
        vp_labeled_dir = args.root / "raw" / "voxpopuli_hu_labeled"
        if (vp_labeled_dir / ".download_complete").exists():
            yield from voxpopuli_labeled_rows(args.root)

    def sessions_rows() -> Iterator[Row]:
        for src_key, (domain, register) in ARCHIVE_SRC_DOMAIN.items():
            src_cfg = sources.get(src_key, {})
            if not src_cfg.get("status", "").startswith("downloaded"):
                continue
            yield from archive_org_rows(args.root, src_key, src_cfg, domain, register)
        vp_un_dir = args.root / "raw" / "voxpopuli_hu_unlabeled"
        if (vp_un_dir / ".download_complete").exists():
            yield from voxpopuli_unlabeled_rows(args.root)

    def manifest_rows() -> Iterator[Row]:
        """All training-ready rows: human-transcribed + Whisper-pseudo + chunked."""
        print("[manifest] streaming transcribed (human text) rows...", file=sys.stderr)
        yield from transcribed_rows()
        if not args.skip_mosel:
            print("[manifest] streaming pseudo_transcribed (MOSEL Whisper) rows...",
                  file=sys.stderr)
            yield from mosel_pseudo_transcribed_rows(args.root)
        print("[manifest] streaming untranscribed_chunks (VAD-chunked long-form) rows...",
              file=sys.stderr)
        yield from untranscribed_chunks_rows(args.root)

    manifest_path = manifests_dir / "manifest.jsonl"
    sessions_path = manifests_dir / "manifest_sessions.jsonl"

    print("[build] writing manifest.jsonl...")
    n_manifest = write_jsonl(manifest_rows(), manifest_path)
    print(f"  {n_manifest:,} rows -> manifest.jsonl")

    print("[build] writing manifest_sessions.jsonl (long-form parent index)...")
    n_sessions = write_jsonl(sessions_rows(), sessions_path)
    print(f"  {n_sessions:,} rows -> manifest_sessions.jsonl")

    stats = {
        "schema_version": 4,
        "manifest": stats_for_manifest(manifest_path),
        "sessions": stats_for_sessions(sessions_path),
    }
    (manifests_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )

    print()
    print("[done] manifests written to:", manifests_dir)
    print("[note] run bin/merge_quality_into_manifest.py to merge quality sidecars",
          "and refresh quality counters in stats.json")
    print()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
