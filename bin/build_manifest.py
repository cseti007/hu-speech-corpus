#!/usr/bin/env python3
"""Build unified JSONL manifests (v3 schema) from downloaded sources.

Outputs into processed/manifests/:
  - train_aligned.jsonl       audio + transcript ready
  - train_unaligned.jsonl     long-form audio, no transcript
  - train_pseudo_labeled.jsonl  MOSEL Whisper-pseudo, audio_path=null until segmentation
  - stats.json                per-source counts and totals

Schema v3 — per row (see notes/MANIFEST_SCHEMA.md for full description):
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


# === YODAS2 — sessions + virtual utterance segments ===

def yodas2_rows(root: Path) -> Iterator[Row]:
    base = root / "raw" / "yodas2_hu000" / "data" / "hu000"
    audio_dir = base / "audio"
    text_dir = base / "text"
    dur_dir = base / "duration"

    # Video-level durations
    video_dur: dict[str, float] = {}
    for f in dur_dir.glob("*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                vid, _, secs = line.partition(" ")
                try:
                    video_dur[vid] = float(secs)
                except ValueError:
                    pass

    # Build utterance lookup from JSONs
    # Each JSON is a list of {audio_id, text: {utt_id: transcript}}
    video_to_utts: dict[str, dict[str, str]] = {}
    for f in sorted(text_dir.glob("*.json")):
        d = json.load(f.open())
        for entry in d:
            vid = entry.get("audio_id")
            if vid:
                video_to_utts[vid] = entry.get("text", {})

    # Iterate over video-level wav files
    for wav in tqdm(sorted(audio_dir.glob("*.wav")), desc="yodas2_hu000", unit="video"):
        vid = wav.stem
        utts = video_to_utts.get(vid, {})
        rel = str(wav.relative_to(root))
        video_duration = video_dur.get(vid, 0.0)

        # 1) Emit one session-level row for the whole video (audio-only, unaligned)
        # Actually we'll skip session-level if we have utterance-level alignment
        # (the JSON gives us per-utt text + timestamps). The unsegmented audio is
        # the parent for the virtual segments. We only emit unsegmented if no utts.
        if not utts:
            yield Row(
                utterance_id=f"yodas2/{vid}",
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
                duration_sec=video_duration,
                segment_start_sec=None,
                segment_end_sec=None,
                parent_audio_path=f"https://www.youtube.com/watch?v={vid}",
                transcripts={},
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
                segmentation_status="session_level",
                quality_flags=quality_flags_basic(video_duration, is_segmented=False),
            )
            continue

        # 2) Emit one virtual-segment row per utterance in the JSON
        for utt_id, transcript in utts.items():
            video_id, seg_start, seg_end = parse_yodas_utterance_id(utt_id)
            dur = None
            if seg_start is not None and seg_end is not None:
                dur = seg_end - seg_start
            yield Row(
                utterance_id=utt_id,
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
                segment_start_sec=seg_start,
                segment_end_sec=seg_end,
                parent_audio_path=f"https://www.youtube.com/watch?v={vid}",
                transcripts={"source_caption": transcript.strip()},
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
                segmentation_status="virtual_segment",
                quality_flags=quality_flags_basic(dur, is_segmented=True),
            )


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

def voxpopuli_labeled_rows(root: Path) -> Iterator[Row]:
    import pyarrow.parquet as pq
    base = root / "raw" / "voxpopuli_hu_labeled" / "hu"
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
            # session_id ~ "20110912-0900-PLENARY-16-hu_<datetime>_<seg>"
            # Extract the leading "<date>-NNNN-PLENARY-N" portion before "-hu"
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
                duration_sec=None,
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


# === MOSEL pseudo-labels (audio not yet segmented) ===

def mosel_pseudo_label_rows(root: Path) -> Iterator[Row]:
    base = root / "raw" / "mosel_hu" / "transcripts" / "hu"
    csv.field_size_limit(sys.maxsize)

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
                    audio_path=None,  # will be filled in after segmentation
                    audio_format=None,
                    parquet_row_index=None,
                    relative_audio_path=None,
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
                    segmentation_status="pending_segmentation",
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


def stats_for_file(path: Path) -> dict:
    by_source: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "hours": 0.0, "with_text": 0,
                 "too_short": 0, "too_long": 0,
                 "pending_segmentation": 0,
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
            if r["segmentation_status"] == "pending_segmentation":
                s["pending_segmentation"] += 1
            if qf.get("any_hallucination_flag"):
                s["halluc_flagged"] += 1
            if qf.get("lid") and qf.get("lid") != "hu":
                s["lid_not_hu"] += 1
    total = {
        "count": sum(s["count"] for s in by_source.values()),
        "hours": round(sum(s["hours"] for s in by_source.values()), 2),
        "with_text": sum(s["with_text"] for s in by_source.values()),
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

    # Build each file separately, streaming where possible
    def aligned_rows() -> Iterator[Row]:
        if sources.get("yodas_hu000", {}).get("status", "").startswith("downloaded"):
            yield from yodas_v1_rows(args.root)
        # YODAS2 virtual-segment rows go into aligned because they have text + timestamps
        yodas2_dir = args.root / "raw" / "yodas2_hu000"
        if (yodas2_dir / ".download_complete").exists():
            yield from yodas2_rows(args.root)
        vp_labeled_dir = args.root / "raw" / "voxpopuli_hu_labeled"
        if (vp_labeled_dir / ".download_complete").exists():
            yield from voxpopuli_labeled_rows(args.root)

    def unaligned_rows() -> Iterator[Row]:
        for src_key, (domain, register) in ARCHIVE_SRC_DOMAIN.items():
            src_cfg = sources.get(src_key, {})
            if not src_cfg.get("status", "").startswith("downloaded"):
                continue
            yield from archive_org_rows(args.root, src_key, src_cfg, domain, register)
        vp_un_dir = args.root / "raw" / "voxpopuli_hu_unlabeled"
        if (vp_un_dir / ".download_complete").exists():
            yield from voxpopuli_unlabeled_rows(args.root)

    print("[aligned] building rows...")
    n_aligned = write_jsonl(aligned_rows(), manifests_dir / "train_aligned.jsonl")
    print(f"  {n_aligned:,} rows -> train_aligned.jsonl")

    print("[unaligned] building rows...")
    n_unaligned = write_jsonl(unaligned_rows(), manifests_dir / "train_unaligned.jsonl")
    print(f"  {n_unaligned:,} rows -> train_unaligned.jsonl")

    if not args.skip_mosel:
        print("[pseudo_labeled] building rows from MOSEL TSV (this is large)...")
        n_pseudo = write_jsonl(mosel_pseudo_label_rows(args.root),
                              manifests_dir / "train_pseudo_labeled.jsonl")
        print(f"  {n_pseudo:,} rows -> train_pseudo_labeled.jsonl")
    else:
        n_pseudo = 0

    # Compute stats from the written files
    stats = {
        "schema_version": 3,
        "aligned": stats_for_file(manifests_dir / "train_aligned.jsonl"),
        "unaligned": stats_for_file(manifests_dir / "train_unaligned.jsonl"),
    }
    pseudo_path = manifests_dir / "train_pseudo_labeled.jsonl"
    if pseudo_path.exists():
        stats["pseudo_labeled"] = stats_for_file(pseudo_path)
    (manifests_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False)
    )

    print()
    print("[done] manifests written to:", manifests_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
