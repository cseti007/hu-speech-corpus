#!/usr/bin/env python3
"""VAD-based chunking of long-form audio sources (librivox audiobooks, podcasts)
into 3-30s speech-aligned clips.

Approach:
- ffmpeg decodes each input file to 16kHz mono float32 PCM in memory.
- Silero VAD detects speech segments.
- Greedy merge: consecutive segments with gap <= 1s are merged into a single
  chunk; chunks are capped at 30s and floored at 3s.
- Each emitted chunk is written as a per-clip Ogg/Vorbis file.
- A sidecar JSONL records per-chunk metadata (paths, timings, parent file).

Output:
- processed/chunks/{source}/{file_id}_{chunk_idx:06d}.ogg
- processed/normalization/chunks_{source}.jsonl

Sources currently configured:
- librivox_hu        — 99 audiobook chapter MP3s, ~18.79h
- podcasts_hu_cc     — 42 podcast episode MP3s, ~42.22h

(voxpopuli_unlabeled_gap is handled separately by a dedicated script that
also has to skip regions overlapping with the existing MOSEL clips.)

Run with the dedicated conda env (Silero VAD requires torch):
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python \
      bin/chunk_longform.py --source librivox_hu

Idempotent: overwrites the sidecar and re-emits chunks each run. Use
--limit to process only the first N files for testing.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Per-worker state (loaded once per worker process)
_vad_model = None


def _init_worker():
    """Pool worker initializer: load Silero VAD into per-process global."""
    global _vad_model
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    from silero_vad import load_silero_vad
    _vad_model = load_silero_vad()

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")

SOURCES = {
    "librivox_hu": {
        "root": DATA_ROOT / "raw" / "librivox_hu",
        "glob": "**/*.mp3",
        "out_dir": DATA_ROOT / "processed" / "chunks" / "librivox_hu",
        "sidecar": DATA_ROOT / "processed" / "normalization" / "chunks_librivox_hu.jsonl",
    },
    "podcasts_hu_cc": {
        "root": DATA_ROOT / "raw" / "podcasts_hu",
        "glob": "**/*.mp3",
        "out_dir": DATA_ROOT / "processed" / "chunks" / "podcasts_hu_cc",
        "sidecar": DATA_ROOT / "processed" / "normalization" / "chunks_podcasts_hu_cc.jsonl",
    },
}

TARGET_SR = 16000
MIN_CHUNK_DUR = 3.0   # seconds
MAX_CHUNK_DUR = 30.0  # seconds
GAP_THRESHOLD = 1.0   # seconds — gap within a chunk; larger = new chunk

# Silero VAD tuning
VAD_MIN_SPEECH_MS = 250    # drop sub-250ms speech bursts
VAD_MIN_SILENCE_MS = 500   # merge across <500ms silences (same speech burst)


def decode_to_pcm(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode any audio file to mono 16kHz PCM float32 via ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(path),
        "-ac", "1", "-ar", str(target_sr),
        "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    assert sr == target_sr, f"unexpected sr {sr}"
    return data


def _split_long_segment(seg_start, seg_end, max_dur):
    """Split a single (start, end) segment into pieces of at most max_dur.
    Returns list of (start, end). The last piece may be shorter than max_dur."""
    pieces = []
    s = seg_start
    while seg_end - s > max_dur:
        pieces.append((s, s + max_dur))
        s += max_dur
    if seg_end - s > 0:
        pieces.append((s, seg_end))
    return pieces


def merge_segments_to_chunks(segments, max_dur, gap_threshold, min_dur):
    """Greedy merge VAD segments into chunks (start_sec, end_sec).

    `segments` is a list of dicts with `start` and `end` in seconds.

    Step 1: any single VAD segment longer than max_dur is hard-split at
    max_dur boundaries (e.g. a 200-sec monologue with no >500ms silence
    becomes 7 pieces of ≤30s each). This is necessary because libvorbis
    segfaults on very long single writes and ASR models can't handle
    arbitrarily long inputs anyway.

    Step 2: merge consecutive (post-split) segments greedily while keeping
    chunk duration ≤ max_dur and gap ≤ gap_threshold.
    """
    if not segments:
        return []

    # Step 1: hard-split any oversized segments
    split_segments = []
    for seg in segments:
        for s, e in _split_long_segment(seg["start"], seg["end"], max_dur):
            split_segments.append({"start": s, "end": e})

    # Step 2: greedy merge
    chunks = []
    cur_start = split_segments[0]["start"]
    cur_end = split_segments[0]["end"]
    for seg in split_segments[1:]:
        gap = seg["start"] - cur_end
        merged_dur = seg["end"] - cur_start
        if gap <= gap_threshold and merged_dur <= max_dur:
            cur_end = seg["end"]
        else:
            if cur_end - cur_start >= min_dur:
                chunks.append((cur_start, cur_end))
            cur_start = seg["start"]
            cur_end = seg["end"]
    if cur_end - cur_start >= min_dur:
        chunks.append((cur_start, cur_end))
    return chunks


def file_id_from_path(audio_path: Path, source_root: Path) -> str:
    rel = audio_path.relative_to(source_root).with_suffix("")
    return rel.as_posix().replace("/", "__")


def _process_file_worker(args):
    """Worker function: process one audio file. Returns (file_id, records, status).

    Workers write ogg files directly to disk (per-file-unique IDs prevent
    collisions) and return JSONL records to the main process.
    """
    from silero_vad import get_speech_timestamps

    audio_path_str, source_root_str, source_key, out_dir_str = args
    audio_path = Path(audio_path_str)
    source_root = Path(source_root_str)
    out_dir = Path(out_dir_str)
    file_id = file_id_from_path(audio_path, source_root)

    try:
        pcm = decode_to_pcm(audio_path)
    except Exception as ex:
        return file_id, [], f"error: decode failed: {ex}"
    file_duration_sec = len(pcm) / TARGET_SR

    audio_tensor = torch.from_numpy(pcm).float()
    segments = get_speech_timestamps(
        audio_tensor, _vad_model, sampling_rate=TARGET_SR,
        return_seconds=True,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
    )

    chunks = merge_segments_to_chunks(
        segments, MAX_CHUNK_DUR, GAP_THRESHOLD, MIN_CHUNK_DUR
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_dur = 0.0
    for idx, (start_sec, end_sec) in enumerate(chunks):
        start_frame = int(start_sec * TARGET_SR)
        end_frame = int(end_sec * TARGET_SR)
        clip = pcm[start_frame:end_frame]
        out_path = out_dir / f"{file_id}_{idx:06d}.ogg"
        sf.write(str(out_path), clip, TARGET_SR, format="OGG", subtype="VORBIS")
        records.append({
            "source": source_key,
            "file_id": file_id,
            "chunk_index": idx,
            "audio_path": str(out_path),
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration_sec": round(end_sec - start_sec, 3),
            "parent_file_path": str(audio_path),
            "parent_file_duration_sec": round(file_duration_sec, 3),
        })
        total_dur += end_sec - start_sec

    return file_id, records, f"ok: {len(records)} chunks, "\
        f"{total_dur/60:.1f}min ({total_dur/file_duration_sec*100:.1f}% retention)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=list(SOURCES.keys()))
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N files (testing).")
    parser.add_argument("--n_workers", type=int, default=6,
                        help="Worker process count (default: 6).")
    args = parser.parse_args()

    cfg = SOURCES[args.source]
    files = sorted(cfg["root"].rglob(cfg["glob"]))
    if args.limit:
        files = files[:args.limit]

    print(f"[{args.source}] {len(files)} files to chunk "
          f"({args.n_workers} workers)", file=sys.stderr)
    cfg["sidecar"].parent.mkdir(parents=True, exist_ok=True)

    work_items = [
        (str(f), str(cfg["root"]), args.source, str(cfg["out_dir"]))
        for f in files
    ]

    total_chunks = 0
    total_chunk_dur = 0.0
    total_file_dur = 0.0
    file_durations_seen = {}  # file_id -> parent_file_duration_sec (dedup)

    with cfg["sidecar"].open("w", encoding="utf-8") as out:
        with Pool(processes=args.n_workers, initializer=_init_worker) as pool:
            for file_id, records, status in pool.imap_unordered(
                _process_file_worker, work_items, chunksize=1
            ):
                print(f"[{args.source}] {file_id}: {status}", file=sys.stderr)
                if status.startswith("error"):
                    continue
                if records:
                    file_durations_seen[file_id] = records[0]["parent_file_duration_sec"]
                for rec in records:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_chunks += 1
                    total_chunk_dur += rec["duration_sec"]

    total_file_dur = sum(file_durations_seen.values())

    print()
    print(f"=== {args.source} chunking summary ===")
    print(f"Input files:                {len(files)}")
    print(f"Files processed (no error): {len(file_durations_seen)}")
    print(f"Total input duration:       {total_file_dur/3600:.2f}h")
    print(f"Chunks emitted:             {total_chunks:,}")
    print(f"Total chunk duration:       {total_chunk_dur/3600:.2f}h")
    if total_file_dur > 0:
        print(f"Speech retention ratio:     {total_chunk_dur/total_file_dur*100:.1f}%")
    print(f"Output dir:                 {cfg['out_dir']}")
    print(f"Sidecar:                    {cfg['sidecar']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
