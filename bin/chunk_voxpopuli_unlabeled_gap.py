#!/usr/bin/env python3
"""VAD-chunk the ~4,300h of VoxPopuli HU unlabeled audio NOT covered by MOSEL.

Parallel implementation: a multiprocessing.Pool of workers, each loading its
own Silero VAD model on init. The main process reads MOSEL coverage and
dispatches per-session work items, then writes results to a single sidecar
(no file-locking complexity).

Memory note: Silero VAD is lightweight (~200 MB inference RAM per worker)
plus per-session decode buffers (~100 MB for a 1-hour session). At 8 workers
peak RAM is ~4-5 GB — well under the 123 GB limit on this box. Per the
torchaudio-workers caution memory: default n_workers=8, max recommended 12.

Sentinel-based idempotency: re-running skips sessions where
processed/chunks/voxpopuli_unlabeled_gap/.sentinels/{session_id}.done exists.

Output:
- processed/chunks/voxpopuli_unlabeled_gap/{session_id}_{chunk_idx:06d}.ogg
- processed/normalization/chunks_voxpopuli_unlabeled_gap.jsonl  (appended)

Run with the dedicated conda env:
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python \
      bin/chunk_voxpopuli_unlabeled_gap.py --n_workers 8

Estimated runtime: 17,297 sessions × ~2-3 sec/session / 8 workers ≈ 1-2 hours.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
VP_RAW_DIR = DATA_ROOT / "raw" / "voxpopuli_hu_unlabeled" / "raw_audios" / "hu"
ALIGNMENT_TSV = (
    DATA_ROOT / "raw" / "voxpopuli_hu_unlabeled" / "annotations" /
    "unlabelled_v2.tsv.gz"
)
OUT_DIR = DATA_ROOT / "processed" / "chunks" / "voxpopuli_unlabeled_gap"
SIDECAR_PATH = (
    DATA_ROOT / "processed" / "normalization" /
    "chunks_voxpopuli_unlabeled_gap.jsonl"
)
SENTINEL_DIR = OUT_DIR / ".sentinels"

TARGET_SR = 16000
MIN_CHUNK_DUR = 3.0
MAX_CHUNK_DUR = 30.0
GAP_THRESHOLD = 1.0
MIN_GAP_INTERVAL = 3.0

VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 500

# Per-worker state (loaded once per worker process)
_vad_model = None


def _init_worker():
    """Pool worker initializer: load Silero VAD model into per-process global."""
    global _vad_model
    # Quieter Silero JIT loading
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    from silero_vad import load_silero_vad
    _vad_model = load_silero_vad()


# --- Utility functions (shared between main and worker)


def load_mosel_coverage() -> dict[str, list[tuple[float, float]]]:
    coverage: dict[str, list[tuple[float, float]]] = defaultdict(list)
    print(f"[align] loading {ALIGNMENT_TSV.name}", file=sys.stderr)
    with gzip.open(ALIGNMENT_TSV, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            event_id = r.get("event_id", "")
            if not event_id.endswith("_hu"):
                continue
            try:
                start = float(r["start"])
                end = float(r["end"])
            except (ValueError, KeyError):
                continue
            coverage[event_id].append((start, end))
    for sid in coverage:
        coverage[sid].sort()
    print(f"[align] {len(coverage):,} sessions with MOSEL coverage", file=sys.stderr)
    return dict(coverage)


def compute_gap_intervals(covered, session_dur, min_interval=MIN_GAP_INTERVAL):
    if session_dur <= 0:
        return []
    gaps = []
    cursor = 0.0
    for s, e in covered:
        if s > cursor and (s - cursor) >= min_interval:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if session_dur - cursor >= min_interval:
        gaps.append((cursor, session_dur))
    return gaps


def decode_to_pcm(path, start_sec=0.0, dur_sec=None, target_sr=TARGET_SR):
    cmd = ["ffmpeg", "-loglevel", "error"]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", str(path)]
    if dur_sec is not None:
        cmd += ["-t", f"{dur_sec:.3f}"]
    cmd += ["-ac", "1", "-ar", str(target_sr), "-f", "wav", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    assert sr == target_sr
    return data


def _split_long_segment(seg_start, seg_end, max_dur):
    pieces = []
    s = seg_start
    while seg_end - s > max_dur:
        pieces.append((s, s + max_dur))
        s += max_dur
    if seg_end - s > 0:
        pieces.append((s, seg_end))
    return pieces


def merge_segments_to_chunks(segments, max_dur, gap_threshold, min_dur):
    if not segments:
        return []
    split_segments = []
    for seg in segments:
        for s, e in _split_long_segment(seg["start"], seg["end"], max_dur):
            split_segments.append({"start": s, "end": e})
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


def get_session_duration(audio_path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(audio_path)],
            timeout=30,
        )
        return float(out.strip())
    except Exception:
        return 0.0


def _process_session_worker(args):
    """Worker function. Returns (session_id, list_of_chunk_records, status).

    Strategy: per-gap-interval ffmpeg decode (only fetch audio we actually
    need). For sessions where MOSEL covers most of the audio, this is far
    more efficient than decoding the whole session.

    `status` is 'ok' / 'skipped' / 'error: <msg>'. Workers write ogg files
    directly and touch the sentinel; per-session-unique IDs prevent collisions.
    """
    from silero_vad import get_speech_timestamps

    audio_path_str, session_id, coverage = args
    audio_path = Path(audio_path_str)
    sentinel = SENTINEL_DIR / f"{session_id}.done"
    if sentinel.exists():
        return session_id, [], "skipped"

    session_dur = get_session_duration(audio_path)
    if session_dur <= 0:
        sentinel.touch()
        return session_id, [], "error: zero duration"

    gap_intervals = compute_gap_intervals(coverage, session_dur)
    if not gap_intervals:
        sentinel.touch()
        return session_id, [], "ok"

    records = []
    n_chunks = 0
    try:
        for gap_idx, (g_start, g_end) in enumerate(gap_intervals):
            gap_dur = g_end - g_start
            try:
                pcm = decode_to_pcm(audio_path, start_sec=g_start, dur_sec=gap_dur)
            except Exception as ex:
                return session_id, records, f"error: decode {g_start:.1f}-{g_end:.1f}: {ex}"
            if pcm.size == 0:
                continue
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
            for ch_local_start, ch_local_end in chunks:
                sf_start = int(ch_local_start * TARGET_SR)
                sf_end = int(ch_local_end * TARGET_SR)
                clip = pcm[sf_start:sf_end]
                out_path = OUT_DIR / f"{session_id}_{n_chunks:06d}.ogg"
                sf.write(str(out_path), clip, TARGET_SR, format="OGG", subtype="VORBIS")
                global_start = g_start + ch_local_start
                global_end = g_start + ch_local_end
                records.append({
                    "source": "voxpopuli_unlabeled_gap",
                    "session_id": session_id,
                    "chunk_index": n_chunks,
                    "audio_path": str(out_path),
                    "start_sec": round(global_start, 3),
                    "end_sec": round(global_end, 3),
                    "duration_sec": round(global_end - global_start, 3),
                    "parent_file_path": str(audio_path),
                    "parent_file_duration_sec": round(session_dur, 3),
                    "gap_interval_index": gap_idx,
                })
                n_chunks += 1
    except Exception as ex:
        return session_id, records, f"error: {ex}"

    sentinel.touch()
    return session_id, records, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Worker process count (default: 8, max recommended 12).")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess sessions whose sentinels exist.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.force:
        for s in SENTINEL_DIR.glob("*.done"):
            s.unlink()

    coverage = load_mosel_coverage()
    session_files = sorted(VP_RAW_DIR.rglob("*.ogg"))
    if args.limit:
        session_files = session_files[:args.limit]
    print(f"[init] {len(session_files):,} session files found", file=sys.stderr)

    # Build work items (filter already-done sessions)
    work_items = []
    n_already_done = 0
    for sf_path in session_files:
        sid = sf_path.stem
        if (SENTINEL_DIR / f"{sid}.done").exists() and not args.force:
            n_already_done += 1
            continue
        cov = coverage.get(sid, [])
        work_items.append((str(sf_path), sid, cov))
    print(f"[init] {n_already_done:,} sessions already done (skipping), "
          f"{len(work_items):,} to process", file=sys.stderr)

    if not work_items:
        print("[done] nothing to do", file=sys.stderr)
        return 0

    print(f"[init] launching pool of {args.n_workers} workers", file=sys.stderr)

    n_processed = 0
    n_errors = 0
    total_chunks = 0
    total_dur = 0.0
    progress_every = max(50, len(work_items) // 100)

    # Append-mode sidecar (resumable)
    mode = "w" if args.force else "a"
    with SIDECAR_PATH.open(mode, encoding="utf-8") as out:
        with Pool(processes=args.n_workers, initializer=_init_worker) as pool:
            for session_id, records, status in pool.imap_unordered(
                _process_session_worker, work_items, chunksize=1
            ):
                if status.startswith("error"):
                    n_errors += 1
                    print(f"[error] {session_id}: {status}", file=sys.stderr)
                elif status == "ok":
                    for rec in records:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_chunks += 1
                        total_dur += rec["duration_sec"]
                # skipped status: nothing to write
                n_processed += 1
                if n_processed % progress_every == 0:
                    print(f"[progress] {n_processed:,}/{len(work_items):,} "
                          f"sessions ({total_chunks:,} chunks, "
                          f"{total_dur/3600:.1f}h, {n_errors} errors)",
                          file=sys.stderr)

    print()
    print("=== voxpopuli_unlabeled_gap chunking summary ===")
    print(f"Sessions found:          {len(session_files):,}")
    print(f"Already done (skipped):  {n_already_done:,}")
    print(f"Processed this run:      {n_processed:,}")
    print(f"Errors:                  {n_errors:,}")
    print(f"Chunks emitted:          {total_chunks:,}")
    print(f"Total chunk duration:    {total_dur/3600:.2f}h")
    print(f"Output dir:              {OUT_DIR}")
    print(f"Sidecar:                 {SIDECAR_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
