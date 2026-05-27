#!/usr/bin/env python3
"""Re-segment the full 22,076 h VoxPopuli HU unlabeled corpus with Silero VAD.

Replaces the mosel 30-second sliding-window boundaries (which are NOT
alignment — see notes/JOURNEY.md "VoxPopuli unlabelled is 30-second
windows" entry) with natural speech-boundary chunks.

For each raw EP session OGG:
  1. ffmpeg-decode to mono 16 kHz PCM
  2. Run Silero VAD with conservative params (min_silence_duration_ms=300,
     min_speech_duration_ms=250)
  3. Merge consecutive speech segments into 3-30 s chunks (gap ≤ 1.0 s),
     same logic as `bin/chunk_voxpopuli_unlabeled_gap.py`
  4. Add 200 ms of natural ambient padding to each chunk on both sides
     (clipped to session bounds, takes real audio from the parent — NOT
     zero-padding, since real room tone is needed for ASR/TTS robustness)
  5. Re-encode each chunk to OGG Vorbis

Output:
  - processed/voxpopuli_resegmented/{session_id}_{chunk_idx:06d}.ogg
  - processed/normalization/voxpopuli_resegmented.jsonl  (append-only)

Sentinel-based idempotency: re-running skips sessions where
processed/voxpopuli_resegmented/.sentinels/{session_id}.done exists.

Designed for the dedicated conda env (Silero + soundfile + ffmpeg):
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python \
      bin/resegment_voxpopuli_silero.py --n_workers 8

Estimated runtime: 17,297 sessions × ~10 sec/session / 8 workers ≈ 6 h.

This produces a fresh per-utterance corpus that REPLACES the mosel
boundaries. The mosel pseudo Whisper labels remain (for reference) but
will not be the primary text source after Phase 4 consensus.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
VP_RAW_DIR = DATA_ROOT / "raw" / "voxpopuli_hu_unlabeled" / "raw_audios" / "hu"
OUT_DIR = DATA_ROOT / "processed" / "voxpopuli_resegmented"
SIDECAR_PATH = (
    DATA_ROOT / "processed" / "normalization" / "voxpopuli_resegmented.jsonl"
)
SENTINEL_DIR = OUT_DIR / ".sentinels"

TARGET_SR = 16000
MIN_CHUNK_DUR = 3.0
MAX_CHUNK_DUR = 30.0
GAP_THRESHOLD = 1.0      # merge adjacent VAD segments if gap ≤ 1.0 s
PADDING_SEC = 0.2        # 200 ms ambient padding on each side
PADDING_SAMPLES = int(PADDING_SEC * TARGET_SR)

VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 300

# Per-worker state
_vad_model = None


def _init_worker():
    """Pool worker init: load Silero VAD once per process, pin BLAS threads."""
    global _vad_model
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    _vad_model = load_silero_vad()


# ============================================================
# Audio decode + encode helpers
# ============================================================

def decode_to_pcm(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """ffmpeg-decode a full audio file to mono float32 PCM at target_sr."""
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(path),
        "-ac", "1", "-ar", str(target_sr),
        "-f", "wav", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    assert sr == target_sr
    return data


def encode_ogg(audio: np.ndarray, out_path: Path, sr: int = TARGET_SR) -> None:
    """Write float32 PCM as OGG Vorbis."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr, format="OGG", subtype="VORBIS")


# ============================================================
# Segment manipulation
# ============================================================

def _split_long_segment(seg_start: float, seg_end: float,
                        max_dur: float) -> list[tuple[float, float]]:
    """Split a too-long segment into max_dur pieces."""
    pieces = []
    s = seg_start
    while seg_end - s > max_dur:
        pieces.append((s, s + max_dur))
        s += max_dur
    if seg_end - s > 0:
        pieces.append((s, seg_end))
    return pieces


def merge_segments_to_chunks(segments: list[dict], max_dur: float,
                             gap_threshold: float,
                             min_dur: float) -> list[tuple[float, float]]:
    """Merge consecutive VAD speech segments into 3-30s chunks at ≤1s gaps."""
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


def apply_padding(chunks_sec: list[tuple[float, float]], session_dur_sec: float,
                  pad_sec: float = PADDING_SEC) -> list[tuple[float, float]]:
    """Extend each chunk by `pad_sec` on both sides, clipped to session bounds.

    Adjacent chunks may overlap by up to 2*pad_sec (≤400 ms); we accept the
    overlap (acoustically negligible, each chunk is an independent sample
    for training)."""
    out = []
    for start, end in chunks_sec:
        new_start = max(0.0, start - pad_sec)
        new_end = min(session_dur_sec, end + pad_sec)
        if new_end - new_start >= MIN_CHUNK_DUR:
            out.append((new_start, new_end))
    return out


# ============================================================
# Per-session worker
# ============================================================

def _process_session_worker(audio_path_str: str) -> dict:
    """Worker entry: decode → VAD → merge → pad → encode all chunks.

    Returns a status record with counts. Chunks are written directly to
    OUT_DIR; the result dict's `chunks` list has per-chunk metadata for
    the sidecar (so the main process serialises them in order)."""
    from silero_vad import get_speech_timestamps

    audio_path = Path(audio_path_str)
    session_id = audio_path.stem  # e.g. "20090112-0900-PLENARY-10_hu"
    sentinel = SENTINEL_DIR / f"{session_id}.done"
    if sentinel.exists():
        return {"session_id": session_id, "status": "skipped", "chunks": []}

    # 1. Decode
    try:
        audio = decode_to_pcm(audio_path, TARGET_SR)
    except Exception as ex:
        return {"session_id": session_id, "status": f"decode_error: {ex}",
                "chunks": []}

    if len(audio) < int(TARGET_SR * MIN_CHUNK_DUR):
        sentinel.touch()
        return {"session_id": session_id, "status": "too_short", "chunks": []}

    session_dur_sec = len(audio) / TARGET_SR

    # 2. VAD
    try:
        import torch
        t = torch.from_numpy(audio).float()
        segs = get_speech_timestamps(
            t, _vad_model, sampling_rate=TARGET_SR, return_seconds=True,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        )
    except Exception as ex:
        return {"session_id": session_id, "status": f"vad_error: {ex}",
                "chunks": []}

    if not segs:
        sentinel.touch()
        return {"session_id": session_id, "status": "no_speech", "chunks": []}

    # 3. Merge into 3-30s chunks (no padding yet)
    chunks_sec = merge_segments_to_chunks(
        segs, MAX_CHUNK_DUR, GAP_THRESHOLD, MIN_CHUNK_DUR
    )

    # 4. Add 200ms padding on both sides, clipped to session bounds.
    # Padding gives natural room tone, improving ASR/TTS robustness vs
    # tight cuts. May produce up to 400ms overlap between adjacent chunks
    # — acceptable, each chunk is an independent training sample.
    chunks_padded = apply_padding(chunks_sec, session_dur_sec, PADDING_SEC)

    # 5. Encode each chunk to OGG
    chunk_records = []
    for idx, (start_sec, end_sec) in enumerate(chunks_padded):
        s_sample = int(start_sec * TARGET_SR)
        e_sample = int(end_sec * TARGET_SR)
        chunk_audio = audio[s_sample:e_sample]
        if len(chunk_audio) < int(MIN_CHUNK_DUR * TARGET_SR):
            continue
        chunk_filename = f"{session_id}_{idx:06d}.ogg"
        out_path = OUT_DIR / chunk_filename
        try:
            encode_ogg(chunk_audio, out_path, TARGET_SR)
        except Exception as ex:
            chunk_records.append({"chunk_idx": idx, "encode_error": str(ex)[:120]})
            continue
        chunk_records.append({
            "utterance_id": f"voxpopuli_resegmented/{session_id}_{idx:06d}",
            "parent_session_id": session_id,
            "chunk_idx": idx,
            "audio_path": str(out_path),
            "audio_format": "ogg",
            "sample_rate": TARGET_SR,
            "channels": 1,
            "duration_sec": round(end_sec - start_sec, 3),
            "segment_start_sec": round(start_sec, 3),
            "segment_end_sec": round(end_sec, 3),
            "padding_sec": PADDING_SEC,
        })

    sentinel.touch()
    return {
        "session_id": session_id,
        "status": "ok",
        "session_dur_sec": round(session_dur_sec, 2),
        "chunks": chunk_records,
        "n_vad_segments": len(segs),
        "n_chunks": len(chunk_records),
    }


# ============================================================
# Main
# ============================================================

def find_sessions() -> list[Path]:
    """List all raw VoxPopuli HU unlabeled session OGGs, sorted."""
    sessions = []
    if not VP_RAW_DIR.exists():
        return sessions
    for year_dir in sorted(VP_RAW_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        for f in sorted(year_dir.iterdir()):
            if f.suffix == ".ogg":
                sessions.append(f)
    return sessions


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N sessions (for validation).")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[init] scanning sessions under {VP_RAW_DIR}...", file=sys.stderr)
    sessions = find_sessions()
    print(f"[init] found {len(sessions):,} session OGGs", file=sys.stderr)
    if args.limit:
        sessions = sessions[:args.limit]
        print(f"[init] limited to first {len(sessions):,}", file=sys.stderr)

    # Filter out already-done sessions for accurate progress count
    pending = [s for s in sessions if not (SENTINEL_DIR / f"{s.stem}.done").exists()]
    print(f"[init] {len(pending):,} sessions pending "
          f"({len(sessions) - len(pending):,} already done)", file=sys.stderr)
    if not pending:
        print("[done] nothing to do", file=sys.stderr)
        return 0

    t0 = time.time()
    n_done = 0
    n_chunks_total = 0
    n_errors = 0
    progress_every = max(50, len(pending) // 200)

    with SIDECAR_PATH.open("a", encoding="utf-8") as sidecar:
        with Pool(processes=args.n_workers, initializer=_init_worker) as pool:
            work = [str(s) for s in pending]
            for result in pool.imap_unordered(_process_session_worker, work,
                                              chunksize=2):
                n_done += 1
                status = result["status"]
                chunks = result.get("chunks", [])
                if status.startswith("error") or status.startswith("decode_error") \
                        or status.startswith("vad_error"):
                    n_errors += 1
                # Append each chunk record
                for c in chunks:
                    if "encode_error" in c:
                        n_errors += 1
                        continue
                    sidecar.write(json.dumps(c, ensure_ascii=False) + "\n")
                    n_chunks_total += 1
                sidecar.flush()

                if n_done % progress_every == 0:
                    rate = n_done / (time.time() - t0)
                    eta_min = (len(pending) - n_done) / rate / 60 if rate > 0 else 0
                    print(f"[progress] {n_done:,}/{len(pending):,} sessions "
                          f"({n_chunks_total:,} chunks, {n_errors:,} errors, "
                          f"{rate:.1f} sess/s, ETA {eta_min:.1f} min)",
                          file=sys.stderr)

    print()
    print("=== Resegment summary ===")
    print(f"Sessions processed: {n_done:,}")
    print(f"Chunks emitted:     {n_chunks_total:,}")
    print(f"Errors:             {n_errors:,}")
    print(f"Time:               {(time.time()-t0)/60:.1f} min")
    print(f"Output:             {OUT_DIR}")
    print(f"Sidecar:            {SIDECAR_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
