#!/usr/bin/env python3
"""Slice YODAS2 parent WAVs into per-utterance OGG chunks.

Reads `processed/normalization/yodas2_merged.jsonl` (one row per merged
caption segment), groups by parent audio_id, loads each parent WAV once,
slices it into per-utterance chunks resampled to 16 kHz mono OGG/Vorbis.

Output: `processed/chunks/yodas2_hu000/<merged_utt_id>.ogg`
Sidecar: `processed/normalization/yodas2_chunked.jsonl`
  One row per chunk: { utterance_id, audio_path, duration_sec, sample_rate,
                       channels, codec, transcript, parent_audio_id }

Conventions match `bin/chunk_longform.py` (librivox / podcasts) and the
Plan B `voxpopuli_resegmented` layer: 16 kHz mono OGG/Vorbis chunks under
`processed/chunks/<source>/`.

Idempotent: chunks that already exist on disk are skipped. Re-running after
a partial run resumes cleanly.

Run with the base env (numpy + scipy + soundfile):
  /media/cseti/datassd/conda/miniconda3/bin/python bin/chunk_yodas2.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
SIDECAR_IN = DATA_ROOT / "processed" / "normalization" / "yodas2_merged.jsonl"
SIDECAR_OUT = DATA_ROOT / "processed" / "normalization" / "yodas2_chunked.jsonl"
CHUNKS_DIR = DATA_ROOT / "processed" / "chunks" / "yodas2_hu000"
PARENT_DIR = DATA_ROOT / "raw" / "yodas2_hu000" / "data" / "hu000" / "audio"

TARGET_SR = 16000


def _resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio.astype(np.float32, copy=False)
    from math import gcd
    g = gcd(src_sr, tgt_sr)
    up = tgt_sr // g
    down = src_sr // g
    from scipy.signal import resample_poly
    return resample_poly(audio, up, down).astype(np.float32)


def _process_parent(task: tuple) -> tuple[int, int, int]:
    """Worker: load one parent WAV, slice + encode all its utterances.

    Returns (n_written, n_skipped, n_errors)."""
    audio_id, entries = task
    parent_path = PARENT_DIR / f"{audio_id}.wav"
    if not parent_path.is_file():
        return 0, 0, len(entries)

    # Decide work: skip entries whose chunk already exists.
    pending = []
    for entry in entries:
        out_path = CHUNKS_DIR / f"{entry['merged_utt_id']}.ogg"
        if out_path.is_file():
            continue
        pending.append((entry, out_path))
    if not pending:
        return 0, len(entries), 0

    try:
        audio, sr = sf.read(str(parent_path), dtype="float32", always_2d=False)
    except Exception:
        return 0, 0, len(entries)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)
        sr = TARGET_SR

    n_written = 0
    n_errors = 0
    for entry, out_path in pending:
        s = max(0, int(entry["start_sec"] * sr))
        e = min(len(audio), int(entry["end_sec"] * sr))
        if e <= s:
            n_errors += 1
            continue
        clip = audio[s:e]
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        try:
            sf.write(str(tmp), clip, sr, format="OGG", subtype="VORBIS")
            os.replace(tmp, out_path)
            n_written += 1
        except Exception:
            n_errors += 1
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return n_written, len(entries) - len(pending), n_errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=SIDECAR_IN,
                   help="yodas2_merged.jsonl input sidecar.")
    p.add_argument("--output-sidecar", type=Path, default=SIDECAR_OUT,
                   help="yodas2_chunked.jsonl output sidecar.")
    p.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR,
                   help="Output directory for .ogg chunks.")
    p.add_argument("--n-workers", type=int, default=6,
                   help="Parallel pool size (default 6).")
    p.add_argument("--limit-videos", type=int, default=None,
                   help="Process only the first N parent videos "
                        "(useful for smoke-testing).")
    args = p.parse_args()

    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    args.output_sidecar.parent.mkdir(parents=True, exist_ok=True)

    print(f"[chunk] reading {args.input}", file=sys.stderr)
    entries_by_video: dict[str, list[dict]] = defaultdict(list)
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            entries_by_video[r["audio_id"]].append(r)
    n_total_entries = sum(len(v) for v in entries_by_video.values())
    print(f"[chunk] {n_total_entries:,} merged utterances across "
          f"{len(entries_by_video):,} parent videos", file=sys.stderr)

    tasks = list(entries_by_video.items())
    if args.limit_videos:
        tasks = tasks[:args.limit_videos]
        print(f"[chunk] limited to {len(tasks)} videos", file=sys.stderr)

    t0 = time.time()
    n_written = 0
    n_skipped = 0
    n_errors = 0
    progress_anchor = t0
    progress_done = 0
    print(f"[chunk] starting {args.n_workers} workers", file=sys.stderr)
    with Pool(processes=args.n_workers) as pool:
        for i, (w, s, e) in enumerate(
            pool.imap_unordered(_process_parent, tasks, chunksize=4), start=1
        ):
            n_written += w
            n_skipped += s
            n_errors += e
            if i % 200 == 0:
                now = time.time()
                rate = i / (now - t0)
                eta = (len(tasks) - i) / rate / 60.0
                print(f"[chunk] {i:,}/{len(tasks):,} videos "
                      f"({rate:.1f} vid/s, ETA {eta:.1f} min, "
                      f"{n_written:,} new, {n_skipped:,} skip, "
                      f"{n_errors} err)", file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    print(f"\n[chunk] done in {elapsed/60:.1f} min", file=sys.stderr)
    print(f"        {n_written:,} chunks written", file=sys.stderr)
    print(f"        {n_skipped:,} chunks already existed (skipped)",
          file=sys.stderr)
    print(f"        {n_errors:,} errors", file=sys.stderr)

    # Build output sidecar: one row per merged_utt_id with the new audio_path,
    # only for chunks that actually exist on disk now.
    print(f"\n[chunk] writing sidecar {args.output_sidecar}", file=sys.stderr)
    tmp_side = args.output_sidecar.with_suffix(args.output_sidecar.suffix + ".tmp")
    n_side = 0
    with tmp_side.open("w", encoding="utf-8") as out_f:
        for audio_id, entries in entries_by_video.items():
            for entry in entries:
                chunk_path = args.chunks_dir / f"{entry['merged_utt_id']}.ogg"
                if not chunk_path.is_file():
                    continue
                row = {
                    "utterance_id": entry["merged_utt_id"],
                    "audio_path": str(chunk_path),
                    "duration_sec": entry["duration_sec"],
                    "sample_rate": TARGET_SR,
                    "channels": 1,
                    "codec": "ogg",
                    "transcript": entry["text"],
                    "parent_audio_id": audio_id,
                    "video_duration_sec": entry.get("video_duration_sec"),
                    "merged_from": entry.get("merged_from"),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_side += 1
    os.replace(tmp_side, args.output_sidecar)
    print(f"[chunk] {n_side:,} rows in sidecar", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
