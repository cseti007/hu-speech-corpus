#!/usr/bin/env python3
"""Phase 3 Tier-1: cheap audio statistics for every clip-level row.

For each clip-level row in `manifest.jsonl`, compute:
  - rms_dbfs       RMS loudness in dBFS
  - peak_dbfs      peak amplitude in dBFS
  - is_clipped     bool: any sample at ±1.0 full-scale
  - silence_ratio  fraction of 25ms frames below -40 dBFS

Output: processed/quality/tier1.jsonl, one row per utterance_id:
  {"utterance_id": "...", "rms_dbfs": -23.4, "peak_dbfs": -0.5, "is_clipped": false, "silence_ratio": 0.12}

The manifest builder reads this sidecar and merges into `quality_flags` on
the next rebuild.

Audio sources are handled differently by clip type:
  - Standalone ogg (MOSEL pseudo, librivox/podcasts/voxpopuli_gap chunks):
    decode directly. Parallel pool, ~30-40 min for ~3.2M clips.
  - YODAS2 merged clips (virtual segments of parent WAV): group by parent,
    decode parent once, slice each clip. ~5 min sequential.
  - VoxPopuli labeled (parquet-internal): decode embedded audio per row.
    ~5 min sequential.

Idempotent: rows whose utterance_id is already in `tier1.jsonl` are skipped.
Run with the dedicated conda env (needs torch-free Python + numpy + soundfile):
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python bin/quality_tier1.py
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
MANIFESTS_DIR = DATA_ROOT / "processed" / "manifests"
OUT_DIR = DATA_ROOT / "processed" / "quality"
OUT_PATH = OUT_DIR / "tier1.jsonl"

SILENCE_THRESHOLD_DBFS = -40.0
FRAME_DURATION_SEC = 0.025
CLIPPING_THRESHOLD = 0.9999


def compute_metrics(audio: np.ndarray, sr: int) -> dict:
    """Compute the 4 Tier-1 metrics on a float32 PCM array."""
    if audio.ndim > 1:
        # mono-mix if stereo
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return {"rms_dbfs": -120.0, "peak_dbfs": -120.0,
                "is_clipped": False, "silence_ratio": 1.0}
    abs_audio = np.abs(audio)
    peak = float(abs_audio.max())
    peak_dbfs = 20.0 * np.log10(peak + 1e-12)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_dbfs = 20.0 * np.log10(rms + 1e-12)
    is_clipped = bool(peak >= CLIPPING_THRESHOLD)

    frame_size = max(1, int(FRAME_DURATION_SEC * sr))
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        silence_ratio = 1.0 if rms_dbfs < SILENCE_THRESHOLD_DBFS else 0.0
    else:
        frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        frame_dbfs = 20.0 * np.log10(frame_rms + 1e-12)
        silence_ratio = float(np.mean(frame_dbfs < SILENCE_THRESHOLD_DBFS))

    return {
        "rms_dbfs": round(rms_dbfs, 2),
        "peak_dbfs": round(peak_dbfs, 2),
        "is_clipped": is_clipped,
        "silence_ratio": round(silence_ratio, 4),
    }


# --- Worker: standalone ogg files (parallel pool) ---

def _worker_standalone_ogg(args):
    """Worker: decode a single ogg file, return (utterance_id, metrics_or_None)."""
    utterance_id, audio_path = args
    try:
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception as ex:
        return utterance_id, None, f"decode_error: {ex}"
    metrics = compute_metrics(audio, sr)
    return utterance_id, metrics, None


# --- Sequential: YODAS2 merged (group by parent video) ---

def process_yodas2_merged(manifest_path: Path, already_done: set,
                          out_writer) -> int:
    """Legacy yodas2 stage: parent WAV + segment_start/end slicing.

    Post-2026-05-26: yodas2 rows are typically chunked to standalone OGG
    files (see bin/chunk_yodas2.py + yodas2_chunked.jsonl). Such rows have
    segment_start_sec / segment_end_sec = None and are handled by the
    standalone-ogg stage instead. Skip them here."""
    by_parent: dict[str, list[dict]] = defaultdict(list)
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] != "yodas2_hu000":
                continue
            if r["utterance_id"] in already_done:
                continue
            # Skip chunked rows (standalone OGG with no segment offsets) —
            # they'll be picked up by collect_standalone_ogg_rows.
            if r.get("segment_start_sec") is None:
                continue
            by_parent[r["audio_path"]].append(r)

    n_done = 0
    t0 = time.time()
    for parent_path, rows in by_parent.items():
        try:
            wav, sr = sf.read(parent_path, dtype="float32", always_2d=False)
        except Exception as ex:
            print(f"[yodas2] decode failed {parent_path}: {ex}", file=sys.stderr)
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        for r in rows:
            s = int(r["segment_start_sec"] * sr)
            e = int(r["segment_end_sec"] * sr)
            e = min(e, len(wav))
            if e <= s:
                continue
            metrics = compute_metrics(wav[s:e], sr)
            out_writer.write(json.dumps(
                {"utterance_id": r["utterance_id"], **metrics},
                ensure_ascii=False
            ) + "\n")
            n_done += 1
    print(f"[yodas2] {n_done:,} clips processed in {time.time()-t0:.1f}s",
          file=sys.stderr)
    return n_done


# --- Sequential: VoxPopuli labeled (parquet-internal audio) ---

def process_voxpopuli_labeled(manifest_path: Path, already_done: set,
                              out_writer) -> int:
    import pyarrow.parquet as pq

    # Group rows by parquet file path + row_index
    by_parquet: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] != "voxpopuli_hu_labeled":
                continue
            if r["utterance_id"] in already_done:
                continue
            if r.get("parquet_row_index") is None:
                continue
            by_parquet[r["audio_path"]].append((r["parquet_row_index"], r["utterance_id"]))

    n_done = 0
    t0 = time.time()
    for parquet_path, entries in by_parquet.items():
        table = pq.read_table(parquet_path, columns=["audio"])
        audio_col = table.column("audio").to_pylist()
        for row_idx, utt_id in entries:
            audio_struct = audio_col[row_idx]
            try:
                audio, sr = sf.read(io.BytesIO(audio_struct["bytes"]),
                                    dtype="float32", always_2d=False)
            except Exception as ex:
                print(f"[voxpopuli_labeled] decode failed for {utt_id}: {ex}",
                      file=sys.stderr)
                continue
            metrics = compute_metrics(audio, sr)
            out_writer.write(json.dumps(
                {"utterance_id": utt_id, **metrics},
                ensure_ascii=False
            ) + "\n")
            n_done += 1
    print(f"[voxpopuli_labeled] {n_done:,} clips processed in {time.time()-t0:.1f}s",
          file=sys.stderr)
    return n_done


# --- Parallel: all standalone-ogg rows ---

def collect_standalone_ogg_rows(already_done: set) -> list[tuple[str, str]]:
    """Walk manifest.jsonl and collect (utterance_id, audio_path) for rows whose
    audio is a standalone file (not a parquet-internal or YODAS2-parent-virtual
    segment). Covers mosel pseudo + librivox/podcasts/voxpopuli_gap chunks."""
    standalone_sources = {
        "mosel_hu_voxpopuli",
        "librivox_hu",
        "podcasts_hu_cc",
        "voxpopuli_unlabeled_gap",
        "voxpopuli_resegmented",
        "common_voice_25_0_hu",  # MP3 standalone clips; sf.read handles MP3
        "yodas2_hu000",  # post-chunking, 16 kHz mono OGG standalone files
    }
    work = []
    path = MANIFEST_PATH
    if not path.exists():
        return work
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["utterance_id"] in already_done:
                continue
            if r["source"] not in standalone_sources:
                continue
            if not r.get("audio_path"):
                continue
            work.append((r["utterance_id"], r["audio_path"]))
    return work


def process_standalone_ogg(work: list[tuple[str, str]], n_workers: int,
                           out_writer, progress_every: int = 20000) -> int:
    if not work:
        return 0
    n_done = 0
    n_errors = 0
    t0 = time.time()
    progress_anchor = t0
    progress_done = 0
    print(f"[standalone] {len(work):,} clips, {n_workers} workers", file=sys.stderr)
    with Pool(processes=n_workers) as pool:
        for utterance_id, metrics, error in pool.imap_unordered(
            _worker_standalone_ogg, work, chunksize=200
        ):
            if metrics is None:
                n_errors += 1
                continue
            out_writer.write(json.dumps(
                {"utterance_id": utterance_id, **metrics},
                ensure_ascii=False
            ) + "\n")
            n_done += 1
            if n_done - progress_done >= progress_every:
                now = time.time()
                rate = (n_done - progress_done) / (now - progress_anchor)
                eta = (len(work) - n_done) / rate
                print(f"[standalone] {n_done:,}/{len(work):,} "
                      f"({rate:.0f} clips/s, ETA {eta/60:.1f} min, "
                      f"{n_errors} errors)", file=sys.stderr)
                progress_anchor = now
                progress_done = n_done
    print(f"[standalone] {n_done:,} clips processed in {time.time()-t0:.1f}s "
          f"({n_errors} errors)", file=sys.stderr)
    return n_done


def load_existing_done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    ids = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["utterance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=MANIFESTS_DIR / "manifest.jsonl",
                        help="Manifest JSONL to iterate (default: manifest.jsonl). "
                             "Pass manifest_v5.jsonl when scoring the new "
                             "voxpopuli_resegmented chunks. Pass a mini "
                             "manifest_v5 (smoke/dev) to score just those clips.")
    parser.add_argument("--output", type=Path, default=OUT_PATH,
                        help="Sidecar JSONL output path "
                             "(default: processed/quality/tier1.jsonl). "
                             "Pass an alternate path when scoring smoke/dev sets.")
    parser.add_argument("--n_workers", type=int, default=12,
                        help="Worker count for standalone-ogg parallel pool (default 12).")
    parser.add_argument("--skip-yodas2", action="store_true")
    parser.add_argument("--skip-voxpopuli-labeled", action="store_true")
    parser.add_argument("--skip-standalone", action="store_true")
    args = parser.parse_args()

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = load_existing_done(out_path)
    print(f"[init] {len(already_done):,} rows already in {out_path.name}",
          file=sys.stderr)

    manifest_path = args.input
    # Make MANIFEST_PATH visible to the helper used by collect_standalone_ogg_rows
    globals()["MANIFEST_PATH"] = manifest_path
    grand_total = 0
    with out_path.open("a", encoding="utf-8") as out:
        if not args.skip_yodas2:
            print("[stage 1/3] YODAS2 merged clips (sequential, group by parent WAV)",
                  file=sys.stderr)
            grand_total += process_yodas2_merged(manifest_path, already_done, out)
        if not args.skip_voxpopuli_labeled:
            print("[stage 2/3] VoxPopuli labeled (sequential, parquet-internal)",
                  file=sys.stderr)
            grand_total += process_voxpopuli_labeled(manifest_path, already_done, out)
        if not args.skip_standalone:
            print("[stage 3/3] Standalone clips (parallel pool: ogg + mp3)",
                  file=sys.stderr)
            already_done.update(load_existing_done(out_path))  # refresh after stages 1+2
            work = collect_standalone_ogg_rows(already_done)
            grand_total += process_standalone_ogg(work, args.n_workers, out)

    print()
    print(f"=== Tier-1 quality scoring summary ===")
    print(f"New rows added:   {grand_total:,}")
    print(f"Total in sidecar: {len(load_existing_done(out_path)):,}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
