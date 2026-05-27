#!/usr/bin/env python3
"""Phase 2.6a: post-hoc boundary refinement for mosel_hu_voxpopuli clips.

For each in-scope clip, this tool:
  1. Looks up the clip's parent session + position via the manifest
  2. Loads the parent session audio (cached per session)
  3. Pads the clip's (start, end) by `--pad-sec` on both sides, bounded
     by adjacent mosel utterance gaps (never overlap a neighbor by > 1ms)
  4. Runs Silero VAD on the padded region with loose silence params
  5. Finds new boundaries:
       - new_start = the LATEST silence point within [orig_start - pad, orig_start + pad]
       - new_end   = the EARLIEST silence point within [orig_end - pad, orig_end + pad]
       (silence = midpoint between two VAD speech segments, or the gap to the closest edge)
  6. Re-encodes the refined clip to OGG Vorbis at the new boundaries
  7. Writes a sidecar entry with original + refined timings + path

Outputs:
  - Refined OGGs: `processed/normalization/mosel_refined/{utterance_id_safe}.ogg`
  - Sidecar JSONL: `processed/normalization/mosel_boundary_refined.jsonl`

Idempotent: utterance_ids already in the sidecar are skipped.

Scope:
  Default scope is `--sample-index notes/poc_100h/sample_index.jsonl`
  (the 13k mosel clips of the Phase 4a PoC). Pass `--all` to operate on
  every mosel_hu_voxpopuli row.

Run:
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python \
      bin/refine_mosel_boundaries.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
MANIFEST_PATH = DATA_ROOT / "processed" / "manifests" / "manifest.jsonl"
SAMPLE_INDEX_DEFAULT = Path(
    "/home/cseti/data2/Develop/Github-cseti/cseti-os/projects/hu-speech-corpus/"
    "notes/poc_100h/sample_index.jsonl"
)

OUT_AUDIO_DIR = DATA_ROOT / "processed" / "normalization" / "mosel_refined"
OUT_SIDECAR = DATA_ROOT / "processed" / "normalization" / "mosel_boundary_refined.jsonl"

TARGET_SR = 16000
PARENT_SR = 16000  # voxpopuli unlabeled is 16k mono OGG vorbis

# Per-worker state
_model = None
_parent_cache: dict[str, tuple[np.ndarray, int]] = {}

_FNAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(utt: str) -> str:
    return _FNAME_SAFE.sub("_", utt)


# ============================================================
# Neighbor-aware bounds for a single mosel clip
# ============================================================

def parent_session_path(parent_audio_path_str: str) -> Path | None:
    """Convert the manifest's `parent_audio_path` (which may contain a
    glob '*') to a concrete file path. The VoxPopuli unlabeled raw audios
    live under raw/voxpopuli_hu_unlabeled/raw_audios/hu/YYYY/...ogg."""
    if not parent_audio_path_str:
        return None
    p = parent_audio_path_str
    if p.startswith("raw/"):
        p = str(DATA_ROOT / p)
    # The manifest stores '/*/' for the year segment — we need to find which year
    if "/*/" in p:
        prefix, suffix = p.split("/*/", 1)
        prefix_dir = Path(prefix)
        if not prefix_dir.exists():
            return None
        # Try each year subdir
        for year_dir in sorted(prefix_dir.iterdir(), reverse=True):
            if not year_dir.is_dir():
                continue
            candidate = year_dir / suffix
            if candidate.exists():
                return candidate
        return None
    return Path(p)


def build_session_index(manifest_path: Path) -> dict[str, list[dict]]:
    """For each VoxPopuli session, return the list of mosel utterances
    sorted by segment_start_sec. Used to compute max-padding bounds."""
    by_session: dict[str, list[dict]] = defaultdict(list)
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") != "mosel_hu_voxpopuli":
                continue
            sess = r.get("parent_session_id")
            if not sess:
                continue
            by_session[sess].append({
                "utterance_id": r["utterance_id"],
                "start": r.get("segment_start_sec"),
                "end": r.get("segment_end_sec"),
            })
    for sess in by_session:
        by_session[sess].sort(key=lambda x: x["start"] or 0.0)
    return by_session


def neighbor_bounds(session_utts: list[dict], utt_id: str,
                    pad_sec: float) -> tuple[float, float] | None:
    """For the target utterance in this session, return (max_left_pad,
    max_right_pad) in seconds — bounded by adjacent utterances so we
    never extend INTO a neighbor."""
    idx = next((i for i, u in enumerate(session_utts) if u["utterance_id"] == utt_id), None)
    if idx is None:
        return None
    target = session_utts[idx]
    if target["start"] is None or target["end"] is None:
        return None
    # left gap = our_start - prev_end; 0 if no previous or overlap
    left_gap = target["start"] - 0.0
    if idx > 0:
        prev = session_utts[idx - 1]
        if prev["end"] is not None:
            left_gap = max(0.0, target["start"] - prev["end"])
    # right gap = next_start - our_end; pad_sec if no next
    right_gap = pad_sec * 2.0  # if last in session, allow full pad
    if idx + 1 < len(session_utts):
        nxt = session_utts[idx + 1]
        if nxt["start"] is not None:
            right_gap = max(0.0, nxt["start"] - target["end"])
    max_left = min(pad_sec, left_gap - 0.001)  # 1ms safety margin
    max_right = min(pad_sec, right_gap - 0.001)
    return (max(0.0, max_left), max(0.0, max_right))


# ============================================================
# Worker
# ============================================================

def _init_worker():
    """Each worker: load Silero VAD once, pin torch BLAS to 1 thread."""
    global _model
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    _model = load_silero_vad()


def _load_parent(parent_path: Path) -> tuple[np.ndarray, int] | None:
    """Load + cache a parent session audio in the worker."""
    key = str(parent_path)
    if key in _parent_cache:
        return _parent_cache[key]
    try:
        audio, sr = sf.read(str(parent_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        _parent_cache[key] = (audio, sr)
        # Keep cache bounded (LRU-ish): if too large, drop one
        if len(_parent_cache) > 4:
            first_key = next(iter(_parent_cache))
            if first_key != key:
                del _parent_cache[first_key]
        return _parent_cache[key]
    except Exception:
        return None


def _find_silence_near(
    audio: np.ndarray, target_sample: int, search_half_window: int,
    direction: str, segs: list[dict]
) -> int:
    """Return the sample index of a silence point near `target_sample`.

    direction = "start": prefer the LATEST silence within [target-window, target+window]
                         that is still BEFORE the first speech segment touching the target
    direction = "end":   prefer the EARLIEST silence after the last speech segment
                         touching the target

    Falls back to `target_sample` if no clean silence is found."""
    if not segs:
        return target_sample
    lo = max(0, target_sample - search_half_window)
    hi = min(len(audio), target_sample + search_half_window)

    if direction == "start":
        # Find the speech segment that contains or is closest after target_sample
        candidate_seg = None
        for seg in segs:
            if seg["start"] >= lo and seg["start"] <= hi:
                candidate_seg = seg
                break
        if candidate_seg is None:
            return target_sample
        # Move to just before that segment's start — but not before lo
        new_start = max(lo, candidate_seg["start"] - int(0.05 * TARGET_SR))
        return new_start

    if direction == "end":
        candidate_seg = None
        # Find the speech segment that contains or is closest before target_sample
        for seg in reversed(segs):
            if seg["end"] >= lo and seg["end"] <= hi:
                candidate_seg = seg
                break
        if candidate_seg is None:
            return target_sample
        new_end = min(hi, candidate_seg["end"] + int(0.05 * TARGET_SR))
        return new_end

    return target_sample


def _worker_refine(args: dict) -> dict:
    """Refine one clip. args has: utt_id, parent_path, orig_start_sec,
    orig_end_sec, max_left_pad, max_right_pad, out_audio_path."""
    import torch
    from silero_vad import get_speech_timestamps

    utt_id = args["utt_id"]
    parent_path = Path(args["parent_path"])
    orig_start = args["orig_start_sec"]
    orig_end = args["orig_end_sec"]
    max_left = args["max_left_pad"]
    max_right = args["max_right_pad"]
    out_path = Path(args["out_audio_path"])

    if not parent_path.exists():
        return {"utterance_id": utt_id, "refined": False,
                "refine_error": "parent_audio_not_found"}

    parent = _load_parent(parent_path)
    if parent is None:
        return {"utterance_id": utt_id, "refined": False,
                "refine_error": "parent_load_failed"}
    parent_audio, parent_sr = parent

    pad_start = max(0.0, orig_start - max_left)
    pad_end = min(len(parent_audio) / parent_sr, orig_end + max_right)
    pad_start_sample = int(pad_start * parent_sr)
    pad_end_sample = int(pad_end * parent_sr)
    if pad_end_sample <= pad_start_sample:
        return {"utterance_id": utt_id, "refined": False,
                "refine_error": "padded_window_empty"}

    padded = parent_audio[pad_start_sample:pad_end_sample].astype(np.float32)

    try:
        t = torch.from_numpy(padded).float()
        segs = get_speech_timestamps(
            t, _model, sampling_rate=parent_sr, return_seconds=False,
            min_speech_duration_ms=80,
            min_silence_duration_ms=150,
        )
    except Exception as ex:
        return {"utterance_id": utt_id, "refined": False,
                "refine_error": f"vad_failed: {str(ex)[:80]}"}

    if not segs:
        # No speech detected — keep original boundaries; don't write a new clip
        return {"utterance_id": utt_id, "refined": False,
                "refine_error": "no_speech_in_padded_region"}

    # Compute target boundaries IN THE PADDED COORDINATE FRAME
    orig_start_in_pad = int((orig_start - pad_start) * parent_sr)
    orig_end_in_pad = int((orig_end - pad_start) * parent_sr)
    search_half_window = int(max(max_left, max_right) * parent_sr)

    new_start_in_pad = _find_silence_near(
        padded, orig_start_in_pad, search_half_window, "start", segs
    )
    new_end_in_pad = _find_silence_near(
        padded, orig_end_in_pad, search_half_window, "end", segs
    )
    if new_end_in_pad <= new_start_in_pad:
        return {"utterance_id": utt_id, "refined": False,
                "refine_error": "refined_window_empty"}

    refined_audio = padded[new_start_in_pad:new_end_in_pad]
    new_start_sec = pad_start + new_start_in_pad / parent_sr
    new_end_sec = pad_start + new_end_in_pad / parent_sr

    # Write refined OGG via soundfile (vorbis encoder)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(out_path), refined_audio, parent_sr, format="OGG",
                 subtype="VORBIS")
    except Exception as ex:
        return {"utterance_id": utt_id, "refined": False,
                "refine_error": f"write_failed: {str(ex)[:80]}"}

    return {
        "utterance_id": utt_id,
        "refined": True,
        "refined_audio_path": str(out_path),
        "orig_start_sec": round(orig_start, 3),
        "orig_end_sec": round(orig_end, 3),
        "new_start_sec": round(new_start_sec, 3),
        "new_end_sec": round(new_end_sec, 3),
        "change_start_ms": round((new_start_sec - orig_start) * 1000, 1),
        "change_end_ms": round((new_end_sec - orig_end) * 1000, 1),
        "n_vad_segments": len(segs),
    }


# ============================================================
# Main
# ============================================================

def load_done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["utterance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--sample-index", type=Path, default=SAMPLE_INDEX_DEFAULT,
                   help="JSONL with utterance_ids to operate on (default "
                        "PoC sample). Ignored when --all is set.")
    p.add_argument("--all", action="store_true",
                   help="Refine every mosel_hu_voxpopuli row (heavy).")
    p.add_argument("--out-dir", type=Path, default=OUT_AUDIO_DIR)
    p.add_argument("--out-sidecar", type=Path, default=OUT_SIDECAR)
    p.add_argument("--pad-sec", type=float, default=0.5,
                   help="Padding on each side before VAD trim (default 0.5s).")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N clips this run (for validation).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_sidecar.parent.mkdir(parents=True, exist_ok=True)

    # Scope: which utterance_ids to refine
    scope_ids: set | None = None
    if not args.all:
        if not args.sample_index.exists():
            print(f"[ERROR] sample-index not found: {args.sample_index}",
                  file=sys.stderr)
            return 1
        scope_ids = set()
        with args.sample_index.open(encoding="utf-8") as f:
            for line in f:
                try:
                    scope_ids.add(json.loads(line)["utterance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[init] scope: {len(scope_ids):,} utterance_ids "
              f"from {args.sample_index.name}", file=sys.stderr)
    else:
        print("[init] scope: ALL mosel_hu_voxpopuli rows", file=sys.stderr)

    done_ids = load_done_ids(args.out_sidecar)
    print(f"[init] {len(done_ids):,} clips already refined", file=sys.stderr)

    print(f"[init] indexing parent sessions...", file=sys.stderr)
    by_session = build_session_index(args.manifest)
    print(f"[init] {len(by_session):,} parent sessions indexed", file=sys.stderr)

    print(f"[init] building work list...", file=sys.stderr)
    work_items = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") != "mosel_hu_voxpopuli":
                continue
            utt = r["utterance_id"]
            if scope_ids is not None and utt not in scope_ids:
                continue
            if utt in done_ids:
                continue
            parent_p = parent_session_path(r.get("parent_audio_path", ""))
            if parent_p is None:
                continue
            sess = r.get("parent_session_id")
            sess_utts = by_session.get(sess, [])
            bounds = neighbor_bounds(sess_utts, utt, args.pad_sec)
            if bounds is None:
                continue
            max_left, max_right = bounds
            work_items.append({
                "utt_id": utt,
                "parent_path": str(parent_p),
                "orig_start_sec": r["segment_start_sec"],
                "orig_end_sec": r["segment_end_sec"],
                "max_left_pad": max_left,
                "max_right_pad": max_right,
                "out_audio_path": str(args.out_dir / (safe_filename(utt) + ".ogg")),
            })

    # Sort by parent_path so each worker hits the same session repeatedly
    # → parent cache hits
    work_items.sort(key=lambda x: x["parent_path"])

    if args.limit:
        work_items = work_items[:args.limit]
        print(f"[init] limited to {len(work_items):,} clips", file=sys.stderr)

    print(f"[init] {len(work_items):,} clips to refine, {args.n_workers} workers",
          file=sys.stderr)
    if not work_items:
        print("[done] nothing to do", file=sys.stderr)
        return 0

    t0 = time.time()
    n_done = 0
    n_refined = 0
    n_errors = 0
    progress_every = max(200, len(work_items) // 100)

    with args.out_sidecar.open("a", encoding="utf-8") as out:
        with Pool(processes=args.n_workers, initializer=_init_worker) as pool:
            for result in pool.imap_unordered(_worker_refine, work_items,
                                              chunksize=4):
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                n_done += 1
                if result.get("refined"):
                    n_refined += 1
                if "refine_error" in result:
                    n_errors += 1
                if n_done % progress_every == 0:
                    out.flush()
                    rate = n_done / (time.time() - t0)
                    eta = (len(work_items) - n_done) / rate / 60 if rate > 0 else 0
                    print(f"[progress] {n_done:,}/{len(work_items):,} "
                          f"({n_refined:,} refined, {n_errors:,} errors, "
                          f"{rate:.0f} clips/s, ETA {eta:.1f} min)",
                          file=sys.stderr)

    print()
    print("=== Boundary refinement summary ===")
    print(f"Processed:   {n_done:,}")
    print(f"Refined:     {n_refined:,} ({n_refined/n_done*100:.1f}%)")
    print(f"Errors:      {n_errors:,}")
    print(f"Time:        {(time.time()-t0)/60:.1f} min")
    print(f"Audio out:   {args.out_dir}")
    print(f"Sidecar:     {args.out_sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
