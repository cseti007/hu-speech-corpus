#!/usr/bin/env python3
"""Audit per-clip boundary quality — detect mid-utterance cuts.

For each clip:
  1. Decode audio (16 kHz mono if not already)
  2. Run Silero VAD with loose params (min_silence_duration_ms=100)
  3. Check whether a speech segment touches the first or last `--edge-ms`
     of the clip (default 50ms)
  4. Flag as `boundary_cut_start` and/or `boundary_cut_end`

Outputs:
  - Per-clip sidecar JSONL at `processed/quality/clip_boundary_audit.jsonl`
  - Aggregated summary printed to stderr (overall % cut, per-source breakdown)

Motivated by the 2026-05-23 manual audio review that revealed inherited
boundary defects in `mosel_hu_voxpopuli`. Those clips were cut at
`unlabelled_v2.tsv.gz` boundaries (Facebook's forced alignment), not by
our own VAD — so the audit measures the IMPORTED defect rate.

Run:
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python \
      bin/audit_clip_boundaries.py --source mosel_hu_voxpopuli --limit 5000

  # Full mosel audit (~6-10 hours at 8 workers):
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python \
      bin/audit_clip_boundaries.py --source mosel_hu_voxpopuli --no-limit

Idempotent: utterance_ids already in the sidecar are skipped on re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
MANIFEST_PATH = DATA_ROOT / "processed" / "manifests" / "manifest.jsonl"
OUT_PATH = DATA_ROOT / "processed" / "quality" / "clip_boundary_audit.jsonl"

TARGET_SR = 16000

# Per-worker state
_model = None


# ============================================================
# Worker init + per-clip audit
# ============================================================

def _init_worker():
    """Load Silero VAD once per worker. Pin torch BLAS to 1 thread."""
    global _model
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    _model = load_silero_vad()


def _load_audio(row: dict, parent_cache: dict) -> np.ndarray | None:
    """Return mono float32 PCM @ 16 kHz for a manifest row. Mirrors the
    loader in `bin/quality_tier2.py`."""
    source = row.get("source", "")
    audio_path = row.get("audio_path")
    if not audio_path:
        return None

    if source in {"mosel_hu_voxpopuli", "librivox_hu", "podcasts_hu_cc",
                  "voxpopuli_unlabeled_gap"}:
        try:
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        except Exception:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            return None
        return audio

    if source == "yodas2_hu000":
        if audio_path not in parent_cache:
            try:
                wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                parent_cache[audio_path] = (wav, sr)
            except Exception:
                parent_cache[audio_path] = (None, 0)
        wav, sr = parent_cache[audio_path]
        if wav is None:
            return None
        s = int(row["segment_start_sec"] * sr)
        e = int(row["segment_end_sec"] * sr)
        e = min(e, len(wav))
        if e <= s:
            return None
        clip = wav[s:e]
        if sr != TARGET_SR:
            from math import gcd
            g = gcd(sr, TARGET_SR)
            up = TARGET_SR // g
            down = sr // g
            try:
                from scipy.signal import resample_poly
                clip = resample_poly(clip, up, down).astype(np.float32)
            except ImportError:
                clip = clip[::max(1, sr // TARGET_SR)].astype(np.float32)
        return clip

    return None


def _worker_audit(args):
    """Audit a single clip. args = (row, edge_ms). Returns dict."""
    row, edge_ms = args
    import torch
    from silero_vad import get_speech_timestamps

    parent_cache: dict = {}
    audio = _load_audio(row, parent_cache)
    if audio is None or len(audio) < int(0.3 * TARGET_SR):
        return {"utterance_id": row["utterance_id"],
                "boundary_audit_error": "audio_too_short"}

    try:
        t = torch.from_numpy(audio).float()
        segs = get_speech_timestamps(
            t, _model, sampling_rate=TARGET_SR, return_seconds=False,
            min_speech_duration_ms=80,
            min_silence_duration_ms=100,
        )
    except Exception as ex:
        return {"utterance_id": row["utterance_id"],
                "boundary_audit_error": str(ex)[:120]}

    if not segs:
        # No speech at all — likely a silent / noise-only clip; flag
        return {
            "utterance_id": row["utterance_id"],
            "boundary_cut_start": False,
            "boundary_cut_end": False,
            "no_speech": True,
            "n_speech_segments": 0,
        }

    edge_samples = int(edge_ms * TARGET_SR / 1000)
    clip_len = len(audio)

    # Start cut: a speech segment begins within `edge_samples` of sample 0
    first_seg_start = segs[0]["start"]
    boundary_cut_start = first_seg_start < edge_samples

    # End cut: a speech segment ends within `edge_samples` of clip_len
    last_seg_end = segs[-1]["end"]
    boundary_cut_end = (clip_len - last_seg_end) < edge_samples

    # Mid-word cut detection: VAD says speech is at the boundary AND the
    # last/first 100ms has high RMS (no natural fade-out / fade-in).
    # Natural utterance ends typically decay to < -30 dBFS; mid-word cuts
    # stay near -15..-25 dBFS at the cut point.
    win_samples = int(0.1 * TARGET_SR)  # 100ms

    def _rms_dbfs(chunk: np.ndarray) -> float:
        if len(chunk) == 0:
            return -240.0
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms <= 1e-12:
            return -240.0
        return 20.0 * np.log10(rms)

    start_rms = _rms_dbfs(audio[:win_samples])
    end_rms = _rms_dbfs(audio[-win_samples:])

    mid_word_cut_start = boundary_cut_start and start_rms > -25.0
    mid_word_cut_end = boundary_cut_end and end_rms > -25.0

    return {
        "utterance_id": row["utterance_id"],
        "boundary_cut_start": bool(boundary_cut_start),
        "boundary_cut_end": bool(boundary_cut_end),
        "mid_word_cut_start": bool(mid_word_cut_start),
        "mid_word_cut_end": bool(mid_word_cut_end),
        "first_speech_offset_ms": round(first_seg_start * 1000.0 / TARGET_SR, 1),
        "last_speech_to_end_ms": round((clip_len - last_seg_end) * 1000.0 / TARGET_SR, 1),
        "start_100ms_rms_dbfs": round(start_rms, 1),
        "end_100ms_rms_dbfs": round(end_rms, 1),
        "n_speech_segments": len(segs),
    }


# ============================================================
# Manifest iteration + main
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


def iter_candidates(manifest: Path, source_filter: str | None,
                    done_ids: set):
    """Stream manifest rows that match the source filter and have audio."""
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if source_filter and r.get("source") != source_filter:
                continue
            if r["utterance_id"] in done_ids:
                continue
            if not r.get("audio_path"):
                continue
            yield {
                "utterance_id": r["utterance_id"],
                "source": r["source"],
                "audio_path": r["audio_path"],
                "segment_start_sec": r.get("segment_start_sec"),
                "segment_end_sec": r.get("segment_end_sec"),
            }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--source", default="mosel_hu_voxpopuli",
                   help="source key to audit (default mosel_hu_voxpopuli; "
                        "pass empty string to audit all sources)")
    p.add_argument("--limit", type=int, default=5000,
                   help="random-sample N clips for the audit (default 5000); "
                        "use --no-limit to process all candidates")
    p.add_argument("--no-limit", action="store_true",
                   help="disable --limit and process all matching clips")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--edge-ms", type=int, default=50,
                   help="how close to the clip edge counts as a boundary cut "
                        "(default 50ms)")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    source_filter = args.source if args.source else None
    print(f"[init] manifest={args.manifest.name}, source={source_filter or 'ALL'}",
          file=sys.stderr)
    print(f"[init] sidecar={args.out}", file=sys.stderr)

    done_ids = load_done_ids(args.out)
    print(f"[init] {len(done_ids):,} clips already audited", file=sys.stderr)

    print("[init] scanning manifest for candidates...", file=sys.stderr)
    candidates = list(iter_candidates(args.manifest, source_filter, done_ids))
    print(f"[init] {len(candidates):,} candidates remaining", file=sys.stderr)

    if not args.no_limit and args.limit and len(candidates) > args.limit:
        rng = random.Random(args.seed)
        candidates = rng.sample(candidates, args.limit)
        print(f"[init] subsampled to {len(candidates):,} clips (seed={args.seed})",
              file=sys.stderr)

    if not candidates:
        print("[done] nothing to do", file=sys.stderr)
        return 0

    t0 = time.time()
    n_done = 0
    n_errors = 0
    per_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "cut_start": 0, "cut_end": 0,
                 "mw_start": 0, "mw_end": 0, "mw_either": 0,
                 "no_speech": 0, "errors": 0,
                 "either_cut": 0, "both_cut": 0}
    )
    progress_every = max(500, len(candidates) // 100)

    with args.out.open("a", encoding="utf-8") as out:
        with Pool(processes=args.n_workers,
                  initializer=_init_worker) as pool:
            work_iter = ((c, args.edge_ms) for c in candidates)
            for result in pool.imap_unordered(_worker_audit, work_iter,
                                              chunksize=10):
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                n_done += 1

                # Look up source for the aggregator (cheap; build dict once)
                # Inline lookup is N*N; instead read from the candidates dict
                # we built above
                utt = result["utterance_id"]
                # Build candidate index lazily
                src = None
                # Quick linear: since N is small (~5000) this is OK; but
                # for 2M we want O(1). Build a dict outside the pool.
                # We'll do that below by passing source in result instead.

                if "boundary_audit_error" in result:
                    n_errors += 1

                if n_done % progress_every == 0:
                    now = time.time()
                    rate = n_done / (now - t0)
                    remaining = (len(candidates) - n_done) / rate if rate > 0 else 0
                    print(f"[progress] {n_done:,}/{len(candidates):,} "
                          f"({rate:.0f} clips/s, ETA {remaining/60:.1f} min, "
                          f"{n_errors} errors)", file=sys.stderr)
                out.flush()

    # Re-pass over the sidecar to build the summary (since `src` wasn't
    # in the per-clip result; cheaper than carrying it through the pool).
    cand_src = {c["utterance_id"]: c["source"] for c in candidates}
    with args.out.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            utt = r.get("utterance_id")
            src = cand_src.get(utt)
            if src is None:
                continue
            d = per_source[src]
            d["total"] += 1
            if "boundary_audit_error" in r:
                d["errors"] += 1
                continue
            if r.get("no_speech"):
                d["no_speech"] += 1
                continue
            cs = r.get("boundary_cut_start", False)
            ce = r.get("boundary_cut_end", False)
            if cs:
                d["cut_start"] += 1
            if ce:
                d["cut_end"] += 1
            if cs or ce:
                d["either_cut"] += 1
            if cs and ce:
                d["both_cut"] += 1
            mws = r.get("mid_word_cut_start", False)
            mwe = r.get("mid_word_cut_end", False)
            if mws:
                d["mw_start"] += 1
            if mwe:
                d["mw_end"] += 1
            if mws or mwe:
                d["mw_either"] += 1

    print()
    print("=== Clip boundary audit summary ===")
    print(f"Edge threshold: {args.edge_ms} ms")
    print(f"Sample size: {n_done:,} clips processed this run, "
          f"{n_done + len(done_ids):,} total in sidecar")
    print()
    print("Boundary-touch = VAD speech within edge_ms of clip boundary.")
    print("Mid-word cut    = boundary-touch AND last/first 100ms RMS > -25 dBFS")
    print("                  (no natural fade — likely truncated mid-word).")
    print()
    print(f"{'source':30s} {'total':>8s} "
          f"{'touch_start%':>13s} {'touch_end%':>11s} "
          f"{'mw_start%':>11s} {'mw_end%':>9s} "
          f"{'mw_either%':>11s}")
    print("-" * 96)
    for src in sorted(per_source.keys()):
        d = per_source[src]
        if d["total"] == 0:
            continue
        n = d["total"]
        print(f"{src:30s} {n:>8,} "
              f"{d['cut_start']/n*100:>12.2f}% "
              f"{d['cut_end']/n*100:>10.2f}% "
              f"{d['mw_start']/n*100:>10.2f}% "
              f"{d['mw_end']/n*100:>8.2f}% "
              f"{d['mw_either']/n*100:>10.2f}%")

    print()
    print(f"Output: {args.out}")
    print(f"Time: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
