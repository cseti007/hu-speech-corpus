#!/usr/bin/env python3
"""Phase 3.5 v2: full-clip language analysis with foreign region detection.

Extension of `audit_clip_language_purity.py` that handles the new
`voxpopuli_resegmented` chunks (3-30 sec) AND detects foreign content
anywhere in the clip — prefix, suffix, or middle — not just the leading
5 seconds.

Two-pass design (chosen 2026-05-25 to keep GPU cost tractable):

  Pass 1: whole-clip LID on EVERY clip (~1 LID input each)
    - Output: whole_clip_top1, whole_clip_hu_prob
    - Flag clips where top1 != 'hu' OR hu_prob < HU_PROB_THRESHOLD as
      `needs_pass2 = true`
    - Clean HU clips (top1='hu', high prob) skip Pass 2 entirely

  Pass 2: per-clip windowed LID + Silero VAD on flagged clips only
    - 1-sec windows, 0.5-sec stride over the full duration
    - Run Silero VAD on the clip to find internal silence gaps
    - Group consecutive same-language windows into regions
    - Snap region boundaries to nearest VAD silence (sub-sec precision)
    - Output: language_regions = [(start_sec, end_sec, lang), ...]

Output sidecar `processed/quality/clip_language_purity_v2.jsonl`:
  {
    "utterance_id": "...",
    "whole_clip_top1": "hu",          // pass 1
    "whole_clip_hu_prob": 0.97,       // pass 1
    "needs_pass2": false,             // pass 1
    "language_regions": null,         // pass 2 (null if pass1 was sufficient)
    "n_foreign_regions": 0,           // pass 2 derived
    "foreign_duration_sec": 0.0,      // pass 2 derived
    "first_hu_start_sec": null,       // pass 2 derived (offset of first HU region)
    "last_hu_end_sec": null,          // pass 2 derived
  }

For flagged clips a typical record looks like:
  {
    "utterance_id": "voxpopuli_resegmented/.../...",
    "whole_clip_top1": "sk",
    "whole_clip_hu_prob": 0.04,
    "needs_pass2": true,
    "language_regions": [[0.0, 4.6, "sk"], [4.6, 18.3, "hu"]],
    "n_foreign_regions": 1,
    "foreign_duration_sec": 4.6,
    "first_hu_start_sec": 4.6,
    "last_hu_end_sec": 18.3,
  }

NO TRIMMING — this tool only writes metadata. The optional `trim_foreign.py`
tool consumes `language_regions` to do the actual audio cuts (deferred until
visual validation in the curator).

Idempotent: utterance_ids already in the sidecar are skipped.

Run (audio_ds env has speechbrain + GPU torch + silero_vad + soundfile):
  /media/cseti/datassd/conda/miniconda3/envs/audio_ds/bin/python \
      bin/audit_clip_language_purity_v2.py \
      --input /home/cseti/datassd2/hu-speech-corpus/processed/manifests/manifest_v5.jsonl \
      --source voxpopuli_resegmented
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
MANIFEST_DEFAULT = DATA_ROOT / "processed" / "manifests" / "manifest_v5.jsonl"
OUT_PATH = DATA_ROOT / "processed" / "quality" / "clip_language_purity_v2.jsonl"

TARGET_SR = 16000
HU_PROB_THRESHOLD = 0.85       # below this, pass 2 runs
WINDOW_SEC = 1.0               # window size for pass 2
WINDOW_STRIDE_SEC = 0.5        # stride for pass 2
WHOLE_CLIP_CAP_SEC = 10.0      # cap audio length for whole-clip LID (memory)
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 300


# ============================================================
# Audio loading
# ============================================================

def load_audio(path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        # Resample to TARGET_SR (e.g. CV25 48 kHz MP3 → 16 kHz).
        from math import gcd
        g = gcd(sr, TARGET_SR)
        up = TARGET_SR // g
        down = sr // g
        try:
            from scipy.signal import resample_poly
            audio = resample_poly(audio, up, down).astype(np.float32)
        except ImportError:
            audio = audio[::max(1, sr // TARGET_SR)].astype(np.float32)
    return audio


# ============================================================
# Sidecar I/O
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


# ============================================================
# Pass 1: whole-clip LID, GPU-batched
# ============================================================

def pass1_whole_clip(work_rows: list[dict], lid_model, hu_idx: int,
                    out_sidecar, batch_size: int = 24,
                    hu_prob_threshold: float = HU_PROB_THRESHOLD,
                    progress_every: int = 2000) -> dict:
    """Whole-clip LID over every work_row. Writes a partial record per clip
    (just whole_clip_* fields + needs_pass2 flag). Returns stats dict."""
    import torch

    n_done = 0
    n_errors = 0
    n_flagged = 0
    t0 = time.time()

    i = 0
    while i < len(work_rows):
        batch_rows = work_rows[i:i + batch_size]
        i += batch_size

        audios: list[np.ndarray] = []
        valid_rows: list[dict] = []
        for row in batch_rows:
            audio = load_audio(row["audio_path"])
            if audio is None or len(audio) < int(0.5 * TARGET_SR):
                rec = {"utterance_id": row["utterance_id"],
                       "pass1_error": "audio_load_failed"}
                out_sidecar.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_errors += 1
                n_done += 1
                continue
            audios.append(audio[: int(WHOLE_CLIP_CAP_SEC * TARGET_SR)])
            valid_rows.append(row)

        if not audios:
            continue

        try:
            max_len = max(len(a) for a in audios)
            padded = np.zeros((len(audios), max_len), dtype=np.float32)
            wav_lens_rel = np.zeros(len(audios), dtype=np.float32)
            for j, a in enumerate(audios):
                padded[j, :len(a)] = a
                wav_lens_rel[j] = len(a) / max_len
            signal = torch.from_numpy(padded)
            wav_lens = torch.from_numpy(wav_lens_rel)
            with torch.no_grad():
                pred = lid_model.classify_batch(signal, wav_lens)
            out_prob = pred[0]
            text_labs = pred[3]
            probs = torch.softmax(out_prob, dim=1)
        except Exception as ex:
            for row in valid_rows:
                rec = {"utterance_id": row["utterance_id"],
                       "pass1_error": str(ex)[:120]}
                out_sidecar.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_errors += 1
                n_done += 1
            out_sidecar.flush()
            continue

        for j, row in enumerate(valid_rows):
            text_lab = text_labs[j]
            top1 = text_lab.split(":")[0].strip() if ":" in text_lab else text_lab
            hu_prob = float(probs[j, hu_idx])
            needs_pass2 = (top1 != "hu") or (hu_prob < hu_prob_threshold)
            if needs_pass2:
                n_flagged += 1
            rec = {
                "utterance_id": row["utterance_id"],
                "whole_clip_top1": top1,
                "whole_clip_hu_prob": round(hu_prob, 4),
                "needs_pass2": needs_pass2,
            }
            out_sidecar.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1

        if n_done % progress_every < batch_size:
            out_sidecar.flush()
            rate = n_done / (time.time() - t0)
            eta = (len(work_rows) - n_done) / rate / 60 if rate > 0 else 0
            print(f"[pass1] {n_done:,}/{len(work_rows):,} "
                  f"({n_flagged:,} flagged for pass2, {n_errors:,} errors, "
                  f"{rate:.0f} clips/s, ETA {eta:.1f} min)", file=sys.stderr)

    return {"n_done": n_done, "n_errors": n_errors, "n_flagged": n_flagged}


# ============================================================
# Pass 2: windowed LID + VAD silence snap
# ============================================================

def windowed_lid(audio: np.ndarray, lid_model, hu_idx: int,
                 window_sec: float = WINDOW_SEC,
                 stride_sec: float = WINDOW_STRIDE_SEC,
                 lid_batch_size: int = 16) -> list[tuple[float, str, float]]:
    """Run windowed LID over the full clip. Returns
    [(start_sec, top1_lang, hu_prob), ...]."""
    import torch

    duration = len(audio) / TARGET_SR
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    starts = list(range(0, max(1, len(audio) - win_samples + 1), stride_samples))
    if not starts:
        return []

    results: list[tuple[float, str, float]] = []
    for chunk_start in range(0, len(starts), lid_batch_size):
        sub_starts = starts[chunk_start:chunk_start + lid_batch_size]
        windows = [audio[s:s + win_samples] for s in sub_starts]
        max_len = max(len(w) for w in windows)
        padded = np.zeros((len(windows), max_len), dtype=np.float32)
        wav_lens_rel = np.zeros(len(windows), dtype=np.float32)
        for k, w in enumerate(windows):
            padded[k, :len(w)] = w
            wav_lens_rel[k] = len(w) / max_len
        signal = torch.from_numpy(padded)
        wav_lens = torch.from_numpy(wav_lens_rel)
        with torch.no_grad():
            pred = lid_model.classify_batch(signal, wav_lens)
        out_prob = pred[0]
        text_labs = pred[3]
        probs = torch.softmax(out_prob, dim=1)
        for k, s in enumerate(sub_starts):
            text_lab = text_labs[k]
            top1 = text_lab.split(":")[0].strip() if ":" in text_lab else text_lab
            hu_p = float(probs[k, hu_idx])
            results.append((s / TARGET_SR, top1, hu_p))
    return results


def group_into_regions(
    window_results: list[tuple[float, str, float]],
    clip_duration_sec: float,
) -> list[tuple[float, float, str]]:
    """Group consecutive same-language windows into regions.
    Returns [(start_sec, end_sec, lang), ...] covering the whole clip."""
    if not window_results:
        return []

    regions: list[tuple[float, float, str]] = []
    cur_start = 0.0
    cur_lang = window_results[0][1]
    for i, (start, lang, _hu_p) in enumerate(window_results[1:], start=1):
        if lang != cur_lang:
            # End the current region at the midpoint between this window and prev
            prev_start = window_results[i - 1][0]
            split_point = (prev_start + WINDOW_SEC + start) / 2.0
            split_point = min(split_point, clip_duration_sec)
            regions.append((round(cur_start, 3), round(split_point, 3), cur_lang))
            cur_start = split_point
            cur_lang = lang
    regions.append((round(cur_start, 3), round(clip_duration_sec, 3), cur_lang))
    return regions


def snap_to_silences(
    regions: list[tuple[float, float, str]],
    vad_segs: list[dict],
    clip_duration_sec: float,
) -> list[tuple[float, float, str]]:
    """Snap each internal region boundary to the nearest VAD silence midpoint.

    VAD segs are speech intervals; gaps between consecutive segs (or before
    the first / after the last) are silence intervals. We move each boundary
    to the nearest silence-midpoint within a search radius."""
    if not regions or len(regions) < 2:
        return regions

    # Build silence midpoints from VAD speech segments
    silence_mids: list[float] = []
    prev_end_sec = 0.0
    for seg in vad_segs:
        seg_start = seg["start"] / TARGET_SR
        seg_end = seg["end"] / TARGET_SR
        if seg_start > prev_end_sec + 0.05:  # at least 50ms gap
            silence_mids.append((prev_end_sec + seg_start) / 2.0)
        prev_end_sec = seg_end
    if clip_duration_sec > prev_end_sec + 0.05:
        silence_mids.append((prev_end_sec + clip_duration_sec) / 2.0)

    if not silence_mids:
        return regions

    SEARCH_RADIUS_SEC = 0.75  # snap if a silence midpoint is within ±0.75s
    snapped: list[tuple[float, float, str]] = []
    new_starts = [regions[0][0]]
    for i in range(len(regions) - 1):
        boundary = regions[i][1]
        # Find the closest silence midpoint
        closest = min(silence_mids, key=lambda m: abs(m - boundary))
        if abs(closest - boundary) <= SEARCH_RADIUS_SEC:
            new_starts.append(round(closest, 3))
        else:
            new_starts.append(round(boundary, 3))
    new_starts.append(round(clip_duration_sec, 3))

    for i, (_, _, lang) in enumerate(regions):
        snapped.append((new_starts[i], new_starts[i + 1], lang))
    return snapped


def _init_pass2_workers():
    """One-shot per worker: load both LID + Silero VAD on GPU/CPU."""
    # Used only if we go multi-process for pass 2. For now we run pass 2
    # in the main process since GPU is already saturated by pass 1.
    pass


def pass2_for_flagged_clip(
    row: dict, audio: np.ndarray, lid_model, hu_idx: int, vad_model,
) -> dict:
    """Full per-clip language region analysis. Returns the record to write."""
    import torch
    from silero_vad import get_speech_timestamps

    clip_duration = len(audio) / TARGET_SR

    # Windowed LID over the full clip
    windows = windowed_lid(audio, lid_model, hu_idx)

    # Silero VAD on the full clip
    try:
        t = torch.from_numpy(audio).float()
        vad_segs = get_speech_timestamps(
            t, vad_model, sampling_rate=TARGET_SR, return_seconds=False,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        )
    except Exception:
        vad_segs = []

    # Group into language regions
    regions = group_into_regions(windows, clip_duration)

    # Snap region boundaries to nearest VAD silences
    if regions and vad_segs:
        regions = snap_to_silences(regions, vad_segs, clip_duration)

    # Derived stats
    n_foreign = sum(1 for _, _, lang in regions if lang != "hu")
    foreign_dur = sum(end - start for start, end, lang in regions if lang != "hu")
    hu_regions = [(s, e) for s, e, l in regions if l == "hu"]
    first_hu_start = hu_regions[0][0] if hu_regions else None
    last_hu_end = hu_regions[-1][1] if hu_regions else None

    return {
        "language_regions": [list(r) for r in regions],
        "n_foreign_regions": n_foreign,
        "foreign_duration_sec": round(foreign_dur, 3),
        "first_hu_start_sec": first_hu_start,
        "last_hu_end_sec": last_hu_end,
        "n_windows_scanned": len(windows),
        "n_vad_segments": len(vad_segs),
    }


def pass2_run(flagged_rows: list[dict], lid_model, hu_idx: int, vad_model,
              out_sidecar, progress_every: int = 200) -> dict:
    """Pass 2 on all flagged clips. Sequential per clip (VAD + windowed LID
    are both fast enough that batching across clips isn't critical)."""
    n_done = 0
    n_errors = 0
    t0 = time.time()

    for row in flagged_rows:
        audio = load_audio(row["audio_path"])
        if audio is None or len(audio) < int(0.5 * TARGET_SR):
            rec = {"utterance_id": row["utterance_id"],
                   "pass2_error": "audio_load_failed"}
            out_sidecar.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_errors += 1
            n_done += 1
            continue

        try:
            extra = pass2_for_flagged_clip(row, audio, lid_model, hu_idx, vad_model)
        except Exception as ex:
            rec = {"utterance_id": row["utterance_id"],
                   "pass2_error": str(ex)[:120]}
            out_sidecar.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_errors += 1
            n_done += 1
            continue

        rec = {"utterance_id": row["utterance_id"], **extra}
        out_sidecar.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_done += 1

        if n_done % progress_every == 0:
            out_sidecar.flush()
            rate = n_done / (time.time() - t0)
            eta = (len(flagged_rows) - n_done) / rate / 60 if rate > 0 else 0
            print(f"[pass2] {n_done:,}/{len(flagged_rows):,} "
                  f"({n_errors:,} errors, {rate:.1f} clips/s, "
                  f"ETA {eta:.1f} min)", file=sys.stderr)

    return {"n_done": n_done, "n_errors": n_errors}


# ============================================================
# Main
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=MANIFEST_DEFAULT,
                   help="Manifest JSONL to iterate (default: manifest_v5.jsonl)")
    p.add_argument("--source", type=str, default="voxpopuli_resegmented",
                   help="Source filter (default: voxpopuli_resegmented; "
                        "pass '' for all multi-source manifests like smoke/dev). "
                        "Note: voxpopuli_hu_labeled rows are skipped "
                        "automatically because their audio is parquet-internal "
                        "(not whole standalone files).")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--stage", choices=["1", "2", "all"], default="all",
                   help="Run only pass 1, only pass 2, or both (default).")
    p.add_argument("--hu-prob-threshold", type=float, default=HU_PROB_THRESHOLD,
                   help="hu_prob below this triggers pass 2 (default 0.85)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N rows this run (for validation).")
    p.add_argument("--batch-size", type=int, default=24,
                   help="Pass 1 GPU batch size (whole-clip LID inputs per batch).")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Sources whose audio is NOT a whole standalone file. The current
    # load_audio() reads the whole file; supporting these would need a
    # quality_tier2-style per-source loader. Skip with a count.
    # yodas2_hu000 was unsupported until 2026-05-26 morning when its
    # parent WAVs were sliced into standalone 16 kHz mono OGG chunks
    # (see bin/chunk_yodas2.py + yodas2_chunked.jsonl).
    # voxpopuli_hu_labeled was unsupported until 2026-05-26 evening when
    # its parquet-internal audio was extracted to standalone OGG (see
    # bin/extract_vp_labeled.py + voxpopuli_hu_labeled_extracted.jsonl).
    UNSUPPORTED_SOURCES: set[str] = set()

    # Read input manifest
    print(f"[init] reading {args.input.name}...", file=sys.stderr)
    work: list[dict] = []
    skipped_unsupported = 0
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if args.source and r.get("source") != args.source:
                continue
            if not r.get("audio_path"):
                continue
            if r.get("source") in UNSUPPORTED_SOURCES:
                skipped_unsupported += 1
                continue
            work.append({
                "utterance_id": r["utterance_id"],
                "audio_path": r["audio_path"],
            })
    if skipped_unsupported:
        print(f"[init] skipped {skipped_unsupported:,} rows from unsupported "
              f"sources ({sorted(UNSUPPORTED_SOURCES)} — parquet-internal "
              f"audio, not whole standalone files)", file=sys.stderr)
    print(f"[init] {len(work):,} candidates", file=sys.stderr)

    done_ids = load_done_ids(args.out)
    print(f"[init] {len(done_ids):,} clips already in sidecar", file=sys.stderr)
    pending = [r for r in work if r["utterance_id"] not in done_ids]
    if args.limit:
        pending = pending[:args.limit]
    print(f"[init] {len(pending):,} clips pending this run", file=sys.stderr)

    if not pending and args.stage in ("1", "all"):
        print("[done] nothing to do for pass 1", file=sys.stderr)
        if args.stage == "1":
            return 0

    # Load models
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] loading VoxLingua107 on {device}...", file=sys.stderr)
    from speechbrain.inference.classifiers import EncoderClassifier
    lid_model = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        run_opts={"device": device},
    )
    hu_idx = None
    for idx, lab in lid_model.hparams.label_encoder.ind2lab.items():
        if lab.split(":")[0].strip().lower() == "hu" or "hungarian" in lab.lower():
            hu_idx = idx
            break
    if hu_idx is None:
        print("[ERROR] HU class not found", file=sys.stderr)
        return 1
    print(f"[init] HU class index: {hu_idx}", file=sys.stderr)

    vad_model = None
    if args.stage in ("2", "all"):
        print(f"[init] loading Silero VAD...", file=sys.stderr)
        from silero_vad import load_silero_vad
        vad_model = load_silero_vad()

    # ========== Pass 1 ==========
    pass1_stats = {"n_done": 0, "n_errors": 0, "n_flagged": 0}
    if args.stage in ("1", "all") and pending:
        print(f"[pass1] starting whole-clip LID on {len(pending):,} clips",
              file=sys.stderr)
        t0 = time.time()
        with args.out.open("a", encoding="utf-8") as out:
            pass1_stats = pass1_whole_clip(
                pending, lid_model, hu_idx, out,
                batch_size=args.batch_size,
                hu_prob_threshold=args.hu_prob_threshold,
            )
        print(f"[pass1] done: {pass1_stats['n_done']:,} clips "
              f"({pass1_stats['n_flagged']:,} flagged, "
              f"{pass1_stats['n_errors']:,} errors, "
              f"{(time.time()-t0)/60:.1f} min)", file=sys.stderr)

    # ========== Pass 2 ==========
    if args.stage in ("2", "all"):
        # Collect flagged clips that DON'T already have language_regions
        print(f"[pass2] identifying flagged clips...", file=sys.stderr)
        flagged_ids: set = set()
        has_regions: set = set()
        with args.out.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                utt = r.get("utterance_id")
                if not utt:
                    continue
                if r.get("needs_pass2"):
                    flagged_ids.add(utt)
                if "language_regions" in r:
                    has_regions.add(utt)
        pending_pass2_ids = flagged_ids - has_regions
        # Map ids back to work rows
        utt_to_row = {r["utterance_id"]: r for r in work}
        flagged_rows = [utt_to_row[u] for u in pending_pass2_ids
                        if u in utt_to_row]
        if args.limit:
            flagged_rows = flagged_rows[:args.limit]
        print(f"[pass2] {len(flagged_rows):,} flagged clips to process",
              file=sys.stderr)

        if flagged_rows:
            t0 = time.time()
            with args.out.open("a", encoding="utf-8") as out:
                pass2_stats = pass2_run(
                    flagged_rows, lid_model, hu_idx, vad_model, out,
                )
            print(f"[pass2] done: {pass2_stats['n_done']:,} clips "
                  f"({pass2_stats['n_errors']:,} errors, "
                  f"{(time.time()-t0)/60:.1f} min)", file=sys.stderr)

    print()
    print(f"=== clip_language_purity_v2 summary ===")
    print(f"Stage: {args.stage}")
    print(f"Sidecar: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
