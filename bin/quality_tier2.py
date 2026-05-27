#!/usr/bin/env python3
"""Phase 3 Tier-2: neural audio-quality scoring per clip.

Currently implemented metrics:
  - vad   Silero VAD → vad_speech_ratio, vad_num_segments (CPU; ~20 min for 3.2M clips)
  - lid   VoxLingua107 → lid_top1, lid_top1_prob, lid_is_hu_prob (GPU; ~2-4 h)

Future additions (heavier deps, defer):
  - dnsmos  Microsoft DNSMOS P.835 SIG/BAK/OVRL
  - utmosv2 sarulab UTMOSv2 (TTS naturalness MOS)
  - music   YAMNet / PANNs (music vs speech)

Per-metric design: each metric has its own sidecar in processed/quality/.
Re-running picks up from where it stopped: rows whose utterance_id already
appears in the sidecar are skipped. Append-mode JSONL writes are flushed
per-line, so Ctrl+C / kill only loses the in-flight row.

Audio loading mirrors quality_tier1.py:
  - Standalone ogg (MOSEL pseudo, untranscribed_chunks): decode directly
  - YODAS2 merged: load parent WAV once, slice each clip (group by parent)
  - VoxPopuli labeled: parquet-internal audio bytes via soundfile

Run examples:
  # VAD only:
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python \
      bin/quality_tier2.py --metric vad --n_workers 12
  # LID only:
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python \
      bin/quality_tier2.py --metric lid --n_workers 4
"""

from __future__ import annotations

import argparse
import io
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
MANIFESTS_DIR = DATA_ROOT / "processed" / "manifests"
OUT_DIR = DATA_ROOT / "processed" / "quality"

TARGET_SR = 16000

# Per-worker state (loaded once per worker via initializer)
_model = None
_metric_kind = None


# ============================================================
# Audio loading (shared with tier1)
# ============================================================

def _load_audio_for_row(row: dict, parent_wav_cache: dict) -> np.ndarray | None:
    """Return mono float32 PCM @ 16 kHz for a manifest row.

    parent_wav_cache: a dict shared in the calling worker to cache YODAS2 parent WAVs.
    """
    source = row.get("source", "")
    audio_path = row.get("audio_path")

    # yodas2 post-chunking (2026-05-26+): audio_path points to a standalone
    # 16 kHz mono OGG chunk; segment offsets are null. Detect and treat as
    # standalone (the yodas2_hu000 branch below still handles legacy v4 rows
    # with parent WAV + segment offsets).
    #
    # vp_labeled post-extraction (2026-05-26 evening+): audio_path no longer
    # ends in .parquet; it points at the extracted OGG. Treat as standalone.
    if source == "yodas2_hu000" and row.get("segment_start_sec") is None:
        source_for_loading = "voxpopuli_resegmented"  # any standalone bucket
    elif (source == "voxpopuli_hu_labeled"
          and audio_path and not audio_path.endswith(".parquet")):
        source_for_loading = "voxpopuli_resegmented"
    else:
        source_for_loading = source

    # Standalone ogg / wav / mp3: read directly.
    if source_for_loading in {"mosel_hu_voxpopuli", "librivox_hu", "podcasts_hu_cc",
                  "voxpopuli_unlabeled_gap", "voxpopuli_resegmented",
                  "common_voice_25_0_hu"}:
        if not audio_path:
            return None
        try:
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        except Exception:
            return None
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            # CV25 is 48 kHz MP3; others are 16 kHz by construction. Resample
            # via scipy.signal.resample_poly (matches the yodas2 branch below).
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

    # YODAS2 virtual segment: parent WAV (24 kHz) + slice + downsample to 16 kHz.
    if source == "yodas2_hu000":
        if audio_path not in parent_wav_cache:
            try:
                wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                parent_wav_cache[audio_path] = (wav, sr)
            except Exception:
                parent_wav_cache[audio_path] = (None, 0)
        wav, sr = parent_wav_cache[audio_path]
        if wav is None:
            return None
        s = int(row["segment_start_sec"] * sr)
        e = int(row["segment_end_sec"] * sr)
        e = min(e, len(wav))
        if e <= s:
            return None
        clip = wav[s:e]
        if sr != TARGET_SR:
            # Simple resample via scipy.signal.resample_poly would be ideal;
            # for VAD we can downsample with stride, which is approximate but OK.
            from math import gcd
            g = gcd(sr, TARGET_SR)
            up = TARGET_SR // g
            down = sr // g
            try:
                from scipy.signal import resample_poly
                clip = resample_poly(clip, up, down).astype(np.float32)
            except ImportError:
                # Fallback: nearest-stride decimation (lossy but workable for VAD/LID)
                clip = clip[::max(1, sr // TARGET_SR)].astype(np.float32)
        return clip

    # VoxPopuli labeled: parquet-internal audio, decoded by main process
    # (we don't reload parquet per-row in workers — too slow).
    # For Tier-2, voxpopuli_labeled is handled in a separate sequential pass.
    return None


# ============================================================
# Metric: Silero VAD
# ============================================================

def _init_worker_vad():
    """Pool worker init: load Silero VAD. Pin torch to 1 BLAS thread per worker
    so 12 workers × 1 thread = 12 threads, not 12 × N_cores."""
    global _model, _metric_kind
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad
    _model = load_silero_vad()
    _metric_kind = "vad"


def _worker_vad(rows_batch):
    """Process a batch of rows for VAD. Returns list of result dicts."""
    import torch
    from silero_vad import get_speech_timestamps
    parent_cache: dict = {}
    out = []
    for row in rows_batch:
        audio = _load_audio_for_row(row, parent_cache)
        if audio is None or len(audio) < int(0.1 * TARGET_SR):
            continue
        try:
            t = torch.from_numpy(audio).float()
            segs = get_speech_timestamps(
                t, _model, sampling_rate=TARGET_SR, return_seconds=True,
                min_speech_duration_ms=100,
                min_silence_duration_ms=300,
            )
            speech_dur = sum(s["end"] - s["start"] for s in segs)
            clip_dur = len(audio) / TARGET_SR
            ratio = speech_dur / clip_dur if clip_dur > 0 else 0.0
            out.append({
                "utterance_id": row["utterance_id"],
                "vad_speech_ratio": round(float(ratio), 4),
                "vad_num_segments": len(segs),
                "vad_speech_sec": round(float(speech_dur), 3),
            })
        except Exception as ex:
            out.append({
                "utterance_id": row["utterance_id"],
                "vad_error": str(ex)[:120],
            })
    return out


# ============================================================
# Metric: VoxLingua107 LID
# ============================================================

def _init_worker_lid():
    """Pool worker init: load VoxLingua107 ECAPA-TDNN."""
    global _model, _metric_kind
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        run_opts={"device": device},
    )
    _metric_kind = "lid"


def _worker_lid(rows_batch):
    """Process a batch of rows for LID with proper GPU batching.

    Audio tensors are padded to the max length in the batch and passed
    through speechbrain's classify_batch in a single GPU forward pass.
    This is ~10-20x faster than per-clip inference on GPU.
    """
    import numpy as np
    import torch

    parent_cache: dict = {}
    max_samples = 10 * TARGET_SR  # cap clips at 10s for LID

    # Find HU index once per worker (cached in worker state)
    global _hu_idx
    try:
        _hu_idx
    except NameError:
        _hu_idx = None
        label_to_idx = _model.hparams.label_encoder.ind2lab
        for idx, lab in label_to_idx.items():
            if lab.split(":")[0].strip().lower() == "hu" or "hungarian" in lab.lower():
                _hu_idx = idx
                break

    audios = []
    valid_rows = []
    out = []
    for row in rows_batch:
        audio = _load_audio_for_row(row, parent_cache)
        if audio is None or len(audio) < int(0.5 * TARGET_SR):
            out.append({"utterance_id": row["utterance_id"],
                        "lid_error": "audio_too_short"})
            continue
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        audios.append(audio)
        valid_rows.append(row)

    if not audios:
        return out

    try:
        # Pad to max length within batch
        max_len = max(len(a) for a in audios)
        padded = np.zeros((len(audios), max_len), dtype=np.float32)
        wav_lens_rel = np.zeros(len(audios), dtype=np.float32)
        for i, a in enumerate(audios):
            padded[i, :len(a)] = a
            wav_lens_rel[i] = len(a) / max_len

        signal = torch.from_numpy(padded)
        wav_lens = torch.from_numpy(wav_lens_rel)
        # classify_batch returns (out_prob, score, idx, text_lab)
        # out_prob: (B, n_lang), score: (B,), idx: (B,), text_lab: list[str]
        prediction = _model.classify_batch(signal, wav_lens)
        out_prob = prediction[0]
        scores = prediction[1]
        text_labs = prediction[3]

        # Softmax per row
        probs = torch.softmax(out_prob, dim=1)

        for i, row in enumerate(valid_rows):
            text_lab = text_labs[i]
            score = float(scores[i])
            hu_prob = float(probs[i, _hu_idx]) if _hu_idx is not None else None
            out.append({
                "utterance_id": row["utterance_id"],
                "lid_top1": text_lab.split(":")[0].strip() if ":" in text_lab else text_lab,
                "lid_top1_label": text_lab,
                "lid_top1_score": round(score, 4),
                "lid_is_hu_prob": round(hu_prob, 4) if hu_prob is not None else None,
            })
    except Exception as ex:
        # On batch-wide failure, emit one error row per valid input
        for row in valid_rows:
            out.append({
                "utterance_id": row["utterance_id"],
                "lid_error": str(ex)[:120],
            })

    return out


# ============================================================
# Metric: DNSMOS P.835 (Microsoft, ONNX)
# ============================================================

DNSMOS_MODEL_PATH = "/home/cseti/data2/AI/models/dnsmos/sig_bak_ovr.onnx"
DNSMOS_INPUT_LEN = 144160  # 9.01 sec at 16 kHz


def _init_worker_dnsmos():
    """Pool worker init: load DNSMOS P.835 ONNX model on CPU."""
    global _model, _metric_kind
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    _model = ort.InferenceSession(
        DNSMOS_MODEL_PATH,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    _metric_kind = "dnsmos"


def _worker_dnsmos(rows_batch):
    """DNSMOS inference: pad/crop to 9.01s window, batch through ONNX,
    returns SIG/BAK/OVRL MOS scores (1-5)."""
    import numpy as np

    parent_cache: dict = {}
    audios = []
    valid_rows = []
    out = []
    for row in rows_batch:
        audio = _load_audio_for_row(row, parent_cache)
        if audio is None or len(audio) < int(0.5 * TARGET_SR):
            out.append({"utterance_id": row["utterance_id"],
                        "dnsmos_error": "audio_too_short"})
            continue
        # Pad to 9.01s, or center-crop if longer
        if len(audio) < DNSMOS_INPUT_LEN:
            pad = np.zeros(DNSMOS_INPUT_LEN - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])
        else:
            start = (len(audio) - DNSMOS_INPUT_LEN) // 2
            audio = audio[start:start + DNSMOS_INPUT_LEN]
        audios.append(audio)
        valid_rows.append(row)

    if not audios:
        return out

    try:
        batch = np.stack(audios).astype(np.float32)
        result = _model.run(None, {"input_1": batch})[0]  # shape (N, 3)
        for i, row in enumerate(valid_rows):
            sig, bak, ovrl = float(result[i, 0]), float(result[i, 1]), float(result[i, 2])
            out.append({
                "utterance_id": row["utterance_id"],
                "dnsmos_sig": round(sig, 3),
                "dnsmos_bak": round(bak, 3),
                "dnsmos_ovrl": round(ovrl, 3),
            })
    except Exception as ex:
        for row in valid_rows:
            out.append({"utterance_id": row["utterance_id"],
                        "dnsmos_error": str(ex)[:120]})

    return out


# ============================================================
# Main: iterate manifests, build work list, dispatch to pool
# ============================================================

METRICS = {
    "vad": {
        "sidecar": "tier2_vad.jsonl",
        "init": _init_worker_vad,
        "worker": _worker_vad,
        "default_workers": 12,
    },
    "lid": {
        "sidecar": "tier2_lid.jsonl",
        "init": _init_worker_lid,
        "worker": _worker_lid,
        # LID uses GPU; one worker is fine, batching done internally per-batch
        "default_workers": 1,
    },
    "dnsmos": {
        "sidecar": "tier2_dnsmos.jsonl",
        "init": _init_worker_dnsmos,
        "worker": _worker_dnsmos,
        # ONNX CPU; 8 workers fit comfortably (model is 1 MB, low RAM)
        "default_workers": 8,
    },
}


def load_done_ids(sidecar_path: Path) -> set:
    if not sidecar_path.exists():
        return set()
    ids = set()
    with sidecar_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["utterance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


# Configurable manifest input (overridden by --input via main()).
_MANIFEST_INPUT: Path | None = None


def iter_all_rows():
    """Yield every clip-level row from the configured manifest JSONL."""
    path = _MANIFEST_INPUT or (MANIFESTS_DIR / "manifest.jsonl")
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def iter_work(done_ids: set, source_filter: set | None = None):
    """Generator that yields slim row dicts (only fields the workers need).

    `source_filter`: if provided, only yield rows whose `source` is in the set.

    Streaming avoids holding the entire 3M-row list in main-process RAM
    (which previously caused 26GB peak + COW pressure across worker forks).
    """
    for row in iter_all_rows():
        if row["utterance_id"] in done_ids:
            continue
        # Legacy v4 manifests had vp_labeled audio_path pointing at a
        # parquet shard (not decodable per-worker). Post-2026-05-26 the
        # extract_vp_labeled.py pipeline writes standalone OGG and updates
        # audio_path accordingly — those rows we DO process.
        if (row["source"] == "voxpopuli_hu_labeled"
                and (row.get("audio_path") or "").endswith(".parquet")):
            continue
        if source_filter is not None and row["source"] not in source_filter:
            continue
        if row.get("audio_path") is None:
            continue
        yield {
            "utterance_id": row["utterance_id"],
            "source": row["source"],
            "audio_path": row["audio_path"],
            "segment_start_sec": row.get("segment_start_sec"),
            "segment_end_sec": row.get("segment_end_sec"),
        }


def batched(iterator, batch_size):
    """Yield successive batches of `batch_size` items from `iterator`."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_work_rough(done_ids: set, source_filter: set | None = None) -> int:
    """Fast row count for progress display (no slim-dict overhead)."""
    n = 0
    for row in iter_all_rows():
        if row["utterance_id"] in done_ids:
            continue
        # Legacy v4 manifests had vp_labeled audio_path pointing at a
        # parquet shard (not decodable per-worker). Post-2026-05-26 the
        # extract_vp_labeled.py pipeline writes standalone OGG and updates
        # audio_path accordingly — those rows we DO process.
        if (row["source"] == "voxpopuli_hu_labeled"
                and (row.get("audio_path") or "").endswith(".parquet")):
            continue
        if source_filter is not None and row["source"] not in source_filter:
            continue
        if row.get("audio_path") is None:
            continue
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=MANIFESTS_DIR / "manifest.jsonl",
                        help="Manifest JSONL to iterate (default: manifest.jsonl).")
    parser.add_argument("--metric", required=True, choices=list(METRICS.keys()))
    parser.add_argument("--output", type=Path, default=None,
                        help="Sidecar JSONL output path. If unset, defaults to "
                             "processed/quality/tier2_<metric>.jsonl. Pass an "
                             "alternate path when scoring smoke/dev sets.")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Worker process count. Default depends on metric.")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Rows per pool task (default 20).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N rows (testing).")
    parser.add_argument("--source-filter", type=str, default=None,
                        help="Comma-separated source keys to process "
                             "(e.g. 'yodas2_hu000,librivox_hu,podcasts_hu_cc'). "
                             "Default: all (except voxpopuli_hu_labeled).")
    args = parser.parse_args()

    # Wire the --input flag into the module-level iter_all_rows() helper.
    global _MANIFEST_INPUT
    _MANIFEST_INPUT = args.input

    cfg = METRICS[args.metric]
    sidecar: Path = args.output if args.output else (OUT_DIR / cfg["sidecar"])
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    n_workers = args.n_workers if args.n_workers is not None else cfg["default_workers"]

    print(f"[init] metric={args.metric}, sidecar={sidecar}", file=sys.stderr)
    done_ids = load_done_ids(sidecar)
    print(f"[init] {len(done_ids):,} rows already done", file=sys.stderr)

    source_filter = None
    if args.source_filter:
        source_filter = set(s.strip() for s in args.source_filter.split(",") if s.strip())
        print(f"[init] source_filter: {source_filter}", file=sys.stderr)

    print(f"[init] counting work (fast scan)...", file=sys.stderr)
    total_work = count_work_rough(done_ids, source_filter)
    if args.limit:
        total_work = min(total_work, args.limit)
    print(f"[init] {total_work:,} rows to process with {n_workers} workers",
          file=sys.stderr)
    if total_work == 0:
        print("[done] nothing to do", file=sys.stderr)
        return 0

    # Streaming generator → batches → pool (no materialized list)
    def row_stream():
        for i, row in enumerate(iter_work(done_ids, source_filter)):
            if args.limit is not None and i >= args.limit:
                return
            yield row

    t0 = time.time()
    n_done = 0
    n_errors = 0
    progress_every = max(2000, total_work // 200)
    progress_anchor = t0
    progress_done = 0

    with sidecar.open("a", encoding="utf-8") as out:
        with Pool(processes=n_workers, initializer=cfg["init"]) as pool:
            for results in pool.imap_unordered(
                cfg["worker"],
                batched(row_stream(), args.batch_size),
                chunksize=1,
            ):
                for rec in results:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_done += 1
                    if any(k.endswith("_error") for k in rec):
                        n_errors += 1
                out.flush()
                if n_done - progress_done >= progress_every:
                    now = time.time()
                    rate = (n_done - progress_done) / (now - progress_anchor)
                    remaining = total_work - n_done
                    eta_min = remaining / rate / 60 if rate > 0 else 0
                    print(f"[progress] {n_done:,}/{total_work:,} "
                          f"({rate:.0f} clips/s, ETA {eta_min:.1f} min, "
                          f"{n_errors} errors)", file=sys.stderr)
                    progress_anchor = now
                    progress_done = n_done

    print()
    print(f"=== Tier-2 {args.metric} summary ===")
    print(f"New rows added:   {n_done:,}")
    print(f"Errors:           {n_errors:,}")
    print(f"Time:             {(time.time()-t0)/60:.1f} min")
    print(f"Output: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
