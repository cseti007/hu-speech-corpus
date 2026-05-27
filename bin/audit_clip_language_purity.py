#!/usr/bin/env python3
"""Phase 2.6b: detect foreign-language prefix in mosel_hu_voxpopuli clips.

Some mosel clips begin with 1-2 seconds of non-Hungarian speech (an MEP
quoting English / French / German / Latin at the opening of their HU
intervention) before switching to Hungarian. The two ASR pillars disagree
on what to do with the prefix.

Method:
  - For each clip, run VoxLingua107 LID on 1-second non-overlapping
    windows starting at t=0, up to 5 sec (or clip duration, whichever
    shorter) PLUS the whole clip
  - foreign_prefix_sec = end of the latest contiguous non-HU window
    BEFORE the first HU window in the first 5 sec
  - If whole-clip top1 != "hu", flag as `whole_non_hu` (not a prefix
    issue — the whole clip is foreign, separate problem)
  - If first window IS hu, foreign_prefix_sec = 0.0 (no prefix)

GPU-batched via SpeechBrain ECAPA-TDNN classifier (same as Tier-2 LID).
~75 clips/sec on a single GPU worker; 13k clips ≈ 3 min.

Output sidecar `processed/quality/clip_language_purity.jsonl`:
  {
    "utterance_id": "...",
    "whole_clip_top1": "hu",
    "first_window_top1": "en",  # 1-sec window starting at 0
    "foreign_prefix_sec": 2.0,  # 0.0 if no prefix
    "n_non_hu_windows": 2,
    "n_windows_scanned": 5,
    "whole_non_hu": false        # whole clip != hu
  }

Idempotent: rows already in the sidecar are skipped.

Scope:
  Default scope is `--sample-index notes/poc_100h/sample_index.jsonl`
  (the 13k mosel clips of the Phase 4a PoC). Pass `--all` for the
  full 2.31M mosel corpus.

Run (audio_ds env has speechbrain + GPU torch):
  /media/cseti/datassd/conda/miniconda3/envs/audio_ds/bin/python \
      bin/audit_clip_language_purity.py
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
MANIFEST_PATH = DATA_ROOT / "processed" / "manifests" / "manifest.jsonl"
SAMPLE_INDEX_DEFAULT = Path(
    "/home/cseti/data2/Develop/Github-cseti/cseti-os/projects/hu-speech-corpus/"
    "notes/poc_100h/sample_index.jsonl"
)
OUT_PATH = DATA_ROOT / "processed" / "quality" / "clip_language_purity.jsonl"

TARGET_SR = 16000
WINDOW_SEC = 1.0
MAX_PREFIX_SCAN_SEC = 5.0


def load_clip(audio_path: str) -> np.ndarray | None:
    try:
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        return None
    return audio


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
    p.add_argument("--sample-index", type=Path, default=SAMPLE_INDEX_DEFAULT)
    p.add_argument("--all", action="store_true",
                   help="Audit every mosel_hu_voxpopuli row (heavy).")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Number of CLIPS per GPU batch. Each clip contributes "
                        "up to 6 LID inferences (1 whole + 5 windows). "
                        "batch_size=4 -> ~24 LID inputs per GPU call, safe "
                        "on a 16 GB GPU. The whole-clip input pads up to 10 sec, "
                        "which dominates GPU memory; do not exceed ~32 LID "
                        "inputs/call (batch_size=5).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N clips this run (for validation).")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Scope
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
        print(f"[init] scope: {len(scope_ids):,} utterance_ids", file=sys.stderr)
    else:
        print("[init] scope: ALL mosel_hu_voxpopuli rows", file=sys.stderr)

    done_ids = load_done_ids(args.out)
    print(f"[init] {len(done_ids):,} clips already audited", file=sys.stderr)

    print("[init] building work list...", file=sys.stderr)
    work = []
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
            if not r.get("audio_path"):
                continue
            work.append({
                "utterance_id": utt,
                "audio_path": r["audio_path"],
                "duration_sec": r.get("duration_sec") or 0.0,
            })

    if args.limit:
        work = work[:args.limit]

    if not work:
        print("[done] nothing to do", file=sys.stderr)
        return 0

    print(f"[init] {len(work):,} clips to audit", file=sys.stderr)

    # Load VoxLingua107 once on GPU
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] loading VoxLingua107 on {device}...", file=sys.stderr)
    model = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        run_opts={"device": device},
    )

    # Find HU label index once
    hu_idx = None
    label_to_idx = model.hparams.label_encoder.ind2lab
    idx_to_label = {}
    for idx, lab in label_to_idx.items():
        idx_to_label[idx] = lab
        if lab.split(":")[0].strip().lower() == "hu" or "hungarian" in lab.lower():
            hu_idx = idx
    if hu_idx is None:
        print("[ERROR] couldn't find HU class in VoxLingua107", file=sys.stderr)
        return 1
    print(f"[init] HU class index: {hu_idx}", file=sys.stderr)

    win_samples = int(WINDOW_SEC * TARGET_SR)
    max_prefix_samples = int(MAX_PREFIX_SCAN_SEC * TARGET_SR)

    def classify_batch(input_tensors: list[np.ndarray]) -> list[tuple[str, float]]:
        """Pad to max length, classify, return [(top1_label, hu_prob), ...]."""
        max_len = max(len(a) for a in input_tensors)
        padded = np.zeros((len(input_tensors), max_len), dtype=np.float32)
        wav_lens_rel = np.zeros(len(input_tensors), dtype=np.float32)
        for i, a in enumerate(input_tensors):
            padded[i, :len(a)] = a
            wav_lens_rel[i] = len(a) / max_len
        signal = torch.from_numpy(padded)
        wav_lens = torch.from_numpy(wav_lens_rel)
        with torch.no_grad():
            prediction = model.classify_batch(signal, wav_lens)
        out_prob = prediction[0]
        text_labs = prediction[3]
        probs = torch.softmax(out_prob, dim=1)
        results = []
        for i, text_lab in enumerate(text_labs):
            top1 = text_lab.split(":")[0].strip() if ":" in text_lab else text_lab
            hu_p = float(probs[i, hu_idx])
            results.append((top1, hu_p))
        return results

    t0 = time.time()
    n_done = 0
    n_foreign_prefix = 0
    n_whole_non_hu = 0
    n_errors = 0
    progress_every = max(200, len(work) // 100)

    with args.out.open("a", encoding="utf-8") as out_f:
        # Process clips in batches; each clip contributes 1-6 model inputs
        # (whole + 5 windows). We flatten all inputs across a batch of B
        # clips into one classify_batch call.
        i = 0
        while i < len(work):
            batch_items = work[i:i + args.batch_size]
            i += args.batch_size

            # Build inputs + index mapping
            inputs: list[np.ndarray] = []
            owner: list[tuple[int, str]] = []  # (clip_idx, slot: "whole"|"w0"|"w1"|...)
            audios: dict[int, np.ndarray] = {}

            for clip_idx, item in enumerate(batch_items):
                audio = load_clip(item["audio_path"])
                if audio is None or len(audio) < int(0.5 * TARGET_SR):
                    continue
                audios[clip_idx] = audio
                # Whole-clip input (cap at 10 sec for speed)
                whole = audio[: 10 * TARGET_SR]
                inputs.append(whole)
                owner.append((clip_idx, "whole"))
                # Windowed inputs over first 5 sec
                for w in range(int(MAX_PREFIX_SCAN_SEC / WINDOW_SEC)):
                    s = w * win_samples
                    e = s + win_samples
                    if s >= len(audio):
                        break
                    chunk = audio[s:min(e, len(audio))]
                    if len(chunk) < int(0.5 * TARGET_SR):  # too short
                        break
                    inputs.append(chunk)
                    owner.append((clip_idx, f"w{w}"))

            if not inputs:
                continue

            try:
                results = classify_batch(inputs)
            except Exception as ex:
                # Emit errors for this batch
                for clip_idx, item in enumerate(batch_items):
                    rec = {"utterance_id": item["utterance_id"],
                           "lid_audit_error": str(ex)[:120]}
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_errors += 1
                    n_done += 1
                out_f.flush()
                continue

            # Re-group results per clip
            per_clip: dict[int, dict] = {}
            for (clip_idx, slot), (top1, hu_p) in zip(owner, results):
                per_clip.setdefault(clip_idx, {})[slot] = (top1, hu_p)

            for clip_idx, item in enumerate(batch_items):
                slots = per_clip.get(clip_idx)
                if slots is None:
                    rec = {"utterance_id": item["utterance_id"],
                           "lid_audit_error": "audio_load_failed"}
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_errors += 1
                    n_done += 1
                    continue

                whole_top1, whole_hu_p = slots.get("whole", ("?", 0.0))
                window_results = []
                for w in range(int(MAX_PREFIX_SCAN_SEC / WINDOW_SEC)):
                    key = f"w{w}"
                    if key in slots:
                        window_results.append(slots[key])

                first_window_top1 = window_results[0][0] if window_results else None
                whole_non_hu = (whole_top1 != "hu")

                # foreign_prefix_sec: latest end of contiguous non-hu window
                # before the first hu window (within the scanned prefix).
                # Computed regardless of whole_non_hu: a whole-non-HU clip can
                # still have a foreign PREFIX followed by HU speech that the
                # whole-clip LID is unsure about (see e.g. mosel hu_195 where
                # the 30s window mixes 5+s foreign with 25s HU and the
                # whole-clip LID lands on Slovak with 1% HU prob).
                foreign_prefix_sec = 0.0
                n_non_hu_windows = 0
                if window_results:
                    for w_idx, (lab, _) in enumerate(window_results):
                        if lab != "hu":
                            n_non_hu_windows += 1
                            foreign_prefix_sec = (w_idx + 1) * WINDOW_SEC
                        else:
                            break

                rec = {
                    "utterance_id": item["utterance_id"],
                    "whole_clip_top1": whole_top1,
                    "whole_clip_hu_prob": round(whole_hu_p, 4),
                    "first_window_top1": first_window_top1,
                    "foreign_prefix_sec": foreign_prefix_sec,
                    "n_non_hu_windows": n_non_hu_windows,
                    "n_windows_scanned": len(window_results),
                    "whole_non_hu": whole_non_hu,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if foreign_prefix_sec > 0:
                    n_foreign_prefix += 1
                if whole_non_hu:
                    n_whole_non_hu += 1
                n_done += 1

            if n_done % progress_every < args.batch_size:
                out_f.flush()
                rate = n_done / (time.time() - t0)
                eta = (len(work) - n_done) / rate / 60 if rate > 0 else 0
                print(f"[progress] {n_done:,}/{len(work):,} "
                      f"({n_foreign_prefix:,} foreign-prefix, "
                      f"{n_whole_non_hu:,} whole-non-hu, "
                      f"{n_errors:,} errors, {rate:.0f} clips/s, "
                      f"ETA {eta:.1f} min)", file=sys.stderr)

    print()
    print("=== Language purity audit summary ===")
    print(f"Processed:           {n_done:,}")
    print(f"Foreign prefix:      {n_foreign_prefix:,} "
          f"({n_foreign_prefix/n_done*100:.2f}%)")
    print(f"Whole-clip non-HU:   {n_whole_non_hu:,} "
          f"({n_whole_non_hu/n_done*100:.2f}%)")
    print(f"Errors:              {n_errors:,}")
    print(f"Time:                {(time.time()-t0)/60:.1f} min")
    print(f"Sidecar:             {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
