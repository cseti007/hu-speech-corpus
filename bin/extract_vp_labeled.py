#!/usr/bin/env python3
"""Extract parquet-internal vp_labeled audio to standalone 16 kHz mono OGG.

vp_labeled audio lives inside HuggingFace parquet shards (~600 MB each)
under `raw/voxpopuli_hu_labeled/hu/`. The `audio` column is a struct with
WAV bytes + path. The audio is 16 kHz mono already.

This script reads each manifest_v5 vp_labeled row, loads the parent parquet
shard ONCE per shard (4 train + 1 dev + 1 test = 6 shards), and writes one
standalone OGG/Vorbis file per clip to
`processed/chunks/voxpopuli_hu_labeled/<utterance_id>.ogg`. Output sidecar
`yodas2_chunked.jsonl`-style at
`processed/normalization/voxpopuli_hu_labeled_extracted.jsonl`.

Why: production quality scripts (`quality_tier2.py` VAD/DNSMOS,
`audit_clip_language_purity_v2.py`) skip vp_labeled because their worker
audio loaders can't decode parquet-internal blobs. With standalone OGG
files those scripts work without changes. Training pipelines (NeMo, etc.)
also want one file per sample.

Idempotent: skips clips whose OGG already exists.

Run with the base env (pandas + pyarrow + scipy + soundfile):
  /media/cseti/datassd/conda/miniconda3/bin/python bin/extract_vp_labeled.py
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
DEFAULT_INPUT_MANIFEST = (
    DATA_ROOT / "processed" / "manifests" / "manifest_v5.jsonl"
)
DEFAULT_CHUNKS_DIR = DATA_ROOT / "processed" / "chunks" / "voxpopuli_hu_labeled"
DEFAULT_SIDECAR = (
    DATA_ROOT / "processed" / "normalization" / "voxpopuli_hu_labeled_extracted.jsonl"
)
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


def process_shard(shard_path: Path, entries: list[dict], chunks_dir: Path) -> tuple[int, int, int]:
    """Worker: load one parquet shard, extract + encode all its rows.

    `entries`: list of {utterance_id, parquet_row_index}.
    Returns (n_written, n_skipped, n_errors)."""
    pending = []
    for e in entries:
        out_path = chunks_dir / f"{e['utterance_id'].replace('/', '_')}.ogg"
        if out_path.is_file():
            continue
        pending.append((e, out_path))
    if not pending:
        return 0, len(entries), 0

    import pyarrow.parquet as pq
    try:
        table = pq.read_table(str(shard_path), columns=["audio"])
        audio_col = table.column("audio").to_pylist()
    except Exception as ex:
        print(f"[shard-load-error] {shard_path.name}: {ex}", file=sys.stderr)
        return 0, 0, len(entries)

    n_written = 0
    n_errors = 0
    for e, out_path in pending:
        row_idx = int(e["parquet_row_index"])
        if row_idx < 0 or row_idx >= len(audio_col):
            n_errors += 1
            continue
        rec = audio_col[row_idx]
        if rec is None or "bytes" not in rec:
            n_errors += 1
            continue
        try:
            audio, sr = sf.read(io.BytesIO(rec["bytes"]), dtype="float32",
                                always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                audio = _resample(audio, sr, TARGET_SR)
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            sf.write(str(tmp), audio, TARGET_SR, format="OGG", subtype="VORBIS")
            os.replace(tmp, out_path)
            n_written += 1
        except Exception as ex:
            n_errors += 1
            print(f"[encode-error] {e['utterance_id']}: {ex}", file=sys.stderr)
    return n_written, len(entries) - len(pending), n_errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_MANIFEST,
                   help="manifest_v5.jsonl to read vp_labeled rows from.")
    p.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR,
                   help="Output directory for .ogg chunks.")
    p.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR,
                   help="Sidecar JSONL listing {utterance_id, audio_path, ...}.")
    args = p.parse_args()

    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)

    print(f"[extract] reading {args.input}", file=sys.stderr)
    by_shard: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    n_missing_idx = 0
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("source") != "voxpopuli_hu_labeled":
                continue
            n_total += 1
            shard = row.get("audio_path")
            idx = row.get("parquet_row_index")
            if not shard or idx is None:
                n_missing_idx += 1
                continue
            by_shard[shard].append({
                "utterance_id": row["utterance_id"],
                "parquet_row_index": idx,
                "duration_sec": row.get("duration_sec"),
                "transcripts": row.get("transcripts") or {},
            })
    print(f"[extract] {n_total:,} vp_labeled rows across {len(by_shard)} shards",
          file=sys.stderr)
    if n_missing_idx:
        print(f"[extract] {n_missing_idx:,} rows missing parquet_row_index "
              f"(skipping)", file=sys.stderr)

    t0 = time.time()
    grand_w = 0; grand_s = 0; grand_e = 0
    for shard_str, entries in by_shard.items():
        shard = Path(shard_str)
        print(f"[extract] {shard.name} ({len(entries):,} rows)...",
              file=sys.stderr, flush=True)
        w, s, e = process_shard(shard, entries, args.chunks_dir)
        grand_w += w
        grand_s += s
        grand_e += e
        print(f"  → {w:,} written, {s:,} skipped, {e:,} errors",
              file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\n[extract] done in {elapsed/60:.1f} min", file=sys.stderr)
    print(f"          {grand_w:,} chunks written", file=sys.stderr)
    print(f"          {grand_s:,} chunks already existed (skipped)",
          file=sys.stderr)
    print(f"          {grand_e:,} errors", file=sys.stderr)

    # Build output sidecar: one row per vp_labeled clip with new audio_path.
    print(f"\n[extract] writing sidecar {args.sidecar}", file=sys.stderr)
    tmp_side = args.sidecar.with_suffix(args.sidecar.suffix + ".tmp")
    n_side = 0
    with tmp_side.open("w", encoding="utf-8") as out_f:
        for shard_str, entries in by_shard.items():
            for e in entries:
                ogg_path = args.chunks_dir / f"{e['utterance_id'].replace('/', '_')}.ogg"
                if not ogg_path.is_file():
                    continue
                rec = {
                    "utterance_id": e["utterance_id"],
                    "audio_path": str(ogg_path),
                    "duration_sec": e["duration_sec"],
                    "sample_rate": TARGET_SR,
                    "channels": 1,
                    "codec": "ogg",
                    "parquet_shard": shard_str,
                    "parquet_row_index": e["parquet_row_index"],
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_side += 1
    os.replace(tmp_side, args.sidecar)
    print(f"[extract] {n_side:,} rows in sidecar", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
