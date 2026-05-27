#!/usr/bin/env python3
"""Extract per-row audio duration from voxpopuli_hu_labeled parquet files.

The parquet stores audio as embedded WAV bytes inside an `audio: struct<bytes,
path>` column. The current manifest builder doesn't decode the audio, so
duration_sec is null for all 20,306 voxpopuli_labeled rows. This script
decodes each row's audio (in-memory via soundfile) and writes a sidecar
mapping audio_id -> duration_sec.

Output: processed/normalization/voxpopuli_labeled_durations.jsonl, one row per
audio_id:
  {"audio_id": "...", "duration_sec": 12.34, "sample_rate": 16000, "channels": 1}

Also produces a summary of the duration distribution and flags any rows
outside the 3-30s range (which we expect to be near zero given VoxPopuli's
official VAD segmentation).

Run with the base conda env:
  /media/cseti/datassd/conda/miniconda3/bin/python bin/extract_voxpopuli_labeled_durations.py
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

VP_LABELED_DIR = Path("/home/cseti/datassd2/hu-speech-corpus/raw/voxpopuli_hu_labeled/hu")
OUT_DIR = Path("/home/cseti/datassd2/hu-speech-corpus/processed/normalization")
OUT_PATH = OUT_DIR / "voxpopuli_labeled_durations.jsonl"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquets = sorted(VP_LABELED_DIR.glob("*.parquet"))
    if not parquets:
        print(f"[error] no parquet files in {VP_LABELED_DIR}", file=sys.stderr)
        return 1

    n_total = 0
    total_dur = 0.0
    buckets = Counter()
    n_under_3s = 0
    n_over_30s = 0
    durs = []

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for p in parquets:
            print(f"[read] {p.name}", file=sys.stderr)
            tbl = pq.read_table(p, columns=["audio_id", "audio"])
            rows = tbl.to_pylist()
            for r in rows:
                aid = r["audio_id"]
                audio_bytes = r["audio"]["bytes"]
                try:
                    with io.BytesIO(audio_bytes) as buf:
                        with sf.SoundFile(buf) as f:
                            sr = f.samplerate
                            ch = f.channels
                            frames = f.frames
                            duration_sec = frames / sr
                except Exception as ex:
                    print(f"[error] decode failed for {aid}: {ex}", file=sys.stderr)
                    continue
                row = {
                    "audio_id": aid,
                    "duration_sec": round(duration_sec, 3),
                    "sample_rate": sr,
                    "channels": ch,
                }
                out.write(json.dumps(row) + "\n")
                n_total += 1
                total_dur += duration_sec
                durs.append(duration_sec)

                if duration_sec < 3.0:
                    n_under_3s += 1
                if duration_sec > 30.0:
                    n_over_30s += 1
                if duration_sec < 1: b = "<1s"
                elif duration_sec < 3: b = "1-3s"
                elif duration_sec < 5: b = "3-5s"
                elif duration_sec < 10: b = "5-10s"
                elif duration_sec < 20: b = "10-20s"
                elif duration_sec < 30: b = "20-30s"
                else: b = ">30s"
                buckets[b] += 1
            print(f"  -> {len(rows):,} rows", file=sys.stderr)

    durs.sort()
    n = len(durs)
    print()
    print("=== voxpopuli_hu_labeled duration summary ===")
    print(f"Total rows:    {n_total:,}")
    print(f"Total hours:   {total_dur / 3600:.2f}h")
    print(f"Duration:      min {durs[0]:.2f}s, p25 {durs[n//4]:.2f}s, "
          f"median {durs[n//2]:.2f}s, p75 {durs[3*n//4]:.2f}s, max {durs[-1]:.2f}s")
    print(f"Below 3s:      {n_under_3s:,} ({n_under_3s/n*100:.2f}%)")
    print(f"Above 30s:     {n_over_30s:,} ({n_over_30s/n*100:.2f}%)")
    print()
    print("Bucket distribution:")
    for b in ["<1s", "1-3s", "3-5s", "5-10s", "10-20s", "20-30s", ">30s"]:
        c = buckets[b]
        if c:
            print(f"  {b:>8}: {c:>6,} ({c/n*100:.1f}%)")
    print()
    print(f"Output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
