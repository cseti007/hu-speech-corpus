#!/usr/bin/env python3
"""Sanity-check the official VoxPopuli unlabelled segmentation manifest for HU.

Downloads `unlabelled_v2.tsv.gz` (the per-utterance offset+duration table) from
the official FB CDN, filters to Hungarian rows, and prints:
  - utterance count
  - total covered hours
  - duration distribution (p5, median, p95, max)
  - per-year breakdown
  - output disk estimate (raw ogg + Opus 32 kbps re-encode)
  - sample IDs to verify the format matches MOSEL utterance IDs

Read-only. Saves the TSV to raw/voxpopuli_hu_unlabeled/annotations/ for reuse
by the actual segmentation step.
"""

import csv
import gzip
import statistics
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path("/home/cseti/datassd2/hu-speech-corpus/raw/voxpopuli_hu_unlabeled")
ANNOT_DIR = ROOT / "annotations"
TSV_URL = "https://dl.fbaipublicfiles.com/voxpopuli/annotations/unlabelled_v2.tsv.gz"
TSV_PATH = ANNOT_DIR / "unlabelled_v2.tsv.gz"


def download_if_missing():
    if TSV_PATH.exists():
        print(f"[skip] {TSV_PATH} already exists ({TSV_PATH.stat().st_size / 1e6:.1f} MB)",
              file=sys.stderr)
        return
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] {TSV_URL} -> {TSV_PATH}", file=sys.stderr)
    urllib.request.urlretrieve(TSV_URL, TSV_PATH)
    print(f"[done] {TSV_PATH.stat().st_size / 1e6:.1f} MB", file=sys.stderr)


def main():
    download_if_missing()

    durations = []
    by_year = Counter()
    by_session = Counter()
    sample_ids = []

    with gzip.open(TSV_PATH, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            event_id = row["event_id"]
            # event_id format: "20160118-0900-PLENARY-10_hu"
            if not event_id.endswith("_hu"):
                continue
            seg_no = row["segment_no"]
            start = float(row["start"])
            end = float(row["end"])
            dur = end - start
            durations.append(dur)
            year = event_id[:4]
            by_year[year] += 1
            by_session[event_id] += 1
            if len(sample_ids) < 8:
                sample_ids.append(f"{event_id}_{seg_no} ({start:.2f}-{end:.2f}s, {dur:.2f}s)")

    n = len(durations)
    total_sec = sum(durations)

    print()
    print("=" * 70)
    print("VoxPopuli HU unlabelled segmentation manifest stats")
    print("=" * 70)
    print(f"Utterance count:        {n:,}")
    print(f"Unique sessions:        {len(by_session):,}")
    print(f"Avg utt/session:        {n / max(len(by_session), 1):.1f}")
    print(f"Total covered duration: {total_sec / 3600:.2f}h "
          f"({total_sec / 60:.0f} min)")
    print()
    print("Per-utterance duration distribution (seconds):")
    durations.sort()
    p = lambda q: durations[int(q * len(durations))] if durations else 0
    print(f"  min:    {min(durations):.2f}")
    print(f"  p5:     {p(0.05):.2f}")
    print(f"  median: {statistics.median(durations):.2f}")
    print(f"  p95:    {p(0.95):.2f}")
    print(f"  max:    {max(durations):.2f}")
    print(f"  mean:   {statistics.mean(durations):.2f}")
    print()
    print("Per-year breakdown:")
    for year in sorted(by_year):
        print(f"  {year}: {by_year[year]:>8,} utterances")
    print()
    print("Sample utterance IDs (to verify format matches MOSEL):")
    for s in sample_ids:
        print(f"  {s}")
    print()

    # --- Output size estimates
    # Raw ogg/vorbis at source bitrate (~16 kbps avg for VoxPopuli unlabeled):
    #   bytes_per_sec ~= 2,000 (16 kbps / 8). But torchaudio re-encodes with
    #   default settings which may differ. Conservative estimate: 8 kB/sec audio.
    # Opus 32 kbps: 4 kB/sec audio.
    raw_estimate_gb = (total_sec * 8_000) / 1e9
    opus_estimate_gb = (total_sec * 4_000) / 1e9

    print("Output disk usage estimates:")
    print(f"  Raw ogg (torchaudio default, ~16 kbps): ~{raw_estimate_gb:.0f} GB "
          f"(±50%; actual depends on torchaudio encoder defaults)")
    print(f"  Opus 32 kbps re-encode:                 ~{opus_estimate_gb:.0f} GB")
    print()
    print(f"Free disk on datassd2: see `df -h /home/cseti/datassd2/`")


if __name__ == "__main__":
    main()
