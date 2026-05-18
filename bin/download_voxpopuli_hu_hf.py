#!/usr/bin/env python3
"""Download VoxPopuli HU labeled audio (transcribed_data partition) from HF.

Source: facebook/voxpopuli, hu/ subdirectory
  - hu/train-00000-of-00004.parquet ... train-00003-of-00004.parquet
  - hu/test-00000-of-00001.parquet
  - hu/validation-00000-of-00001.parquet
Target: $HU_CORPUS_ROOT/raw/voxpopuli_hu_labeled/
Size: ~10.78 GB total, 6 parquet files

This is the transcribed (labeled) portion only — ~63 hours per the paper.
The much larger unlabeled portion (~17,700 h) is fetched separately by
`download_voxpopuli_hu_unlabeled.py` via the facebookresearch/voxpopuli github
download path (not on HF).

Idempotent: snapshot_download skips already-fetched parquet shards.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "facebook/voxpopuli"
ALLOW_PATTERNS = ["hu/*"]
EXPECTED_FILES = 6
EXPECTED_TOTAL_BYTES = 11_577_073_664  # ~10.78 GB, approximate

DEFAULT_TOKEN_PATH = Path("/home/cseti/.hf_token")
DEFAULT_DEST = Path("/home/cseti/datassd2/hu-speech-corpus/raw/voxpopuli_hu_labeled")
DEFAULT_CACHE = Path("/home/cseti/datassd2/hu-speech-corpus/cache")


def load_token() -> str:
    env = os.environ.get("HF_TOKEN")
    if env:
        return env.strip()
    if DEFAULT_TOKEN_PATH.is_file():
        return DEFAULT_TOKEN_PATH.read_text().strip()
    raise SystemExit(f"No HF token: set $HF_TOKEN or place at {DEFAULT_TOKEN_PATH}")


def verify(dest: Path) -> tuple[int, int]:
    hu_dir = dest / "hu"
    if not hu_dir.is_dir():
        return 0, 0
    parquets = list(hu_dir.glob("*.parquet"))
    return len(parquets), sum(p.stat().st_size for p in parquets)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    sentinel = args.dest / ".download_complete"
    if sentinel.exists() and not args.dry_run:
        files, total = verify(args.dest)
        print(f"[skip] sentinel exists; {files} parquets, {total / 1024**3:.2f} GB")
        return 0

    print(f"[plan] repo={REPO_ID} patterns={ALLOW_PATTERNS}")
    print(f"[plan] dest={args.dest}")
    print(f"[plan] expected: {EXPECTED_FILES} parquet files, "
          f"~{EXPECTED_TOTAL_BYTES / 1024**3:.2f} GB")

    if args.dry_run:
        return 0

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(args.dest),
        cache_dir=str(args.cache),
        token=load_token(),
        max_workers=4,
    )

    files, total = verify(args.dest)
    print(f"[verify] {files} parquets, {total / 1024**3:.2f} GB")

    tolerance_bytes = 500 * 1024 * 1024
    if files != EXPECTED_FILES or abs(total - EXPECTED_TOTAL_BYTES) > tolerance_bytes:
        print(f"[warn] count/size differs from expected — proceeding anyway",
              file=sys.stderr)

    sentinel.touch()
    print(f"[done] sentinel written: {sentinel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
