#!/usr/bin/env python3
"""Download MOSEL HU transcripts (FBK-MT/mosel, hu/ subdirectory).

Two TSV files:
  - hu/voxpopuli.tsv  — Whisper-pseudo labels for VoxPopuli HU audio
  - hu/ytc.tsv        — YouTube Commons HU (small / sparse)

These are transcripts only; audio must be fetched separately from VoxPopuli
(facebook/voxpopuli on HF) and YouTube Commons (yt-dlp).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "FBK-MT/mosel"
ALLOW_PATTERNS = ["hu/*"]

DEFAULT_TOKEN_PATH = Path("/home/cseti/.hf_token")
DEFAULT_DEST = Path("/home/cseti/datassd2/hu-speech-corpus/raw/mosel_hu/transcripts")
DEFAULT_CACHE = Path("/home/cseti/datassd2/hu-speech-corpus/cache")


def load_token() -> str:
    env = os.environ.get("HF_TOKEN")
    if env:
        return env.strip()
    return DEFAULT_TOKEN_PATH.read_text().strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    print(f"[plan] repo={REPO_ID} patterns={ALLOW_PATTERNS}")
    print(f"[plan] dest={args.dest}")
    print("[plan] expected: 2 files, ~1.17 GB")

    if args.dry_run:
        return 0

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(args.dest),
        cache_dir=str(args.cache),
        token=load_token(),
        max_workers=2,
    )

    # Verify
    expected = ["hu/voxpopuli.tsv", "hu/ytc.tsv"]
    for rel in expected:
        p = args.dest / rel
        if not p.exists():
            print(f"[error] missing: {p}")
            return 1
        print(f"[ok] {rel}: {p.stat().st_size / 1024**2:.1f} MB")

    sentinel = args.dest / ".download_complete"
    sentinel.touch()
    print(f"[done] sentinel written: {sentinel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
