#!/usr/bin/env python3
"""Download YODAS2 hu000 (Hungarian, 24 kHz unsegmented audio).

Same content as YODAS v1 hu000 but higher sample rate (24 kHz) and unsegmented
long-form audio (vs v1's pre-segmented 16 kHz utterances). Tarballs unpack to
arbitrary internal structure; we just extract in place.

Source: espnet/yodas2, config hu000
Target: $HU_CORPUS_ROOT/raw/yodas2_hu000/
Size: 32 tarballs, 39.76 GB

Idempotent: snapshot_download skips already-downloaded files (etag/size match);
extraction skipped per-tarball via `.tar.extracted` markers. Re-running after
a stop simply resumes from the next missing file.
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm

REPO_ID = "espnet/yodas2"
CONFIG = "hu000"
ALLOW_PATTERNS = [f"data/{CONFIG}/*"]
EXPECTED_AUDIO_BYTES = 42_692_944_802  # 39.76 GB, from HF API
EXPECTED_AUDIO_FILES = 32

DEFAULT_TOKEN_PATH = Path("/home/cseti/.hf_token")
DEFAULT_DEST = Path("/home/cseti/datassd2/hu-speech-corpus/raw/yodas2_hu000")
DEFAULT_CACHE = Path("/home/cseti/datassd2/hu-speech-corpus/cache")


def load_token() -> str:
    env = os.environ.get("HF_TOKEN")
    if env:
        return env.strip()
    if DEFAULT_TOKEN_PATH.is_file():
        return DEFAULT_TOKEN_PATH.read_text().strip()
    raise SystemExit(f"No HF token: set $HF_TOKEN or place at {DEFAULT_TOKEN_PATH}")


def extract_tarballs(audio_dir: Path) -> None:
    tarballs = sorted(audio_dir.glob("*.tar.gz"))
    if not tarballs:
        print(f"[extract] no tarballs in {audio_dir}", file=sys.stderr)
        return
    for tb in tqdm(tarballs, desc="extracting", unit="tar"):
        marker = tb.with_suffix(".extracted")
        if marker.exists():
            continue
        with tarfile.open(tb, "r:gz") as tf:
            tf.extractall(audio_dir)
        marker.touch()


def verify(dest: Path) -> tuple[int, int, list[Path]]:
    audio_dir = dest / "data" / CONFIG / "audio"
    if not audio_dir.is_dir():
        return 0, 0, []
    tarballs = list(audio_dir.glob("*.tar.gz"))
    total_bytes = sum(p.stat().st_size for p in tarballs)
    bad = []
    for p in tarballs:
        try:
            with gzip.open(p, "rb") as f:
                f.read(4096)
        except (gzip.BadGzipFile, OSError):
            bad.append(p)
    return len(tarballs), total_bytes, bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-extract", action="store_true",
                   help="Skip tarball extraction after download")
    args = p.parse_args()

    token = load_token()
    args.dest.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    sentinel = args.dest / ".download_complete"
    if sentinel.exists() and not args.dry_run:
        print(f"[skip] sentinel exists: {sentinel}")
        files, total, bad = verify(args.dest)
        print(f"[verify] {files} tarballs, {total / 1024**3:.2f} GB, {len(bad)} bad gzip")
        return 0

    print(f"[plan] repo={REPO_ID} config={CONFIG}")
    print(f"[plan] dest={args.dest}")
    print(f"[plan] expected: {EXPECTED_AUDIO_FILES} tarballs, "
          f"{EXPECTED_AUDIO_BYTES / 1024**3:.2f} GB audio")

    if args.dry_run:
        return 0

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(args.dest),
        cache_dir=str(args.cache),
        token=token,
        max_workers=4,
    )

    files, total, bad = verify(args.dest)
    print(f"[verify] {files} tarballs, {total / 1024**3:.2f} GB "
          f"(expected {EXPECTED_AUDIO_FILES}, "
          f"{EXPECTED_AUDIO_BYTES / 1024**3:.2f} GB)")

    tolerance_bytes = 50 * 1024 * 1024
    if files != EXPECTED_AUDIO_FILES or abs(total - EXPECTED_AUDIO_BYTES) > tolerance_bytes:
        print("[error] verification failed: file count or size mismatch", file=sys.stderr)
        return 1

    if bad:
        print(f"[error] {len(bad)} tarball(s) have invalid gzip headers — "
              f"delete and re-run:", file=sys.stderr)
        for p in bad:
            print(f"  {p}", file=sys.stderr)
        return 1

    if not args.no_extract:
        extract_tarballs(args.dest / "data" / CONFIG / "audio")

    sentinel.touch()
    print(f"[done] sentinel written: {sentinel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
