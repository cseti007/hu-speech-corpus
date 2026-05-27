#!/usr/bin/env python3
"""Download Common Voice Scripted Speech 25.0 - Hungarian via Mozilla Data Collective API.

Source: Mozilla Data Collective (mozilladatacollective.com)
  Dataset: cmn2g9aoi01fyo107xhdrwb5d
  Format:  tar.gz (MP3 audio + TSV metadata)
  Size:    ~3.58 GB (3,842,633,394 bytes)
  License: CC0-1.0

Target:  $HU_CORPUS_ROOT/raw/common_voice_25_0_hu/

API flow:
  1) POST /api/datasets/<id>/download with Bearer key -> presigned URL (Cloudflare R2,
     12 h validity).
  2) GET the presigned URL with HTTP Range header for resume support.
  3) Extract the tar.gz in place.

Idempotent: sentinel files `.download_complete` and `.extract_complete` guard the two
phases. The presigned URL is re-issued on every run, so partial downloads survive
interruptions even past the 12 h URL TTL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://mozilladatacollective.com/api"
DATASET_ID = "cmn2g9aoi01fyo107xhdrwb5d"
EXPECTED_FILENAME = "common-voice-scripted-speech-25-0-hungar-f272f397.tar.gz"
EXPECTED_SIZE = 3_842_633_394  # bytes, from API metadata

DEFAULT_KEY_PATH = Path("/home/cseti/.cv_key")
DEFAULT_DEST = Path("/home/cseti/datassd2/hu-speech-corpus/raw/common_voice_25_0_hu")


def load_key(path: Path) -> str:
    env = os.environ.get("CV_KEY")
    if env:
        return env.strip()
    if path.is_file():
        return path.read_text().strip()
    raise SystemExit(f"No CV key: set $CV_KEY or place at {path}")


def request_download_url(api_key: str) -> str:
    url = f"{API_BASE}/datasets/{DATASET_ID}/download"
    req = urllib.request.Request(
        url,
        method="POST",
        data=b"{}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"[error] POST {url} HTTP {e.code}: {body}")
    durl = data.get("downloadUrl")
    if not durl:
        raise SystemExit(f"[error] no downloadUrl in response: {data}")
    return durl


def download_with_resume(url: str, dest_file: Path, expected_size: int) -> None:
    cur = dest_file.stat().st_size if dest_file.exists() else 0
    if cur > expected_size:
        print(f"[warn] local file larger than expected ({cur} > {expected_size}), restarting")
        dest_file.unlink()
        cur = 0
    if cur == expected_size:
        print(f"[skip] already complete: {dest_file} ({cur:,} bytes)")
        return

    headers = {}
    if cur > 0:
        headers["Range"] = f"bytes={cur}-"
        print(f"[resume] from byte {cur:,} ({cur / expected_size * 100:.1f}%)")
    else:
        print(f"[download] starting fresh ({expected_size:,} bytes)")

    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if cur > 0 else "wb"
    last_log = cur
    log_interval = 50 * 1024 * 1024  # 50 MiB
    with urllib.request.urlopen(req, timeout=60) as r, open(dest_file, mode) as f:
        chunk = 1 << 20  # 1 MiB
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            cur += len(buf)
            if cur - last_log >= log_interval:
                pct = cur / expected_size * 100
                print(f"  {cur:,} / {expected_size:,} bytes ({pct:.1f}%)")
                last_log = cur
    print(f"[download] complete: {cur:,} bytes")


def extract_targz(tarball: Path, dest_dir: Path) -> None:
    print(f"[extract] {tarball.name} -> {dest_dir}")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest_dir)
    print("[extract] done")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--key-path", type=Path, default=DEFAULT_KEY_PATH)
    p.add_argument("--no-extract", action="store_true",
                   help="Download tarball only, skip extraction")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    tarball = args.dest / EXPECTED_FILENAME
    sentinel_dl = args.dest / ".download_complete"
    sentinel_ex = args.dest / ".extract_complete"

    print(f"[plan] dataset_id={DATASET_ID}")
    print(f"[plan] dest={args.dest}")
    print(f"[plan] tarball={tarball.name}")
    print(f"[plan] expected_size={EXPECTED_SIZE:,} bytes "
          f"(~{EXPECTED_SIZE / 1024**3:.2f} GB)")

    if args.dry_run:
        return 0

    # Phase 1: download
    if sentinel_dl.exists() and tarball.exists() and tarball.stat().st_size == EXPECTED_SIZE:
        print(f"[skip] download sentinel exists; tarball {tarball.stat().st_size:,} bytes")
    else:
        key = load_key(args.key_path)
        print(f"[fetch] requesting presigned URL")
        download_url = request_download_url(key)
        host = download_url.split("/")[2] if "://" in download_url else "?"
        print(f"[fetch] got presigned URL (host={host})")
        download_with_resume(download_url, tarball, EXPECTED_SIZE)
        actual = tarball.stat().st_size
        if actual != EXPECTED_SIZE:
            raise SystemExit(f"[error] size mismatch: got {actual:,}, expected {EXPECTED_SIZE:,}")
        sentinel_dl.touch()
        print(f"[done] download sentinel: {sentinel_dl}")

    # Phase 2: extract
    if args.no_extract:
        print("[skip] extraction skipped (--no-extract)")
        return 0
    if sentinel_ex.exists():
        print(f"[skip] extract sentinel exists")
        return 0
    extract_targz(tarball, args.dest)
    sentinel_ex.touch()
    print(f"[done] extract sentinel: {sentinel_ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
