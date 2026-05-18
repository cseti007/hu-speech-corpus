#!/usr/bin/env python3
"""Idempotent downloader for VoxPopuli HU unlabeled session audio (V2 subset).

The facebookresearch/voxpopuli `download_audios.py` is NOT idempotent — every
re-run redownloads everything from scratch. This wrapper is a drop-in that:

  - generates the same URL list (24 tarballs for `hu_v2`: 12 years × {V1, V2})
  - downloads each tarball with a streaming, resumable HTTP request
  - extracts in place under `raw_audios/`
  - deletes the tarball post-extract (matching upstream behavior)
  - persists per-URL completion in `progress.json` so an interrupted run
    resumes from the next not-yet-completed tarball

Source: https://dl.fbaipublicfiles.com/voxpopuli/audios/hu_<year>[_2].tar
Target: $HU_CORPUS_ROOT/raw/voxpopuli_hu_unlabeled/raw_audios/
Size: ~285 GB across 24 tarballs (Ogg Vorbis 16 kHz mono ~32 kbps, ~17,700 h)

Run:
  python -u bin/download_voxpopuli_hu_unlabeled.py            # full V2
  python -u bin/download_voxpopuli_hu_unlabeled.py --v1-only  # 12 tar only

After this completes, run a separate segmentation step (voxpopuli library's
`get_unlabelled_data`) to split session audio into utterances.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

DOWNLOAD_BASE_URL = "https://dl.fbaipublicfiles.com/voxpopuli/audios"
LANG = "hu"
YEARS = range(2009, 2021)  # 2009..2020 inclusive
DEFAULT_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus/raw/voxpopuli_hu_unlabeled")
USER_AGENT = "hu-speech-corpus/0.1"


def url_list(v1_only: bool) -> list[str]:
    urls = [f"{DOWNLOAD_BASE_URL}/{LANG}_{y}.tar" for y in YEARS]
    if not v1_only:
        urls += [f"{DOWNLOAD_BASE_URL}/{LANG}_{y}_2.tar" for y in YEARS]
    return urls


def http_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0))


def download_streaming(url: str, dest_tar: Path, expected_size: int) -> None:
    """Streaming download with resume support via HTTP Range."""
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    start = 0
    if dest_tar.exists():
        start = dest_tar.stat().st_size
        if start == expected_size:
            return
        if start > expected_size:
            dest_tar.unlink()
            start = 0
        else:
            headers["Range"] = f"bytes={start}-"
            mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        with open(dest_tar, mode) as f, tqdm(
            total=expected_size, initial=start,
            unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest_tar.name, miniters=1,
        ) as bar:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def extract_tar(tar_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        # No "r:gz" — these are uncompressed .tar archives per upstream.
        tf.extractall(extract_to)


def process_url(url: str, root: Path, progress: dict) -> str:
    """Returns one of: 'skip', 'fetched', 'error'."""
    if progress.get(url) == "completed":
        return "skip"

    raw_dir = root / "raw_audios"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_name = url.rsplit("/", 1)[-1]
    tar_path = raw_dir / tar_name

    try:
        expected = http_size(url)
    except Exception as e:
        print(f"[error] HEAD failed for {url}: {e}", file=sys.stderr)
        progress[url] = f"error_head:{e}"
        return "error"

    try:
        download_streaming(url, tar_path, expected)
    except Exception as e:
        print(f"[error] download failed for {url}: {e}", file=sys.stderr)
        progress[url] = f"error_download:{e}"
        return "error"

    try:
        extract_tar(tar_path, raw_dir)
    except Exception as e:
        print(f"[error] extract failed for {tar_path}: {e}", file=sys.stderr)
        progress[url] = f"error_extract:{e}"
        return "error"

    tar_path.unlink()
    progress[url] = "completed"
    return "fetched"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--v1-only", action="store_true",
                   help="Download only V1 subset (12 tarballs, no _2 variants)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    progress_path = args.root / "progress.json"
    progress: dict = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())

    urls = url_list(args.v1_only)
    print(f"[plan] root={args.root}")
    print(f"[plan] {len(urls)} tarballs to consider ({'V1 only' if args.v1_only else 'V1 + V2'})")

    # Probe sizes for visibility (HEAD only, fast)
    if args.dry_run:
        total = 0
        for u in urls:
            try:
                sz = http_size(u)
                total += sz
                print(f"  {sz/1024**3:>7.2f} GB  {u}")
            except Exception as e:
                print(f"  HEAD-fail  {u}: {e}")
        print(f"\nTotal: {total/1024**3:.2f} GB")
        return 0

    n_skip = n_fetch = n_err = 0
    for url in urls:
        status = process_url(url, args.root, progress)
        progress_path.write_text(json.dumps(progress, indent=2))
        if status == "skip":
            n_skip += 1
        elif status == "fetched":
            n_fetch += 1
        else:
            n_err += 1
        print(f"[{status}] {url}")

    print()
    print(f"[summary] fetched={n_fetch}, skipped={n_skip}, errors={n_err}")
    if n_err == 0:
        (args.root / ".download_complete").touch()
        print(f"[done] sentinel written")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
