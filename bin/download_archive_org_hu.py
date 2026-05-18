#!/usr/bin/env python3
"""Download Hungarian free-license audio items from archive.org.

Items are read from configs/sources.yaml — every source whose path lives under
raw/ and which has an `items: [...]` list with `identifier` keys is fetched.

Per item:
  1. Save metadata JSON (full archive.org metadata response) as <ident>.meta.json
  2. Download every audio file (mp3/m4a/wav/flac/ogg/opus) into <target>/<ident>/
  3. Skip files that already exist with matching size (idempotent)
  4. Write a .download_complete sentinel per item when finished
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from tqdm import tqdm

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus"}
# Prefer original-bitrate, lossless-leaning formats. Higher number = preferred.
EXT_PRIORITY = {".flac": 5, ".wav": 4, ".opus": 3, ".mp3": 2, ".ogg": 1, ".m4a": 1}
# Recognize archive.org bitrate-variant suffixes like "_64kb", "_128kb", "_VBR"
BITRATE_SUFFIX_RE = re.compile(r"_(?:\d+kb(?:ps)?|VBR|vbr)$")


def pick_canonical_audio(files: list[dict]) -> list[dict]:
    """Group audio files by base stem, return one canonical variant per group.

    archive.org typically stores each track in 3 variants (e.g. .mp3 original,
    .ogg, _64kb.mp3 low-bitrate). We keep the highest-priority extension among
    the non-bitrate-variant files; fall back to bitrate variants only if no
    canonical version exists.
    """
    groups: dict[str, list[dict]] = {}
    for f in files:
        name = f.get("name", "")
        p = Path(name)
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        base = BITRATE_SUFFIX_RE.sub("", p.stem)
        groups.setdefault(base, []).append(f)
    chosen: list[dict] = []
    for base, variants in groups.items():
        non_lowbitrate = [v for v in variants
                          if not BITRATE_SUFFIX_RE.search(Path(v["name"]).stem)]
        candidates = non_lowbitrate or variants
        best = max(candidates,
                   key=lambda v: EXT_PRIORITY.get(Path(v["name"]).suffix.lower(), 0))
        chosen.append(best)
    return chosen

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "sources.yaml"
DEFAULT_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")


def http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "hu-speech-corpus/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def download_file(url: str, dest: Path, expected_size: int | None = None) -> bool:
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hu-speech-corpus/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0)) or expected_size or 0
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name[:40], leave=False,
        ) as bar:
            while True:
                chunk = r.read(1024 * 128)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return True


def fetch_item(ident: str, target_dir: Path) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    sentinel = target_dir / ".download_complete"
    if sentinel.exists():
        print(f"[skip] {ident} (sentinel)")
        return {"identifier": ident, "skipped": True}

    print(f"[item] {ident} -> {target_dir}")
    meta_url = f"https://archive.org/metadata/{ident}"
    meta = http_get_json(meta_url)
    (target_dir / f"{ident}.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    files = meta.get("files", [])
    all_audio = [f for f in files if Path(f.get("name", "")).suffix.lower() in AUDIO_EXTS]
    audio = pick_canonical_audio(all_audio)
    print(f"  {len(audio)} canonical audio files (from {len(all_audio)} variants)")

    fetched = 0
    skipped = 0
    for f in audio:
        name = f["name"]
        size = int(f.get("size", 0)) or None
        url = f"https://archive.org/download/{ident}/{name}"
        dest = target_dir / name
        try:
            if download_file(url, dest, size):
                fetched += 1
            else:
                skipped += 1
        except urllib.error.URLError as e:
            print(f"  [error] {name}: {e}", file=sys.stderr)

    sentinel.touch()
    print(f"  fetched={fetched} skipped={skipped}")
    return {"identifier": ident, "fetched": fetched, "skipped": skipped}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--source", help="Only fetch this source key (e.g. librivox_hu)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    sources = cfg.get("sources", {})

    targets: list[tuple[str, str, Path]] = []
    for src_key, src in sources.items():
        if args.source and src_key != args.source:
            continue
        items = src.get("items") or []
        if not items:
            continue
        rel_path = src.get("path", "")
        if not rel_path:
            continue
        base = args.root / rel_path
        for it in items:
            ident = it.get("identifier")
            if not ident:
                continue
            targets.append((src_key, ident, base / ident))

    if not targets:
        print("[plan] no items to fetch")
        return 0

    print(f"[plan] {len(targets)} items across {len({t[0] for t in targets})} sources:")
    for src_key, ident, dest in targets:
        print(f"  {src_key:25s} | {ident:40s} -> {dest}")

    if args.dry_run:
        print("[dry-run] exiting")
        return 0

    results = []
    for src_key, ident, dest in targets:
        results.append(fetch_item(ident, dest))

    print()
    print("[summary]")
    for r in results:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
