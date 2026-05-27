#!/usr/bin/env python3
"""Normalize YODAS2 captions into 3-30s merged clips.

Logic:
- For each YODAS2 video, sort captions by start_cs.
- Drop the outlier-timestamp segments (same 1.1x video_dur filter as the
  manifest builder).
- Greedily merge consecutive captions where the gap is <= 1 sec AND the
  merged duration stays <= 30 sec.
- Emit only merged clips where duration >= 3 sec and the merged text is
  non-empty.

Output sidecar: processed/normalization/yodas2_merged.jsonl. Each row:
  {
    "audio_id": "...",
    "merged_utt_id": "{audio_id}-{first_seg:05d}-{last_seg:05d}",
    "start_sec": 12.5,
    "end_sec": 18.7,
    "duration_sec": 6.2,
    "text": "merged caption text",
    "merged_from": ["original_utt_id_1", "original_utt_id_2", ...],
    "video_duration_sec": 1234.5
  }

CPU-only, ~5-15 min on the full 177k captions. Idempotent: overwrites the
sidecar on each run.

Run with the base conda env:
  /media/cseti/datassd/conda/miniconda3/bin/python bin/normalize_yodas2.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# --- Paths
YODAS2_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus/raw/yodas2_hu000/data/hu000")
TEXT_DIR = YODAS2_ROOT / "text"
DURATION_DIR = YODAS2_ROOT / "duration"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path("/home/cseti/datassd2/hu-speech-corpus/processed/normalization")
OUT_PATH = OUT_DIR / "yodas2_merged.jsonl"

# --- Constants
# YODAS utt id: {audio_id}-{seg:05d}-{start_cs:08d}-{end_cs:08d}, with cs = centiseconds
UTT_RE = re.compile(r"^(.+)-(\d{5})-(\d{8})-(\d{8})$")
GAP_THRESHOLD_CS = 100       # 1.0 sec
MIN_DURATION_CS = 300        # 3.0 sec (post-merge)
MAX_DURATION_CS = 3000       # 30.0 sec (post-merge)
OUTLIER_MULT = 1.1           # drop individual seg if dur > video_dur * 1.1


def parse_utt_id(utt_id: str):
    m = UTT_RE.match(utt_id)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def load_video_durations():
    """Return dict: video_id -> duration_sec (float)."""
    dur = {}
    for f in sorted(DURATION_DIR.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            vid, secs = parts
            try:
                dur[vid] = float(secs)
            except ValueError:
                pass
    return dur


def load_captions_per_video():
    """Return dict: video_id -> list of {seg, start_cs, end_cs, text, utt_id}.

    Captions per video are sorted by start_cs.
    """
    per_video: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for f in sorted(TEXT_DIR.glob("*.json")):
        with f.open() as h:
            entries = json.load(h)
        for entry in entries:
            vid = entry.get("audio_id")
            if not vid:
                continue
            for utt_id, text in entry.get("text", {}).items():
                parsed = parse_utt_id(utt_id)
                if not parsed:
                    skipped += 1
                    continue
                vid_check, seg, start_cs, end_cs = parsed
                per_video[vid].append({
                    "seg": seg,
                    "start_cs": start_cs,
                    "end_cs": end_cs,
                    "text": (text or "").strip(),
                    "utt_id": utt_id,
                })
    for vid in per_video:
        per_video[vid].sort(key=lambda x: x["start_cs"])
    print(f"[load] {len(per_video):,} videos with captions "
          f"({skipped} unparseable utt_ids skipped)", file=sys.stderr)
    return per_video


def filter_outliers(captions: list[dict], video_dur_sec: float) -> tuple[list[dict], int]:
    """Drop captions whose individual duration exceeds video_dur * OUTLIER_MULT.
    Also drop captions with non-positive duration. Returns (filtered, n_dropped).
    """
    if video_dur_sec <= 0:
        return captions, 0
    max_dur_cs = int(video_dur_sec * 100 * OUTLIER_MULT)
    kept = []
    dropped = 0
    for c in captions:
        dur_cs = c["end_cs"] - c["start_cs"]
        if dur_cs <= 0 or dur_cs > max_dur_cs:
            dropped += 1
            continue
        kept.append(c)
    return kept, dropped


def merge_captions(captions: list[dict]) -> list[list[dict]]:
    """Group consecutive captions into merged clips.

    Rules:
    - gap (next.start - cur.end) <= GAP_THRESHOLD_CS
    - merged duration (last.end - first.start) <= MAX_DURATION_CS

    Returns list of groups; each group is a list of caption dicts.
    """
    if not captions:
        return []
    groups: list[list[dict]] = []
    current: list[dict] = [captions[0]]
    for cap in captions[1:]:
        gap = cap["start_cs"] - current[-1]["end_cs"]
        merged_dur = cap["end_cs"] - current[0]["start_cs"]
        if gap <= GAP_THRESHOLD_CS and merged_dur <= MAX_DURATION_CS:
            current.append(cap)
        else:
            groups.append(current)
            current = [cap]
    groups.append(current)
    return groups


def emit_clip(video_id: str, group: list[dict], video_dur_sec: float):
    """Build the output row for a merged group, or return None if it should be dropped."""
    first = group[0]
    last = group[-1]
    start_cs = first["start_cs"]
    end_cs = last["end_cs"]
    dur_cs = end_cs - start_cs
    if dur_cs < MIN_DURATION_CS or dur_cs > MAX_DURATION_CS:
        return None
    text = " ".join(c["text"] for c in group if c["text"]).strip()
    if not text:
        return None
    merged_utt_id = f"{video_id}-{first['seg']:05d}-{last['seg']:05d}"
    return {
        "audio_id": video_id,
        "merged_utt_id": merged_utt_id,
        "start_sec": round(start_cs / 100.0, 3),
        "end_sec": round(end_cs / 100.0, 3),
        "duration_sec": round(dur_cs / 100.0, 3),
        "text": text,
        "merged_from": [c["utt_id"] for c in group],
        "video_duration_sec": round(video_dur_sec, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-videos", type=int, default=None,
                        help="Debug: process only the first N videos.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    video_durations = load_video_durations()
    per_video = load_captions_per_video()

    # Stats
    n_videos_processed = 0
    n_videos_with_no_captions = 0
    n_captions_input = 0
    n_captions_outlier_dropped = 0
    n_clips_emitted = 0
    n_clips_dropped_short = 0
    n_clips_dropped_long = 0
    n_clips_dropped_empty_text = 0
    total_dur_emitted_sec = 0.0

    video_ids = sorted(per_video.keys())
    if args.limit_videos:
        video_ids = video_ids[:args.limit_videos]

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for vid in video_ids:
            captions = per_video[vid]
            if not captions:
                n_videos_with_no_captions += 1
                continue
            n_videos_processed += 1
            n_captions_input += len(captions)

            video_dur = video_durations.get(vid, 0.0)
            captions, dropped_outlier = filter_outliers(captions, video_dur)
            n_captions_outlier_dropped += dropped_outlier

            groups = merge_captions(captions)
            for group in groups:
                clip = emit_clip(vid, group, video_dur)
                if clip is None:
                    # Distinguish reason for stats
                    first, last = group[0], group[-1]
                    dur_cs = last["end_cs"] - first["start_cs"]
                    if dur_cs < MIN_DURATION_CS:
                        n_clips_dropped_short += 1
                    elif dur_cs > MAX_DURATION_CS:
                        n_clips_dropped_long += 1
                    else:
                        n_clips_dropped_empty_text += 1
                    continue
                out.write(json.dumps(clip, ensure_ascii=False) + "\n")
                n_clips_emitted += 1
                total_dur_emitted_sec += clip["duration_sec"]

    print()
    print("=== YODAS2 normalization summary ===")
    print(f"Videos processed:                 {n_videos_processed:,}")
    print(f"Videos with no captions:          {n_videos_with_no_captions:,}")
    print(f"Captions (input):                 {n_captions_input:,}")
    print(f"Captions dropped (outlier dur):   {n_captions_outlier_dropped:,}")
    print(f"Merged clips emitted:             {n_clips_emitted:,}")
    print(f"Clips dropped (<3s after merge):  {n_clips_dropped_short:,}")
    print(f"Clips dropped (>30s edge case):   {n_clips_dropped_long:,}")
    print(f"Clips dropped (empty text):       {n_clips_dropped_empty_text:,}")
    print(f"Total duration emitted:           {total_dur_emitted_sec / 3600:.2f}h")
    print()
    print(f"Output: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
