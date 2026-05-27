#!/usr/bin/env python3
"""Investigate YODAS v1 vs YODAS2 hu000 duration discrepancy.

Compares pre-segmented v1 against unsegmented v2-with-JSON-alignment to determine
whether the ~3x duration difference is from:
  (A) v2 covering more captions than v1's manual-only subset,
  (B) overlapping v2 JSON timestamps inflating the segment-sum,
  (C) v1 and v2 covering different video sets, or
  (D) v2 JSON segments extending past the real WAV duration.

Read-only. Writes a markdown report to notes/YODAS_v1_v2_analysis.md and prints
key numbers to stdout.
"""

import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/cseti/datassd2/hu-speech-corpus/raw")
V1 = ROOT / "yodas_hu000/data/hu000"
V2 = ROOT / "yodas2_hu000/data/hu000"
OUT = Path(__file__).resolve().parent.parent / "notes" / "YODAS_v1_v2_analysis.md"

# utterance / segment ID: {videoId}-{seq:05d}-{startCS:08d}-{endCS:08d}
# where startCS/endCS are centiseconds (10ms units) per YODAS convention.
UTT_RE = re.compile(r"^(.+)-(\d{5})-(\d{8})-(\d{8})$")


def parse_utt_id(utt_id):
    m = UTT_RE.match(utt_id)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def merge_intervals(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def build_v1_index():
    by_video = defaultdict(list)
    skipped = 0
    for dur_file in sorted((V1 / "duration").glob("*.txt")):
        with dur_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, _ = parts
                parsed = parse_utt_id(utt_id)
                if not parsed:
                    skipped += 1
                    continue
                vid, _seq, s, e = parsed
                by_video[vid].append((s, e))
    if skipped:
        print(f"v1: skipped {skipped} unparseable utt_ids", file=sys.stderr)
    return by_video


def build_v2_index():
    by_video = defaultdict(list)
    wav_dur = {}
    skipped = 0
    for txt_file in sorted((V2 / "text").glob("*.json")):
        with txt_file.open() as f:
            arr = json.load(f)
        for entry in arr:
            vid = entry["audio_id"]
            for seg_id in entry.get("text", {}):
                parsed = parse_utt_id(seg_id)
                if not parsed:
                    skipped += 1
                    continue
                _, _seq, s, e = parsed
                by_video[vid].append((s, e))
    if skipped:
        print(f"v2: skipped {skipped} unparseable seg_ids", file=sys.stderr)

    for dur_file in sorted((V2 / "duration").glob("*.txt")):
        with dur_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                vid, dur_str = parts
                try:
                    wav_dur[vid] = float(dur_str)
                except ValueError:
                    continue
    return by_video, wav_dur


def ffprobe_duration(wav_path):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
            stderr=subprocess.DEVNULL, timeout=30,
        )
        return float(out.strip())
    except Exception as ex:
        print(f"ffprobe failed for {wav_path}: {ex}", file=sys.stderr)
        return None


def find_v2_wav(video_id):
    for shard in sorted((V2 / "audio").glob("*.tar.extracted")):
        candidate = shard / f"{video_id}.wav"
        if candidate.exists():
            return candidate
    return None


def main():
    print("Building v1 index...", file=sys.stderr)
    v1 = build_v1_index()
    print(f"v1: {len(v1)} videos", file=sys.stderr)

    print("Building v2 index...", file=sys.stderr)
    v2, v2_wav_dur = build_v2_index()
    print(f"v2: {len(v2)} videos with segments, {len(v2_wav_dur)} videos with wav-duration",
          file=sys.stderr)

    v1_videos = set(v1)
    v2_videos = set(v2)
    common = v1_videos & v2_videos
    v1_only = v1_videos - v2_videos
    v2_only = v2_videos - v1_videos

    v1_utt_total = sum(len(v) for v in v1.values())
    v1_dur_total = sum((e - s) for ivs in v1.values() for s, e in ivs) / 100.0

    v2_utt_total = sum(len(v) for v in v2.values())
    v2_seg_sum_total = sum((e - s) for ivs in v2.values() for s, e in ivs) / 100.0
    v2_union_total = sum(merge_intervals(ivs) for ivs in v2.values()) / 100.0
    v2_wav_total = sum(v2_wav_dur.values())

    v1_dur_common = sum((e - s) for vid in common for s, e in v1[vid]) / 100.0
    v2_seg_sum_common = sum((e - s) for vid in common for s, e in v2[vid]) / 100.0
    v2_union_common = sum(merge_intervals(v2[vid]) for vid in common) / 100.0
    v2_wav_common = sum(v2_wav_dur.get(vid, 0) for vid in common)

    random.seed(42)
    sample_vids = random.sample(sorted(common), min(10, len(common)))

    sample_rows = []
    for vid in sample_vids:
        v1_n = len(v1[vid])
        v1_d = sum((e - s) for s, e in v1[vid]) / 100.0
        v2_n = len(v2[vid])
        v2_d_sum = sum((e - s) for s, e in v2[vid]) / 100.0
        v2_d_union = merge_intervals(v2[vid]) / 100.0
        v2_wav_d = v2_wav_dur.get(vid, 0.0)
        wav_path = find_v2_wav(vid)
        ffp = ffprobe_duration(wav_path) if wav_path else None
        sample_rows.append({
            "vid": vid,
            "v1_n": v1_n, "v1_d": v1_d,
            "v2_n": v2_n, "v2_d_sum": v2_d_sum, "v2_d_union": v2_d_union,
            "v2_wav_d": v2_wav_d, "ffp": ffp,
        })

    def h(s):
        return s / 3600.0

    lines = []
    lines.append("# YODAS v1 vs YODAS2 hu000 - duration discrepancy analysis")
    lines.append("")
    lines.append("_Generated by `bin/investigate_yodas2.py`._")
    lines.append("")
    lines.append("Goal: determine whether the ~3x duration difference between v1 (172h) and v2 (512h)")
    lines.append("on hu000 reflects real new audio coverage or is an artifact (overlapping timestamps,")
    lines.append("different video sets, or JSON-vs-WAV mismatch).")
    lines.append("")
    lines.append("Note on timestamp units: YODAS utterance/segment IDs encode start/end as")
    lines.append("centiseconds (10ms units), so duration_sec = (end - start) / 100.")
    lines.append("")
    lines.append("## Video-ID overlap")
    lines.append("")
    lines.append(f"- v1 videos: **{len(v1_videos):,}**")
    lines.append(f"- v2 videos: **{len(v2_videos):,}**")
    lines.append(f"- Common: **{len(common):,}**")
    lines.append(f"- v1 only: **{len(v1_only):,}**")
    lines.append(f"- v2 only: **{len(v2_only):,}**")
    lines.append("")
    lines.append("## Aggregate hours")
    lines.append("")
    lines.append("| Metric | Utterances | Hours |")
    lines.append("|---|---:|---:|")
    lines.append(f"| v1 segment-sum (all v1 videos) | {v1_utt_total:,} | {h(v1_dur_total):.2f} |")
    lines.append(f"| v2 segment-sum naive (all v2 videos) | {v2_utt_total:,} | {h(v2_seg_sum_total):.2f} |")
    lines.append(f"| v2 segment union-coverage (all v2 videos) | - | {h(v2_union_total):.2f} |")
    lines.append(f"| v2 full WAV duration sum (all v2 videos) | - | {h(v2_wav_total):.2f} |")
    lines.append("")
    lines.append("### Restricted to common videos")
    lines.append("")
    lines.append("| Metric | Hours |")
    lines.append("|---|---:|")
    lines.append(f"| v1 segment-sum | {h(v1_dur_common):.2f} |")
    lines.append(f"| v2 segment-sum naive | {h(v2_seg_sum_common):.2f} |")
    lines.append(f"| v2 segment union-coverage | {h(v2_union_common):.2f} |")
    lines.append(f"| v2 full WAV duration | {h(v2_wav_common):.2f} |")
    lines.append("")
    lines.append("## 10 random common videos - detailed breakdown")
    lines.append("")
    lines.append("| videoId | v1 utts | v1 dur (s) | v2 utts | v2 seg sum (s) | v2 union (s) | v2 wav dur (s) | ffprobe (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sample_rows:
        ffp_s = f"{r['ffp']:.2f}" if r["ffp"] is not None else "n/a"
        lines.append(
            f"| `{r['vid']}` | {r['v1_n']} | {r['v1_d']:.2f} | {r['v2_n']} | "
            f"{r['v2_d_sum']:.2f} | {r['v2_d_union']:.2f} | {r['v2_wav_d']:.2f} | {ffp_s} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("_To be filled in based on the numbers above (see CHECKLIST and STATS update)._")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}", file=sys.stderr)

    print()
    print("=== KEY NUMBERS ===")
    print(f"Video overlap: |v1|={len(v1_videos)}, |v2|={len(v2_videos)}, "
          f"|common|={len(common)}, v1-only={len(v1_only)}, v2-only={len(v2_only)}")
    print(f"v1 total:          {h(v1_dur_total):.2f}h ({v1_utt_total:,} utts)")
    print(f"v2 seg-sum total:  {h(v2_seg_sum_total):.2f}h ({v2_utt_total:,} utts)")
    print(f"v2 union total:    {h(v2_union_total):.2f}h")
    print(f"v2 wav-dur total:  {h(v2_wav_total):.2f}h")
    print()
    print("On common videos only:")
    print(f"  v1 seg-sum:        {h(v1_dur_common):.2f}h")
    print(f"  v2 seg-sum naive:  {h(v2_seg_sum_common):.2f}h")
    print(f"  v2 union coverage: {h(v2_union_common):.2f}h")
    print(f"  v2 full WAV dur:   {h(v2_wav_common):.2f}h")


if __name__ == "__main__":
    main()
