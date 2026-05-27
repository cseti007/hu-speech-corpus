#!/usr/bin/env python3
"""VoxPopuli HU session segmentation with low-RAM streaming I/O.

Uses the official facebookresearch/voxpopuli manifest (unlabelled_v2.tsv.gz)
for per-utterance offsets, but does the actual audio I/O with `soundfile`
(libsndfile) instead of torchaudio. Streaming means each worker only holds
~one clip's worth of audio in memory at a time (~5 MB), not the entire
1-hour session waveform (~230 MB). This dramatically reduces RAM pressure
and lets us safely run with more workers.

Implementation notes:
- libsndfile cannot seek inside ogg/vorbis (psf_fseek fails), so we sort
  the session's clips by start time and read sequentially, consuming any
  inter-clip gaps with read-and-discard.
- Per-clip output is the same ogg/vorbis format as the source.
- Idempotent: clips whose output ogg already exists are skipped.

Reads sessions from:  raw/voxpopuli_hu_unlabeled/raw_audios/hu/<year>/*.ogg
Writes clips to:      raw/voxpopuli_hu_unlabeled/unlabelled_data/hu/<year>/<event_id>_<seg>.ogg

Run with the dedicated conda env:
  /media/cseti/datassd/conda/miniconda3/envs/hu-speech-corpus/bin/python \
      bin/segment_voxpopuli.py --n_workers 16
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "external" / "voxpopuli"))

import soundfile as sf  # noqa: E402
import voxpopuli.get_unlabelled_data as gud  # noqa: E402
from voxpopuli import utils as vp_utils  # noqa: E402

VP_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus/raw/voxpopuli_hu_unlabeled")
SENTINEL = VP_ROOT / ".segmentation_complete"

# Set by main() before workers are spawned (inherited via fork on Linux).
N_WORKERS = 8


def _patched_segment(item):
    """Streaming replacement for voxpopuli.get_unlabelled_data._segment.

    Uses soundfile sequential reads to keep per-worker RAM constant (~10 MB)
    regardless of session length. Skips clips whose output already exists.
    """
    in_path, segments, out_root = item
    _in_path = Path(in_path)
    event_id = _in_path.stem
    lang = _in_path.parent.parent.stem
    year = _in_path.parent.stem
    out_dir = Path(out_root) / lang / year

    remaining = [
        (i, s, e) for i, s, e in segments
        if not (out_dir / f"{event_id}_{i}.ogg").exists()
    ]
    if not remaining:
        return

    # Sequential read requires sorted clips (libsndfile cannot seek in ogg/vorbis)
    remaining.sort(key=lambda x: float(x[1]))

    try:
        with sf.SoundFile(str(in_path)) as f:
            sr = f.samplerate
            prev_end_frame = 0
            for i, s, e in remaining:
                start_frame = int(float(s) * sr)
                end_frame = int(float(e) * sr)
                if start_frame < prev_end_frame:
                    # Would require backward seek — not supported. Skip.
                    continue
                if start_frame > prev_end_frame:
                    # Consume the inter-clip gap.
                    f.read(start_frame - prev_end_frame)
                num_frames = end_frame - start_frame
                data = f.read(num_frames)
                if data.shape[0] == 0:
                    # Session shorter than the manifest says — stop.
                    break
                out_path = out_dir / f"{event_id}_{i}.ogg"
                sf.write(str(out_path), data, sr,
                         format="OGG", subtype="VORBIS")
                prev_end_frame = start_frame + data.shape[0]
    except Exception as ex:
        print(f"[ERROR] {event_id}: {ex}", file=sys.stderr)


def _patched_multiprocess_run(a_list, func, n_workers_ignored=None):
    """Replacement that pins n_workers to our configured value."""
    from tqdm.contrib.concurrent import process_map
    process_map(func, a_list, max_workers=N_WORKERS, chunksize=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_workers", type=int, default=16,
                        help="Worker process count (default: 16). With the "
                             "streaming soundfile backend, per-worker RAM is "
                             "constant (~10 MB), so this can safely run with "
                             "more workers than the torchaudio backend.")
    parser.add_argument("--subset", default="hu_v2",
                        choices=["hu", "hu_v2"],
                        help="hu = PLENARY only; hu_v2 = all Hungarian sessions "
                             "(matches MOSEL utterance set; recommended).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if .segmentation_complete sentinel exists.")
    args = parser.parse_args()

    if SENTINEL.exists() and not args.force:
        print(f"[skip] segmentation already complete (sentinel: {SENTINEL})")
        return 0

    global N_WORKERS
    N_WORKERS = args.n_workers

    print(f"[segment_voxpopuli] backend=soundfile (streaming)")
    print(f"[segment_voxpopuli] subset={args.subset}, n_workers={args.n_workers}")
    print(f"[segment_voxpopuli] root={VP_ROOT}")

    vp_utils.multiprocess_run = _patched_multiprocess_run
    gud.multiprocess_run = _patched_multiprocess_run
    gud._segment = _patched_segment

    sub_args = argparse.Namespace(root=str(VP_ROOT), subset=args.subset)
    gud.get(sub_args)

    SENTINEL.touch()
    print(f"[done] sentinel written: {SENTINEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
