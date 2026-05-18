# hu-speech-corpus

Hungarian speech corpus assembly for ASR/TTS LoRA training. Free-license sources only (CC0, CC-BY, CC-BY-SA, public domain).

## Layout

- **Code + docs (this directory):** `cseti-os/projects/hu-speech-corpus/`
  - `STATUS.md` — current state (read by morning-briefing)
  - `configs/sources.yaml` — per-source metadata (license, size, priority, path)
  - `bin/` — download / preprocess / manifest scripts
  - `notes/` — decisions, experiments, dataset cards

- **Data (NVMe SSD, separate from code):** `/home/cseti/datassd2/hu-speech-corpus/`
  - `raw/` — untouched source downloads (idempotent, re-downloadable)
  - `processed/` — re-encodes, manifests, derived artifacts
  - `cache/` — HuggingFace cache (set `HF_HOME` to this path)

## Environment

Set before running anything in `bin/`:

```bash
export HF_HOME=/home/cseti/datassd2/hu-speech-corpus/cache
export HU_CORPUS_ROOT=/home/cseti/datassd2/hu-speech-corpus
```

## Source priority

See `STATUS.md` for the up-to-date table of sources, sizes, and licenses.

## Not training data

These are kept for **validation / evaluation only** (not training):

- Common Voice HU — quality too inconsistent for training, fine for eval
- FLEURS HU — 12-hour benchmark, eval set
- VoxPopuli HU labeled (63h) — too small for training, eval scale

## Conventions

- All `bin/` scripts must be idempotent (safe to re-run; skip already-downloaded files).
- All audio paths in manifests are absolute (`/home/cseti/datassd2/...`).
- Manifest schema: `{audio_path, text, duration_sec, source, license, speaker_id?, confidence?, hallucination_flags?}`
- License field MUST be set on every sample — downstream filtering depends on it.
