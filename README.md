# hu-speech-corpus

Hungarian speech corpus assembly for ASR / TTS fine-tuning. Free-license sources only (CC0, CC-BY, CC-BY-SA, public domain).

This repository contains the code, configs, and dataset schema for a free-licensed Hungarian speech corpus aggregated from publicly available sources. The audio data itself is not bundled — this repo provides the tooling and provenance metadata to reproduce or extend the corpus.

## Layout

- `configs/sources.yaml` — per-source metadata (license, size, priority, path)
- `bin/` — download / preprocess / manifest / quality-scoring / curator scripts
- `bin/curator/` — Flask + DuckDB single-page web UI to browse parquets
- `STATS.md` — current corpus statistics (rows, hours per source)
- `STATUS.md` — phase-by-phase project status
- `CHECKLIST.md` — pending work tracking

## Data layout

Audio data lives outside this repository (large; not bundled in git). Set the
following environment variables before running anything in `bin/`:

```bash
export HU_CORPUS_ROOT=/path/to/your/corpus/storage   # holds raw/, processed/, cache/
export HF_HOME=$HU_CORPUS_ROOT/cache
```

Recommended sub-structure under `$HU_CORPUS_ROOT`:

- `raw/` — untouched source downloads (idempotent, re-downloadable)
- `processed/` — re-encodes, manifests, derived artifacts
- `cache/` — HuggingFace cache

## Sources

See `configs/sources.yaml` for the full table of sources, licenses, and per-source metadata. The diligence record (which candidate sources were considered and rejected, and why) is also in that file.

## Not training data (eval-only by default)

These are kept for **validation / evaluation only** (not training):

- FLEURS HU — 12-hour benchmark, eval set
- VoxPopuli HU labeled (63h) — too small for training, eval scale
- (Older Common Voice HU placeholder, replaced by `common_voice_25_0_hu`
  below — designation is now TBD)

### Re-evaluation pending

- **`common_voice_25_0_hu`** (180.85h, CC0-1.0, released 2026-03-09) was
  added 2026-05-25 via the Mozilla Data Collective API. Older CV versions
  were marked eval-only due to inconsistent quality, but v25 is a larger,
  newer CC0 read-speech set. Use_for designation (training vs eval)
  pending curator spot-check.

## Conventions

- All `bin/` scripts are idempotent (safe to re-run; skip already-downloaded files).
- Audio paths in manifests are absolute (resolved against `$HU_CORPUS_ROOT` at build time).
- Manifest schema: v5 (current, lean schema) — produced by `bin/build_manifest_v5.py`. Per-row JSONL with `quality_flags` populated by Tier-1 + Tier-2 (VAD / DNSMOS / LID v2 Pass 1) sidecars via `bin/merge_quality_into_manifest.py`.
- License field is required on every sample — downstream filtering depends on it.
- Scripts use developer-default paths at the top of each file (e.g. `DEFAULT_ROOT = Path("/...")`); override via `HU_CORPUS_ROOT` env var or CLI flags. To use the scripts on your own setup, either export `HU_CORPUS_ROOT` or edit the defaults to your local paths.

## Status

Work-in-progress free-licensed Hungarian speech corpus, ~22.5k hours
acquired. As of 2026-05-26:

- Phases 1-2 (acquisition + segmentation + normalization): complete.
- Plan B Silero VAD re-segmentation of the 22k h VoxPopuli unlabeled corpus:
  complete 2026-05-25 (4.28M chunks / 17,993h, user A/B verified).
- Manifest v5 (lean schema): built (4.36M rows / 18,290h).
- Phase 3 (quality re-scoring on new layer): in progress (Tier-1 done,
  DNSMOS + LID v2 running).
- Phase 4 (multi-ASR consensus, Config 4.B with Parakeet TDT v3 as 3rd
  pillar): pending Phase 3.
- Public release on HuggingFace Datasets: pending Phase 4-5.

See `STATUS.md` and `STATS.md` for current canonical numbers.

## License

Apache-2.0 — see `LICENSE`.
