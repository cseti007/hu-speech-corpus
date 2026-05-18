# Manifest schema v3 (current)

JSONL files written to `processed/manifests/`:

- `train_aligned.jsonl` — utterances with both audio AND transcript ready, ASR-trainable as-is
- `train_unaligned.jsonl` — long-form audio without transcript (LibriVox, podcasts, YODAS2 sessions, VoxPopuli unlabeled sessions)
- `train_pseudo_labeled.jsonl` — MOSEL Whisper-pseudo-labels with `audio_path=null` until segmentation step runs (parent_session_id refs the source .ogg)
- `stats.json` — per-source counts and totals

## v3 changes from v2
- Split aligned/unaligned/pseudo-labeled into 3 files (was 2 files)
- New field `audio_format` (`wav` / `ogg` / `mp3` / `flac` / `parquet`)
- New field `parquet_row_index` (only for parquet-internal audio, e.g. VoxPopuli HU labeled)
- New field `segmentation_status` (`segmented_file` / `virtual_segment` / `session_level` / `pending_segmentation`)
- New field `parent_session_id` (links MOSEL utterances to VoxPopuli session ogg)
- License field uses SPDX shorts derived from license_url

One row per audio item. All fields are present on every row; values that
haven't been computed yet are `null` (placeholder for later pipeline stages).

## Fields

### Identification

| Field | Type | Notes |
|---|---|---|
| `utterance_id` | str | Unique. For YODAS: filename stem. For archive.org: `<item>/<file_stem>`. |
| `source` | str | Source key from `configs/sources.yaml` (e.g. `yodas_hu000`). |
| `source_item_id` | str? | YouTube video ID for YODAS; archive.org item ID otherwise. |

### Audio

| Field | Type | Notes |
|---|---|---|
| `audio_path` | str | Absolute path on this machine. |
| `relative_audio_path` | str | Relative to `/home/cseti/datassd2/hu-speech-corpus`. Portable. |
| `sample_rate` | int | Hz. |
| `channels` | int | 1 (mono) or 2 (stereo). Long-form podcasts often stereo. |
| `codec` | str | `wav`, `mp3`, `flac`, etc. |
| `duration_sec` | float | Total file or segment duration. |
| `segment_start_sec` | float? | If audio is a segment of a parent file (YODAS-style). |
| `segment_end_sec` | float? | Same. |
| `parent_audio_path` | str? | YouTube URL for YODAS; null otherwise. |

### Transcripts — multi-ASR ready

`transcripts` is a dict keyed by transcript source. Currently only the YODAS
`source_caption` is filled. Future ASR runs will add more keys.

| Key (in `transcripts`) | When populated |
|---|---|
| `source_caption` | At manifest build (YODAS only) |
| `whisper_large_v3` | When `multi_asr_transcribe.py` runs |
| `wav2vec2_xlsr_hu` | Same |
| `mms_1b_fl102` | Same |

Consensus result fields:

| Field | Type | Notes |
|---|---|---|
| `text_consensus` | str? | Picked or constructed transcript that all (or majority) ASRs agree on. |
| `consensus_method` | str? | e.g. `exact_match_normalized`, `wer_threshold_0.05`, `rover_majority`. |
| `confidence_level` | str | `HIGH` / `MEDIUM` / `LOW` / `UNKNOWN`. |
| `pairwise_wer` | dict? | `{"whisper_vs_wav2vec2": 0.03, ...}`. |
| `hallucination_flags` | dict? | MOSEL-style flags for Whisper pseudo-labels. |

### Speaker / domain

| Field | Type | Notes |
|---|---|---|
| `speaker_id` | str? | YouTube video ID, audiobook narrator, podcast item ident. |
| `domain` | str | `youtube` / `audiobook` / `podcast` / `parliament` / ... |
| `register` | str | `unknown` / `narrated` / `spontaneous` / `interview` / ... |
| `language` | str | Always `hu`. |

### License / provenance

| Field | Type | Notes |
|---|---|---|
| `license` | str | SPDX-style short ID: `CC-BY-4.0`, `CC-BY-3.0`, `CC-BY-SA-4.0`, `CC0-1.0`, `PD`, etc. |
| `license_url` | str? | The canonical license URL for this item. |
| `attribution` | str? | Human-readable attribution string (required for CC-BY). |

### Split / quality

| Field | Type | Notes |
|---|---|---|
| `split` | str | `train` / `val` / `test`. Currently all `train`; split logic TBD. |
| `quality_flags` | dict | Grows over time. See below. |

## quality_flags evolution

### Tier 1 (computed at manifest build)
Filled now:
- `too_short` (bool) — duration < 1.0s AND segmented utterance
- `too_long` (bool) — duration ≥ 30.0s AND segmented utterance

Length flags only fire for already-segmented utterances (YODAS-style).
Long-form audiobook chapters and podcast episodes never trigger them.

### Tier 2 (deferred to `compute_quality_tier1.py`)
- `rms_dbfs` (float) — average level
- `peak_dbfs` (float) — peak level
- `is_clipped` (bool) — peak ≥ -0.1 dBFS
- `silence_ratio` (float) — fraction of energy-below-threshold frames

### Tier 3 (deferred to `compute_neural_quality.py`)
- `predicted_mos_dnsmos` (float) — DNSMOS-predicted MOS (1-5)
- `predicted_mos_utmos` (float) — UTMOSv2-predicted MOS (1-5)
- `vad_speech_ratio` (float) — pyannote VAD speech fraction
- `music_likelihood` (float) — YAMNet music probability
- `quality_tier` (str) — derived `HIGH` / `MEDIUM` / `LOW`

## Multi-ASR consensus — design notes

Hypothesis: when multiple INDEPENDENT ASRs agree on a transcript, that
agreement is a strong precision signal for "clean training data".

Caveats:
- Whisper v1/v2/v3/v3-turbo share an architecture → highly correlated. Use
  ONE Whisper variant only.
- Three-way diverse trio: Whisper-large-v3 + wav2vec2-xlsr-hu + MMS-1b-fl102.
  Different architectures, partially different training corpora.

Agreement is HIGH-PRECISION but NOT 100% — common phonetic confusions in
Hungarian (ny/ni, j/ly, written vs spoken forms) can be shared across models.

Use the consensus as a **clean-data filter**, not a transcript fixer:
- HIGH (all 3 agree after normalization) → supervised training subset
- MEDIUM (2 of 3 agree) → semi-supervised or held-out
- LOW (all 3 differ) → audio-only for SSL pretraining

## Normalization for consensus

When comparing transcripts:
1. Lowercase
2. Strip punctuation (keep apostrophe and hyphen if they're inside words)
3. Collapse whitespace
4. Normalize Hungarian-specific: `ő` vs `ô`, `ű` vs `û` (rare but seen)
5. Number-words: normalize digit vs word forms? (open question — depends on
   the downstream ASR's tokenizer)

WER computed via `jiwer.wer(reference, hypothesis)` after normalization.

Threshold for HIGH: pairwise WER < 0.05 across all three.
