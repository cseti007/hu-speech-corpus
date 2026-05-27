# Hungarian speech corpus — current statistics

**Last updated:** 2026-05-26 (mid Phase 3 re-run on voxpopuli_resegmented; numbers reflect manifest_v5 with Common Voice 25.0 HU integrated)
**Last refresh source:** `processed/manifests/manifest_v5.jsonl` (rebuilt 2026-05-26 with CV25 ingestion) + `processed/normalization/voxpopuli_resegmented.jsonl`
**Local SSD usage:** ~1.04 TB (post Plan B VAD re-segmentation + Common Voice 25.0)

> [!] **Update this file whenever the corpus state changes.**

## TL;DR — How many hours of audio do we have?

**Canonical going forward: `manifest_v5.jsonl` (lean schema).** The Plan B
Silero VAD re-segmentation of the 22k h VoxPopuli HU unlabeled corpus
(DONE 2026-05-25, see `notes/JOURNEY.md`) replaces the structurally
broken 30-second window approach. The new `voxpopuli_resegmented` source
is the canonical EP-parliament audio layer.

| Bucket (v5 manifest) | Rows | Hours | Status |
|---|---:|---:|---|
| **`manifest_v5.jsonl` total** | **4,477,247** | **18,471h** | Phase 3 quality-merge target |
|   — voxpopuli_resegmented (NEW canonical EP layer) | 4,283,766 | 17,992.84 | Silero VAD chunks, replaces mosel layer |
|   — yodas2_hu000 (chunked 2026-05-26) | 47,817 | 182.30 | YouTube CC-BY captions, OGG 16 kHz mono chunks |
|   — common_voice_25_0_hu (NEW, integrated 2026-05-26) | 117,503 | 180.85 | CC0-1.0, read speech, use_for TBD |
|   — voxpopuli_hu_labeled | 19,051 | 58.45 | parquet, transcribed |
|   — librivox_hu (chunks) | 3,581 | 17.19 | VAD-chunked from 99 chapters |
|   — podcasts_hu_cc (chunks) | 5,529 | 39.54 | VAD-chunked from 42 episodes |
| **`manifest_sessions.jsonl`** (long-form parents, SSL context) | 17,438 | 22,137h | Overlaps chunks above — don't sum |

**One-line answer to "how big is it now":**
**~22.5k hours of free-license Hungarian audio acquired. The canonical
v5 manifest has 4.48M training-ready rows / 18,471h, now including Common
Voice 25.0 HU (180.85h, integrated 2026-05-26). Phase 3 quality re-scoring
(DNSMOS + LID v2 Pass 1) on the voxpopuli_resegmented layer is in progress;
merge into manifest_v5 will follow.**

> **Plan B re-segmentation milestone 2026-05-25:** Silero VAD on the
> full 22k h raw VoxPopuli HU unlabeled corpus completed in 17.4 h
> wall-clock, 0 errors across 17,292 sessions. Output: 4,283,766
> per-utterance OGG chunks (297 GB) at `processed/voxpopuli_resegmented/`.
> Mean 15.12s, median 13.30s. User A/B verified audibly better than the
> old Facebook 30-second sliding windows ("határozottan a mostani VAD
> jobb"). The new layer replaces both the old mosel layer (2.31M / 17,762h
> Whisper-pseudo) and the voxpopuli_unlabeled_gap layer (820k / 1,555h)
> in `manifest_v5`. Snapshot:
> `notes/stats_snapshots/2026-05-25__vad_resegmentation_done.md`.

> **Common Voice 25.0 HU added 2026-05-25:** new source via the Mozilla
> Data Collective API (dataset_id `cmn2g9aoi01fyo107xhdrwb5d`). 117,503
> MP3 clips / 180.85h / CC0-1.0. Released 2026-03-09. Splits: train 88.52h,
> dev 19.05h, test 20.33h, validated 129.12h (superset). The older
> `common_voice_hu` placeholder in `configs/sources.yaml` is now
> superseded by `common_voice_25_0_hu`. Use_for designation (training
> vs eval) pending curator review.

> **Manifest v4 → v5 migration 2026-05-25:** lean schema. Drops
> `mosel_hu_voxpopuli`, `mosel_hu_ytc`, and `voxpopuli_unlabeled_gap`
> (all superseded by `voxpopuli_resegmented`). Adds a derived
> `source_url` field. Per-row schema is otherwise compatible.
> `bin/build_manifest_v5.py` is the builder.

## Breakdown by source

### `manifest_v5.jsonl` — current canonical training-ready rows

| Source | Rows | Hours | Audio format | License | Notes |
|---|---:|---:|---|---|---|
| **voxpopuli_resegmented** | **4,283,766** | **17,992.84** | OGG/Vorbis 16 kHz mono | CC0-1.0 | Silero VAD chunks of raw EP sessions, 200 ms ambient padding. Mean 15.12s, median 13.30s. Replaces old mosel + voxpopuli_gap layers. No transcripts yet (Phase 4 ASR consensus pending). |
| yodas2_hu000 | 47,817 | 182.30 | OGG/Vorbis 16 kHz mono (chunked 2026-05-26) | CC-BY-3.0 | Human captions, merged 3-30s. From 177k raw captions via `bin/normalize_yodas2.py`. Parent WAVs sliced into standalone OGG chunks by `bin/chunk_yodas2.py` (3.2 GB at `processed/chunks/yodas2_hu000/`). `transcripts.source_caption` populated. |
| common_voice_25_0_hu | 117,503 | 180.85 | MP3 48 kHz mono | CC0-1.0 | Read speech, integrated 2026-05-26. `transcripts.source_caption` populated (read sentences). `quality_flags.cv25_status` ∈ {validated, invalidated, other}; `quality_flags.cv25_split` ∈ {train, dev, test, null}. Use_for TBD pending curator review. |
| voxpopuli_hu_labeled | 19,051 | 58.45 | parquet-internal WAV | CC0-1.0 | Per-row durations extracted from parquet; 1,127 dropped < 3s, 136 dropped > 30s. `transcripts.source_caption` populated. |
| librivox_hu (chunks) | 3,581 | 17.19 | Ogg/Vorbis 16 kHz mono | PD | Silero VAD-chunked from 99 audiobook chapters (91.5% retention). |
| podcasts_hu_cc (chunks) | 5,529 | 39.54 | Ogg/Vorbis 16 kHz mono | CC-BY (mixed) | Silero VAD-chunked from 42 podcast episodes (95.4% retention). |
| **Total** | **4,477,247** | **18,471.17** | | | |

### `common_voice_25_0_hu` — split / status detail

| Split | Clips | Hours |
|---|---:|---:|
| train | 59,395 | 88.52 |
| dev | 12,944 | 19.05 |
| test | 12,989 | 20.33 |
| **validated** (superset of train+dev+test) | **86,128** | **129.12** |
| other (insufficient votes) | 27,286 | 45.27 |
| invalidated (failed crowd vote) | 4,089 | 6.45 |
| **All** | **117,503** | **180.85** |

Layout: `raw/common_voice_25_0_hu/cv-corpus-25.0-2026-03-09/hu/clips/*.mp3` +
TSV split files. Per-clip durations from `clip_durations.tsv` (header +
117,503 entries, total 651,057 s = 180.85h). Source URL:
https://mozilladatacollective.com/datasets/cmn2g9aoi01fyo107xhdrwb5d

### Session-level long-form (in `manifest_sessions.jsonl`)

| Source | Files | Hours | Audio format | License | Notes |
|---|---:|---:|---|---|---|
| librivox_hu | 99 | 18.79 | MP3 44.1 kHz | PD | Egri csillagok + János Vitéz |
| podcasts_hu_cc | 42 | 42.22 | MP3 (mostly) + 1 FLAC | mixed CC-BY/CC0/PD | Hetvegi Kotekedo 33 ep + Szabad Europa 9 ep |
| voxpopuli_hu_unlabeled | 17,297 | 22,076.33 | Ogg Vorbis 16 kHz mono | CC0-1.0 | EP sessions 2009-2020 (V1+V2). Parent of voxpopuli_resegmented chunks. |
| **Subtotal** | **17,438** | **22,137.34** | | | |

### Legacy / superseded (kept for audit, will be deleted later)

These layers stay on disk until Phase 4 confirms the new `voxpopuli_resegmented`
layer is meaningfully better via consensus WER analysis. After that they're
deletable (~316 GB recoverable). Not part of `manifest_v5`.

| Layer | Rows / Files | Hours | Status |
|---|---:|---:|---|
| mosel_hu_voxpopuli (Whisper-pseudo on 30s windows) | 2,312,817 | 17,762.97 | Superseded by voxpopuli_resegmented; 43.2% mid-word cut rate measured. |
| voxpopuli_unlabeled_gap (VAD chunks of MOSEL gaps) | 820,030 | 1,555.34 | Superseded by voxpopuli_resegmented (full VAD on raw sessions). |
| mosel_refined (Phase 2.6 boundary refinement) | ~13,000 PoC | — | Mostly no-op because the 0s gap was structural; superseded. |

## Disk usage breakdown

| Path | Size | Notes |
|---|---:|---|
| raw/yodas2_hu000/ | 93 GB | 24 kHz unsegmented WAVs (v1 dir deleted 2026-05-20) |
| raw/librivox_hu/ | 1.1 GB | MP3 audiobooks |
| raw/podcasts_hu/ | 3.3 GB | MP3 + 1 FLAC podcasts |
| raw/mosel_hu/transcripts/ | 1.2 GB | TSV files (still needed for cross-reference; eventually droppable) |
| raw/voxpopuli_hu_labeled/ | 11 GB | 6 parquet files |
| raw/voxpopuli_hu_unlabeled/raw_audios/ | 297 GB | 17,297 long-form Ogg sessions (parent of resegmented chunks) |
| raw/voxpopuli_hu_unlabeled/unlabelled_data/ | 266 GB | **LEGACY** 2.3M per-utterance Ogg clips on 30s windows (deletable after Phase 4) |
| processed/voxpopuli_resegmented/ | **297 GB** | **NEW canonical** 4.28M Silero VAD chunks |
| processed/chunks/yodas2_hu000/ | **3.2 GB** | **NEW (2026-05-26)** 47,817 OGG chunks (yodas2 chunking) |
| processed/normalization/mosel_refined/ | ~50 GB | **LEGACY** Phase 2.6 attempt (deletable) |
| raw/common_voice_25_0_hu/ | 7.7 GB | NEW: tarball 3.58 GB + extracted clips |
| processed/manifests/ | ~6 GB | `manifest.jsonl` (v4) + `manifest_v5.jsonl` + sidecars + `_legacy_v3/` archive |
| processed/quality/ | ~2 GB | Tier-1 + Tier-2 (DNSMOS + VAD + LID) sidecars |

Free on SSD: ~700 GB. After Phase 4 confirms voxpopuli_resegmented quality,
deleting the old mosel + voxpopuli_gap + mosel_refined layers frees ~316 GB.

## Domain breakdown (audio hours by source-type)

| Domain | Hours | % of total acquired |
|---|---:|---:|
| Parliament (EP, VoxPopuli) | 22,139h | 97.6% |
| Crowdsourced read speech (Common Voice 25.0) | 181h | 0.8% |
| YouTube (YODAS2) | ~213h | 0.9% |
| Audiobook (LibriVox) | 19h | <0.1% |
| Podcast (Hetvegi + Szabad Europa) | 42h | <0.2% |

**Domain bias risk:** the corpus is overwhelmingly parliamentary register
(~97.6%). Adding Common Voice 25 contributes 181h of crowdsourced read
speech which slightly increases register diversity but doesn't materially
shift the balance. Mitigation strategies under investigation: TTS-augmentation,
continued pretraining vs from-scratch trade-offs, and domain-balanced sampling.

## License breakdown

| License | Hours | Notes |
|---|---:|---|
| CC0-1.0 | 22,319h | VoxPopuli HU (labeled + unlabeled = 22,139h) + Common Voice 25 (180h) |
| CC-BY-3.0 | ~213h | YODAS2 (YouTube CC-BY filter, post-dedup + outlier filter) |
| CC-BY-4.0 | 31h | Hetvegi Kotekedo + Szabad Europa s02e04 + MOSEL transcripts |
| CC0-1.0 (in podcasts) | 3h | Szabad Europa s02e03 |
| PD | 27h | LibriVox HU (19h) + Szabad Europa s01e01-s02e01 (~8h) |
| **All free for training & redistribution** | **~22,593h** | No NC/ND in the corpus |

## Quality flags (Phase 3 in progress on voxpopuli_resegmented)

### Already populated on legacy v4 manifest
- `too_short` (< 1s): 9,347 (yodas2)
- `too_long` (≥ 30s): 124 (yodas2)
- `any_hallucination_flag` (MOSEL): 34,478
- `lid_is_hu` / `lid_not_hu` (MOSEL, w hole-clip): 176,523 non-Hungarian flagged
- `rms_dbfs`, `peak_dbfs`, `is_clipped`, `silence_ratio` (Tier-1): 3.2M rows
- `predicted_mos_dnsmos` (Tier-2): 3,189,774 rows (83% training-grade OVRL≥3.0)
- `vad_speech_ratio` (Tier-2): 3.19M rows (93.7% > 0.7 speech)

### Phase 3 re-run on voxpopuli_resegmented (in progress 2026-05-26)
- **Tier-1** (rms/peak/clip/silence): **DONE** on 4.28M new chunks
- **DNSMOS** (P.835 SIG/BAK/OVRL): **in progress** ~64% (CPU, ~36 clips/s, ETA ~10h remaining)
- **LID v2** (2-pass: whole-clip + windowed + VAD silence snap):
  - **Pass 1 DONE** 2026-05-26: 4,283,766 clips, 13.48h, 461,912 (10.78%) flagged for Pass 2
  - **Pass 2 in progress** on the 462k flagged clips, ~9.5 clips/s, ETA ~14h. Computes
    `language_regions`, `first_hu_start_sec`, `last_hu_end_sec`, `foreign_duration_sec`.
    **No trimming applied yet** — only metadata for curator review.

## Open investigations

1. ~~**YODAS2 = 512h vs YODAS v1 = 172h discrepancy.**~~ **RESOLVED 2026-05-18.**
   Root cause: 3 v2 videos with corrupt JSON timestamps. Real v2 transcribed coverage: ~204h.
2. ~~**VoxPopuli unlabeled = 22,076h vs paper's 17,700h.**~~ V1+V2 not being
   strict subsets; treat the measured number as truth.
3. ~~**MOSEL boundary defects (43.2% mid-word cuts).**~~ **RESOLVED 2026-05-25.**
   Root cause: VoxPopuli unlabelled_v2.tsv.gz is 30-second sliding windows
   with 0s gaps, NOT forced alignment. Plan B (Silero VAD on raw sessions)
   replaced the old layer. See JOURNEY entry "VoxPopuli unlabelled is 30-second
   windows" + "Silero VAD re-segmentation complete".
4. **Common Voice 25.0 HU use_for designation.** Older CV versions were
   marked eval_only due to inconsistent quality. The new v25 is 180h CC0
   read speech — worth re-evaluating after curator spot-check whether it
   should go into training (low-noise read speech could complement the
   parliamentary register) or stay eval-only.

## How to refresh this file

```bash
# From the repository root:
# 1. Rebuild manifest_v5 (after VAD re-segmentation; lean schema)
python -u bin/build_manifest_v5.py

# 2. Merge quality sidecars into manifest_v5 (after Phase 3 finishes)
python -u bin/merge_quality_into_manifest.py \
    --input processed/manifests/manifest_v5.jsonl

# Then update the tables above from processed/manifests/stats_v5.json
# Bump the "Last updated" date at the top of this file
```

The frozen v4 originals live in `processed/manifests/_legacy_v3/` (despite the
name, this was the v3 → v4 archive; will rename later).
