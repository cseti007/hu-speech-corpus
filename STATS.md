# Hungarian speech corpus — current statistics

**Last updated:** 2026-05-18 09:16
**Last refresh source:** `processed/manifests/stats.json` (v3 schema)
**SSD usage:** 440 GB / 1.8 TB (26%) on `/home/cseti/datassd2/`

> [!] **Update this file whenever the corpus state changes.** See `CLAUDE.md` rule 1.

## TL;DR — How many hours of audio do we have?

| Bucket | Hours | Status |
|---|---:|---|
| **Aligned** (audio + transcript ready) | **685h** | Trainable as-is |
| **Unaligned** (audio only, no transcript) | **22,137h** | SSL pretraining ready / TTS-side use |
| **Pseudo-labeled** (transcript ready, audio pending segmentation) | (covers ~17,700h once segmented) | Awaiting segmentation step |
| **Total unique audio acquired** | **~22,820h** | (after dedup of overlapping sources) |

**One-line answer to "how big is it now":**
**22.8k hours of free-license Hungarian audio, of which ~685h is segmented-and-transcribed and ready to train.**

## Breakdown by source

### Aligned (`train_aligned.jsonl`, 425 MB, 365,758 rows)

| Source | Utterances | Hours | Audio format | License | Notes |
|---|---:|---:|---|---|---|
| yodas_hu000 | 167,705 | 172.04 | WAV 16 kHz mono | CC-BY-3.0 | YouTube manual captions, pre-segmented |
| yodas2_hu000 | 177,747 | 512.50 | WAV 24 kHz mono | CC-BY-3.0 | Same videos as v1 but unsegmented + JSON aligned — **investigate why 3x the duration of v1** |
| voxpopuli_hu_labeled | 20,306 | (parquet-internal) | parquet rows | CC0-1.0 | Audio embedded in 6 parquet files; ~63h labeled per paper |
| **Total** | **365,758** | **684.55** | | | |

### Unaligned (`train_unaligned.jsonl`, 20 MB, 17,438 rows)

| Source | Files | Hours | Audio format | License | Notes |
|---|---:|---:|---|---|---|
| librivox_hu | 99 | 18.79 | MP3 44.1 kHz | PD | Egri csillagok + János Vitéz |
| podcasts_hu_cc | 42 | 42.22 | MP3 (mostly) + 1 FLAC | mixed CC-BY/CC0/PD | Hetvegi Kotekedo 33 ep + Szabad Europa 9 ep |
| voxpopuli_hu_unlabeled | 17,297 | 22,076.33 | Ogg Vorbis 16 kHz mono | CC0-1.0 | EP sessions 2009-2020 (V1+V2). **Measured > paper's 17,700h estimate** |
| **Total** | **17,438** | **22,137.34** | | | |

### Pseudo-labeled (`train_pseudo_labeled.jsonl`, 3.3 GB, 2,314,161 rows)

Audio is pending segmentation. Counts here describe transcript availability;
hours come online after segmentation pulls audio from VoxPopuli unlabeled sessions.

| Source | Utterances | Halluc-flagged | lid≠hu | License | Notes |
|---|---:|---:|---:|---|---|
| mosel_hu_voxpopuli | 2,312,817 | 34,478 (1.5%) | 176,523 (7.6%) | CC-BY-4.0 | Whisper-large-v3 pseudo-labels for VoxPopuli HU sessions |
| mosel_hu_ytc | 1,344 | 0 | 0 | CC-BY-4.0 | YouTube Commons HU (5.95h ready, has offset+duration) |
| **Total** | **2,314,161** | **34,478** | **176,523** | | After lid+halluc filter: ~2.1M clean utterances |

## Disk usage breakdown (440 GB total)

| Path | Size | Notes |
|---|---:|---|
| raw/yodas_hu000/ | 35 GB | 16 kHz segmented WAVs |
| raw/yodas2_hu000/ | 93 GB | 24 kHz unsegmented WAVs |
| raw/librivox_hu/ | 1.1 GB | MP3 audiobooks |
| raw/podcasts_hu/ | 3.3 GB | MP3 + 1 FLAC podcasts |
| raw/mosel_hu/transcripts/ | 1.2 GB | TSV files |
| raw/voxpopuli_hu_labeled/ | 11 GB | 6 parquet files |
| raw/voxpopuli_hu_unlabeled/ | 297 GB | 17,297 Ogg sessions |
| processed/manifests/ | 3.7 GB | 3 JSONL + stats.json |

## Domain breakdown (audio hours by source-type)

| Domain | Hours | % of total |
|---|---:|---:|
| Parliament (EP, VoxPopuli) | 22,076h | 97% |
| YouTube (YODAS v1+v2 — overlap) | ~685h | 3% |
| Audiobook (LibriVox) | 19h | <0.1% |
| Podcast (Hetvegi + Szabad Europa) | 42h | <0.2% |

**Domain bias risk:** the corpus is overwhelmingly parliamentary register.
Mitigation strategies (TTS-augmentation, CPT vs from-scratch, etc.) discussed
in `notes/TTS_AUGMENTATION_IDEA.md`.

## License breakdown

| License | Hours | Notes |
|---|---:|---|
| CC0-1.0 | 22,139h | VoxPopuli HU (labeled + unlabeled) |
| CC-BY-3.0 | 685h | YODAS v1 + YODAS2 (YouTube CC-BY filter) |
| CC-BY-4.0 | 31h | Hetvegi Kotekedo + Szabad Europa s02e04 + MOSEL transcripts |
| CC0-1.0 (in podcasts) | 3h | Szabad Europa s02e03 |
| PD | 27h | LibriVox HU (19h) + Szabad Europa s01e01-s02e01 (~8h) |
| **All free for training & redistribution** | **~22,820h** | No NC/ND in the corpus |

## Quality flags (will grow)

Currently populated:
- `too_short` (duration < 1s, only on segmented rows): 17,991 in aligned
- `too_long` (duration ≥ 30s, only on segmented rows): 132 in aligned
- `any_hallucination_flag` (MOSEL): 34,478
- `lid_is_hu` / `lid_not_hu` (MOSEL): 176,523 non-Hungarian to filter

Pending (Tier 2 GPU work):
- `predicted_mos_dnsmos`, `predicted_mos_utmos`
- `vad_speech_ratio`
- `music_likelihood`
- `rms_dbfs`, `peak_dbfs`, `is_clipped`, `silence_ratio`

## Open investigations

1. **YODAS2 = 512h vs YODAS v1 = 172h discrepancy.** Same source videos.
   Hypothesis: overlapping subtitle timestamps in v2 JSON, or v2 includes more
   captions than v1's manual-only subset. Sample 10 videos and validate before
   trusting the 512h figure.
2. **VoxPopuli unlabeled = 22,076h vs paper's 17,700h.** Likely V1+V2 not being
   strict subsets — V2 added new sessions per-year. Treat as the true measured
   number; the paper figure is outdated.
3. **MOSEL hu pseudo-labels covering ~2.1M utterances after filter** — needs
   the voxpopuli segmentation step to materialize audio_path.

## How to refresh this file

```bash
cd /home/cseti/data2/Develop/Github-cseti/cseti-os/projects/hu-speech-corpus
/media/cseti/datassd/conda/miniconda3/bin/python -u bin/build_manifest.py
# Then update the tables above from processed/manifests/stats.json
# Bump the "Last updated" date at the top of this file
```
