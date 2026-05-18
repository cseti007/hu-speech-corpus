# STATUS — Hungarian speech corpus for ASR/TTS LoRA

**Slug:** `hu-speech-corpus` (matches directory name)
**Started:** 2026-05-17
**Deadline:** TBD
**Last update:** 2026-05-17
**Phase:** scoping

## Goal

Assemble a training-grade Hungarian speech corpus (audio + transcripts) from free-license sources for ASR and TTS LoRA fine-tuning. Target volume: 10k+ hours after filtering. "Done" = clean manifest JSONLs (train/val splits) backed by audio on `/home/cseti/datassd2/hu-speech-corpus/`, ready to feed into Whisper / wav2vec2 / TTS training pipelines.

## Current state

- Source survey complete; primary sources identified and licensed.
- Storage path provisioned: `/home/cseti/datassd2/hu-speech-corpus/` (1.7 TB free on NVMe).
- Directory skeleton created (raw / processed / cache split).
- Nothing downloaded yet.

## Source priority

| # | Source | License | HU hours | Notes |
|---|---|---|---|---|
| 1 | YODAS `hu000` | CC-BY (YouTube filter) | 181.76 | Human captions, manual subset. ~12-15 GB FLAC. |
| 2 | LibriVox HU (archive.org) | Public domain | 18.8 (Egri csillagok + Janos vitez) | Only 2 books exist; smaller than initial estimate. Long-form, TTS-friendly. |
| 2b | Podcasts HU free-license (archive.org) | Mixed PD/CC0/CC-BY | 53 (Hetvegi Kotekedo 33ep + Szabad Europa 9ep) | Conversational + interview register. |
| 3 | MOSEL HU transcripts | CC-BY | 11,660 (Whisper-pseudo) | ~1 GB JSONL. Filter via `hall_*` flags. |
| 4 | MOSEL HU audio — VoxPopuli part | CC0 | ~4,400 | ~285 GB FLAC. From `facebook/voxpopuli`. |
| 5 | MOSEL HU audio — YouTubeCommons part | CC-BY | ~13,000 | ~150-200 GB Opus re-encode, ~800 GB FLAC. yt-dlp. |
| 6 | Filmhíradók 1931-43 | PD (likely, by age) | small | Optional, archaic register. |
| 7 | **Parliament HU (gray zone)** | **GRAY** — content PD per Szjt §1(4), recording 50yr neighboring right | varies | Secondary source. Use only if VoxPopuli register insufficient. Separate dir. |

Excluded as primary training data (validation/eval only): Common Voice HU (quality), FLEURS HU (12h benchmark), VoxPopuli HU transcribed labeled set (63h benchmark).

## Focus order

1. **ASR first** (Whisper / wav2vec2 fine-tuning) — maximize volume, accept Whisper-pseudo labels with filtering
2. **TTS second** (after ASR pipeline proven) — strict alignment, force-aligned LibriVox + cherry-picked clean cuts

## In progress

- [x] ~~`bin/download_yodas.py`~~ — DONE 2026-05-17. 167,705 utterances, 35 GB extracted.
- [x] ~~`bin/download_archive_org_hu.py`~~ — DONE 2026-05-17. 12 items / 141 files / 4.3 GB. Canonical audio picker, ~50% size savings vs naive all-variants.
- [x] ~~`bin/download_yodas2.py`~~ — DONE 2026-05-17. 32 tarballs, 39.76 GB → 93 GB extracted. 1,300 unsegmented WAV 24kHz + JSON time-aligned transcripts.
- [x] ~~`bin/download_voxpopuli_hu_hf.py`~~ — DONE 2026-05-17. 6 parquets, 10.78 GB, ~63h transcribed_data.
- [x] ~~`bin/download_voxpopuli_hu_unlabeled.py`~~ — DONE 2026-05-17 23:00. 24/24 tarballs (V1+V2), 297 GB, 17,297 EP session ogg files. **Actual measured: 22,076h** (paper said 17,700h).
- [x] ~~`bin/build_manifest.py` v3~~ — DONE 2026-05-18. 3 JSONL files:
  - `train_aligned.jsonl` 425 MB, 365,758 utterances / 684h (YODAS v1+v2 + VoxPopuli labeled)
  - `train_unaligned.jsonl` 20 MB, 17,438 files / 22,137h (LibriVox + podcasts + VoxPopuli unlabeled sessions)
  - `train_pseudo_labeled.jsonl` 3.3 GB, 2,314,161 utterances (MOSEL Whisper-pseudo; audio_path=null until segmentation; 34k halluc-flagged + 176k lid≠hu marked for filtering)
- [ ] Next: voxpopuli segmentation step (split 17,297 ~1h sessions into MOSEL-matched utterances)
- [x] ~~`bin/build_manifest.py` v2~~ — DONE 2026-05-17. Schema v2 unified JSONL written to `processed/manifests/`:
  - `train_aligned.jsonl`: **172.04h / 167,705 utterances** (YODAS, ASR-ready). 180 MB.
  - `train_unaligned.jsonl`: **61.01h / 141 files** (LibriVox + podcasts, audio-only). 145 KB.
  - `stats.json`: schema_version=2, per-source breakdown.
  - Schema doc: `notes/MANIFEST_SCHEMA.md` (multi-ASR mezők, quality tier evolution).
  - Per-item SPDX license mapping (CC-BY-4.0, CC0-1.0, PD, etc.). 
- [ ] Next: MOSEL HU transcripts download (~1 GB, FBK-MT/mosel HF repo, hu/ subset)

## Next steps (ASR phase)

- LibriVox HU scrape script (parallel with YODAS while validating pipeline)
- MOSEL transcript download + hallucination flag filtering script
- VoxPopuli HU audio download (first big chunk, ~285 GB)
- Manifest builder: normalize all sources to a common JSONL schema `{audio_path, text, duration, source, license, speaker_id?, confidence?}`
- Re-encode pipeline: FLAC → Opus 32 kbps mono for space optimization
- (Deferred to TTS phase) Force-alignment pipeline for LibriVox, cherry-pick clean speaker subsets

## Blockers

- None currently.

## Notes / decisions

- 2026-05-17: Storage allocated to `/home/cseti/datassd2/` (NVMe, ROTA=0). HDDs (`data1`, `data2`) avoided due to random-read penalty during training.
- 2026-05-17: Separating data (SSD) from code (cseti-os hub). Data path treated as ephemeral / re-downloadable; code + STATUS in the hub.
- 2026-05-17: MOSEL transcripts are Whisper-large-v3 pseudo-labels — usable but noisy. Strategy: (a) use as-is for pretraining-style supervised, (b) filter aggressively via `hall_*` flags, (c) optionally iterative self-distillation later.
- 2026-05-17: YODAS `hu000` is human captions (not Whisper) — treat as the higher-quality anchor among the larger sources.
- 2026-05-17: ASR-first focus. TTS-specific work (force-alignment, strict speaker curation) deferred to phase 2.
- 2026-05-17: HF auth via `/home/cseti/.hf_token` (verified, user `Cseti`). Scripts read this path; never commit the token itself.
- 2026-05-17: LibriVox HU survey: only **2 books** exist (Egri csillagok 17:28h, Janos vitez 1:18h). Original 50-150h estimate was wrong. Found additional CC-BY HU podcast (Hetvegi Kotekedo, 31:17h, Tilos Radio) on archive.org — added as `podcasts_hu_cc` source.
- 2026-05-17: License survey on user-suggested sources: **TED Talks** = CC BY-NC-ND 4.0 (rejected, NC+ND); **Mindentudas Egyeteme** = CC BY-NC-ND 2.5 HU (rejected, NC+ND); **NAVA / NFI Filmarchive** = streaming-only, no CC. **MEK default** = personal/non-commercial — usable only for explicitly CC-tagged items.
- 2026-05-17: Found **Szabad Europa Podcast** on archive.org — 9 episodes under PD/CC0/CC-BY licenses, ~22h Hungarian interview content. Added to podcasts_hu_cc.
- 2026-05-17: Surveyed user-suggested **public broadcaster (MTVA M1-M5, Duna, Kossuth, Petofi)** — all "all rights reserved", NO CC license. Rejected.
- 2026-05-17: **Parliament HU (Orszaggyules) decision**: included as GRAY-ZONE secondary source. Speech content is PD per Szjt §1(4) ("hivatalos kozlemeny"), but the recording itself has a 50-year neighboring right (Szjt §73). No explicit CC license granted by parlament.hu. User accepted the legal ambiguity for practical low-risk use; stored separately in `raw/parliament_hu_gray/` for traceability. VoxPopuli (CC0) preferred for parliamentary register.
