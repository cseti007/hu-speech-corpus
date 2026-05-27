# STATUS — Hungarian speech corpus for ASR/TTS LoRA

**Slug:** `hu-speech-corpus` (matches directory name)
**Started:** 2026-05-17
**Deadline:** TBD
**Last update:** 2026-05-26
**Phase:** 3 (quality scoring re-run on voxpopuli_resegmented) + 4 prep

## Goal

Assemble a training-grade Hungarian speech corpus (audio + transcripts) from
free-license sources for ASR and TTS LoRA fine-tuning. Target volume: 10k+
hours after filtering. "Done" = clean manifest JSONLs (train/val splits)
backed by audio under `$HU_CORPUS_ROOT`, ready to feed into ASR / TTS
training pipelines.

## Current state

- **Phase 1 (acquisition) DONE:** ~22.5k hours of free-license HU audio on
  SSD (CC0 / CC-BY / PD only). Primary sources: VoxPopuli HU labeled +
  unlabeled (22,139h), YODAS2 hu000 (~213h), Common Voice 25.0 HU (180h,
  NEW 2026-05-25), LibriVox HU (19h), podcasts CC (42h).
- **Phase 2 (segmentation) DONE:** VoxPopuli session segmentation
  (2026-05-19), YODAS2 caption merging (2026-05-20), VAD-chunking of
  long-form audio (LibriVox + podcasts, 2026-05-20), per-row duration
  extraction for VoxPopuli labeled (2026-05-20).
- **Phase 2.5 (normalization) DONE:** all clip-level rows now 3-30s.
- **Phase 2.6 (cleanup attempt + Plan B finding) DONE 2026-05-24:** big
  finding — VoxPopuli unlabelled is 30-second sliding windows, NOT
  forced alignment. 43.2% of MOSEL clips have mid-word cuts. Plan A
  (post-hoc boundary refinement) cannot fix it. Plan B (full Silero VAD
  re-segmentation on raw EP sessions) chosen.
- **Plan B Silero VAD re-segmentation DONE 2026-05-25:** 17,292 EP
  sessions → **4,283,766 chunks / 17,992.84h** in 17.4h wall-clock,
  0 errors. User A/B verified audibly better than old 30s windows.
  Output: `processed/voxpopuli_resegmented/` (297 GB). This is the new
  canonical EP audio layer.
- **Manifest v5 (lean schema) built 2026-05-25:** 4,359,744 rows / 18,290h.
  Drops `mosel_*` + `voxpopuli_unlabeled_gap`, adds `voxpopuli_resegmented`
  + `common_voice_25_0_hu`. `bin/build_manifest_v5.py`.
- **Common Voice 25.0 HU added 2026-05-25:** new source via Mozilla Data
  Collective API. 117,503 MP3 clips / 180.85h / CC0-1.0. Downloader:
  `bin/download_common_voice_hu.py`.
- **Corpus curator (web UI) DONE 2026-05-23:** Parquet + DuckDB + Flask
  stack. Sortable + paginated table, filters, inline audio playback,
  refined-audio dual playback for boundary inspection. Used for Phase
  2.6 spot-checks.

## In progress

**Phase 3 re-run on voxpopuli_resegmented** (since 2026-05-25):

| Step | Status | Runtime | Notes |
|---|---|---|---|
| Tier-1 (rms/peak/clip/silence) | **DONE** | ~30 min CPU | 4.28M new chunks scored |
| Tier-2 DNSMOS P.835 | **in progress** | ~24-26h CPU total, ~64% done | 8 workers, ~36 clips/s |
| LID v2 Pass 1 (whole-clip GPU) | **DONE** | 13.48h | 4.28M clips, 461,912 (10.78%) flagged for Pass 2 |
| LID v2 Pass 2 (windowed + VAD snap) | **in progress** | ETA ~14h | ~9.5 clips/s on 462k flagged. Computes language_regions per clip. NO trimming yet. |
| Merge sidecars → manifest_v5 | pending | ~1 min | after DNSMOS + LID v2 finish |
| Build manifest_v5 parquet | pending | ~3 min | for curator + downstream |
| Build multi-source PoC parquet | pending | ~5 min | ~280 clips with outliers per source |
| Restart curator | pending | — | `bash bin/curator/serve.sh multi` |

ETA for Phase 3 completion: ~24-26 hours from start (DNSMOS is the long pole).

## Next phases

**Phase 4 — multi-ASR consensus on voxpopuli_resegmented + transcribed sources:**
- Phase 4a-pre smoke test on Config 4.A (Canary v2 + Qwen FT + KenLM): DONE
  2026-05-20 (pairwise WER 10.46% on labeled parliament).
- Phase 4a PoC on 100h: DONE 2026-05-23. **Result: 29.64% pairwise WER,
  16.67% median, 3.3% exact match → 581h projected GOLD. Config 4.A
  insufficient.**
- Phase 4b decision: **Add Parakeet TDT v3 as 3rd pillar (Config 4.B).**
  Pre-Phase-4c: re-run the 100h PoC with Parakeet for 3-pillar exact-match
  measurement. After Plan B clean-audio baseline is established, also
  evaluate Phase 4d (4th pillar with text-prompt context).
- Phase 4c full consensus run: 10-15 days local on 1 GPU at current Parakeet
  speed; multi-GPU rental or pre-filter on quality_flags considered.

**Phase 5 — training pipeline:**
4 paths under consideration depending on Phase 4c GOLD volume:
- Path A: Canary 1B v2 fine-tune (Granary-trained, already knows HU).
- Path B: Continual pretraining on 17,993h SSL → fine-tune.
- Path C: TTS-augmentation (F5-TTS-hungarian or CosyVoice 2 HU).
- Path D: Whisper-large-v3 fine-tune.

User-preferred order: Path A baseline first, measure, then pick augmentation.

## Source priority (post-Plan-B)

| # | Source | License | HU hours | Status / notes |
|---|---|---|---|---|
| 1 | **voxpopuli_resegmented** (NEW canonical) | CC0 | 17,992.84 | Silero VAD chunks of EP raw sessions; replaces mosel layer |
| 2 | YODAS2 hu000 | CC-BY-3.0 | 182.30 | Human captions, post-merge |
| 3 | **common_voice_25_0_hu** (NEW) | CC0-1.0 | 180.85 | Read speech, use_for TBD |
| 4 | voxpopuli_hu_labeled | CC0-1.0 | 58.45 | Transcribed parquet |
| 5 | podcasts_hu_cc | mixed PD/CC0/CC-BY | 39.54 | Hetvegi Kotekedo + Szabad Europa |
| 6 | librivox_hu | PD | 17.19 | Egri csillagok + János Vitéz |
| 7 | Parliament HU (gray) | GRAY | varies | Secondary; not yet downloaded; VoxPopuli covers register |
| 8 | Filmhíradók 1931-43 | PD likely | small | Optional; archaic register / robustness |
| 9 | YouTube Commons HU (MOSEL part) | CC-BY | ~13,000 | NOT downloaded; mosel transcripts only |

Excluded as primary training data, evaluation only (per current policy;
re-evaluate the new Common Voice 25 designation in curator):
- FLEURS HU — 12h benchmark, eval set
- VoxPopuli HU transcribed labeled set (63h) — too small for training, eval scale

## Focus order

1. ✅ **Acquisition + license vetting** (Phase 1) — complete
2. ✅ **Segmentation + normalization** (Phase 2-2.5) — complete
3. ✅ **VAD re-segmentation Plan B** — complete 2026-05-25
4. 🟡 **Quality scoring on new layer** (Phase 3 re-run) — in progress
5. ⬜ **Multi-ASR consensus** (Phase 4) — Phase 4a PoC done, Config 4.B
     decided, 3-pillar PoC + full run pending
6. ⬜ **Training pipeline** (Phase 5) — design ready, depends on Phase 4c volume

## Blockers

- None currently. Phase 3 is CPU+GPU compute-bound, will finish naturally.

## Notes / decisions

- 2026-05-26: **Phase 3 re-run in progress** on voxpopuli_resegmented after
  Plan B finished. Tier-1 done. DNSMOS + LID v2 (2-pass) running. Merge into
  manifest_v5 deferred until both finish.
- 2026-05-25: **Plan B Silero VAD re-segmentation COMPLETE.** Manual user
  A/B confirmed audibly better than the broken old 30s windows. Most
  important pipeline step of the project.
- 2026-05-25: **Common Voice 25.0 HU added** via Mozilla Data Collective
  API. New downloader script. Designation TBD (older CV was eval-only).
- 2026-05-24: **VoxPopuli 30s windows finding.** unlabelled_v2.tsv.gz is
  sliding windows, not forced alignment. Plan A insufficient → Plan B
  chosen.
- 2026-05-23: **Phase 4a PoC 100h done.** 581h projected GOLD, Config 4.A
  insufficient. Decided to add Parakeet TDT v3 as 3rd pillar (Config 4.B).
- 2026-05-23: **Corpus curator (web UI) live.** Phase 2.6 spot-checks
  happen here. Used to validate the Plan B output.
- 2026-05-22: **Manifest unification v3 → v4.** Single `manifest.jsonl` +
  `manifest_sessions.jsonl`. (Later superseded by v5 with the Plan B
  layer 2026-05-25.)
- 2026-05-21: Tier-2 LID + VAD done on old v4 manifest (later re-run on
  resegmented layer). UTMOSv2 dropped (too slow per-clip).
- 2026-05-20: Phase 2.5 normalization done; all clip rows 3-30s.
- 2026-05-19: VoxPopuli session segmentation done (using the old broken
  alignment manifest — later superseded by Plan B).
- 2026-05-17 to 2026-05-18: Phase 1 acquisitions, source survey,
  Apache-2.0 license decisions, downloader scripts written.
