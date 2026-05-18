# Checklist — what's left to do

**Updated:** 2026-05-18

Mark items as you complete them. Add new ones as they come up. Keep this file
honest — don't leave stale entries.

## Done

- [x] Source survey (TED, ME, MEK, MTVA, NAVA, Tilos, YouTube CC-BY HU) — most rejected as NC/ND/no-license
- [x] Storage allocation on NVMe SSD `/home/cseti/datassd2/`
- [x] Project skeleton (STATUS.md, README.md, configs/sources.yaml, bin/)
- [x] HF auth (`/home/cseti/.hf_token`, user "Cseti")
- [x] Downloads:
  - [x] YODAS v1 hu000 (35 GB, 167,705 utts, 16kHz segmented)
  - [x] YODAS2 hu000 (93 GB, 1,300 unsegmented videos + JSON aligned text)
  - [x] LibriVox HU (1.1 GB, 99 mp3, 2 audiobooks)
  - [x] Podcasts HU (3.3 GB, 42 mp3, 12 items)
  - [x] MOSEL HU transcripts (1.2 GB, 2,314,161 utterances)
  - [x] VoxPopuli HU labeled HF (11 GB, 6 parquet, 20,306 utts)
  - [x] VoxPopuli HU unlabeled (297 GB, 17,297 EP session ogg, ~22,076h)
- [x] Manifest v3 built (`train_aligned.jsonl`, `train_unaligned.jsonl`, `train_pseudo_labeled.jsonl`)
- [x] Schema documentation (`notes/MANIFEST_SCHEMA.md`)
- [x] STATS.md established, CHECKLIST.md, CLAUDE.md created

## In progress

(none)

## Next — phase 2: segmentation & manifest refinement

- [ ] **VoxPopuli session segmentation** — split the 17,297 ~1h ogg sessions
      into utterance-level clips that MOSEL transcripts can reference by id.
  - Options:
    - (A) Use facebook/voxpopuli `get_unlabelled_data --subset hu` (needs git
      clone + pip install of the upstream repo)
    - (B) Custom segmenter from MOSEL TSV offsets (TSV lacks offset for
      voxpopuli rows; only ytc has it) — would need to derive offsets from
      VoxPopuli alignment metadata
  - Estimated output: ~2.1M per-utterance ogg clips (~80-200 GB depending on
      codec choice). Re-encoding to Opus 32 kbps would compress to ~50 GB.
- [ ] **YODAS2 segmentation verification** — investigate why JSON timestamps
      sum to 512h vs YODAS v1's 172h on the same video set. Sample 10 videos,
      compare timestamp ranges, decide whether to use v2 utterance partition
      or stick with v1.
- [ ] **Rebuild manifest** after segmentation — fill `audio_path` in
      `train_pseudo_labeled.jsonl` rows.

## Next — phase 3: quality scoring (GPU work)

- [ ] Tier-1 cheap stats (CPU, no GPU): `rms_dbfs`, `peak_dbfs`, `is_clipped`,
      `silence_ratio`. Single pass over all aligned + unaligned audio.
- [ ] Tier-2 neural MOS (GPU):
  - [ ] DNSMOS (general audio quality MOS predictor)
  - [ ] UTMOSv2 (TTS-specific MOS predictor)
  - [ ] Pyannote VAD `vad_speech_ratio`
  - [ ] Music detection (YAMNet / CREPE)
- [ ] Derived `quality_tier` (HIGH/MEDIUM/LOW) computed from above

## Next — phase 4: multi-ASR consensus (GPU work)

- [ ] Pick the 3-ASR trio:
  - [ ] Whisper-large-v3 (encoder-decoder)
  - [ ] wav2vec2-XLS-R or wav2vec2-large-xlsr-53-hungarian (CTC)
  - [ ] Meta MMS-1b-fl102 or Canary-1b-v2 (third independent arch)
- [ ] Transcribe all unaligned audio with all 3 ASRs
- [ ] Compute pairwise WER and ROVER-style consensus
- [ ] Fill `transcripts` dict and `text_consensus` / `confidence_level` fields

## Next — phase 5: training pipeline decision

Need to pick one (or sequence):
- [ ] **Path A — Canary-1b-v2 fine-tune.** Modern multilingual base, already
      knows Hungarian (Granary-trained). 230h labeled fine-tune. ~$50-150 compute.
- [ ] **Path B — Continual pretraining (CPT) on 17,700h SSL.** Then fine-tune.
      ~$250-400 compute. Likely best end-quality but bigger investment.
- [ ] **Path C — TTS-augmentation.** Build reference-conditioned synthetic
      diverse data with F5-TTS-hungarian or CosyVoice 2. See
      `notes/TTS_AUGMENTATION_IDEA.md`. ~$400-800 compute.
- [ ] **Path D — Whisper-large-v3 fine-tune.** Simpler pipeline, $100-200.

Order of operations the user prefers: ASR baseline first (Path A), measure
domain failures, then pick augmentation if needed (Path B or C).

## Backlog (optional, not blocking)

- [ ] Parliament HU (Magyar Orszaggyules) download via HLS + ffmpeg —
      grey-zone license, stored separately. Use `github.com/fodi/parlament-dl`
      as reference.
- [ ] Common Voice HU train partition (~50h crowdsourced) — could be added
      to the labeled fine-tune set despite the "validation only" memo, since
      it's CC0.
- [ ] Tier-3+ acoustic augmentations (noise/reverb/codec) for training.

## Open questions

- Investigate YODAS2 vs v1 utterance overlap — does v2 add real new audio
  coverage or just re-segments the same content?
- Do we need a held-out test split from our own data, or do CV-HU + FLEURS-HU
  suffice as evaluation?
- For TTS phase: F5-TTS-hungarian (community) vs CosyVoice 2 HU (Alibaba) —
  benchmark both on a small subset before committing.
