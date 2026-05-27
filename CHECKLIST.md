# Checklist — what's left to do

**Updated:** 2026-05-26 (mid Phase 3 re-run on voxpopuli_resegmented; Plan B VAD re-segmentation DONE; manifest_v5 built; Common Voice 25.0 HU added; DNSMOS + LID v2 Pass 2 in progress)

Mark items as you complete them. Add new ones as they come up. Keep this file
honest — don't leave stale entries.

## Done

- [x] Source survey (TED, ME, MEK, MTVA, NAVA, Tilos, YouTube CC-BY HU) — most rejected as NC/ND/no-license
- [x] Storage allocation on NVMe SSD `/home/cseti/datassd2/`
- [x] Project skeleton (STATUS.md, README.md, configs/sources.yaml, bin/)
- [x] HF auth (`/home/cseti/.hf_token`, user "Cseti")
- [x] Downloads:
  - [x] ~~YODAS v1 hu000~~ (35 GB, 167,705 utts, 16kHz segmented) — **DELETED 2026-05-20**, superseded by v2
  - [x] YODAS2 hu000 (93 GB, 1,300 unsegmented videos + JSON aligned text)
  - [x] LibriVox HU (1.1 GB, 99 mp3, 2 audiobooks)
  - [x] Podcasts HU (3.3 GB, 42 mp3, 12 items)
  - [x] MOSEL HU transcripts (1.2 GB, 2,314,161 utterances)
  - [x] VoxPopuli HU labeled HF (11 GB, 6 parquet, 20,306 utts)
  - [x] VoxPopuli HU unlabeled (297 GB, 17,297 EP session ogg, ~22,076h)
- [x] Manifest v3 built (`train_transcribed.jsonl`, `train_untranscribed.jsonl`, `train_pseudo_transcribed.jsonl`) — **superseded by v4 unification 2026-05-22**
- [x] Schema documentation (`notes/MANIFEST_SCHEMA.md`)
- [x] STATS.md established, CHECKLIST.md, CLAUDE.md created
- [x] **Manifest unification v3 -> v4** — DONE 2026-05-22. Collapsed 4 manifest
      files (transcribed + pseudo + chunks + sessions, plus `_with_quality`
      variants) into 2: `manifest.jsonl` (3,210,169 training-ready rows /
      19,615.79h, quality_flags inline) + `manifest_sessions.jsonl` (17,438
      long-form parent rows). v4 `stats.json` with `manifest.{total,by_source}`
      + `sessions.{total,by_source}`. Migration script `bin/unify_manifests.py`,
      consumer scripts (quality_tier1/2, merge_quality_into_manifest,
      validate_manifest_quality, asr_consensus_poc_100h) and producer
      (`build_manifest.py`) all updated. Legacy originals archived under
      `processed/manifests/_legacy_v3/`. See `notes/JOURNEY.md` entry for
      rationale.

## In progress

- [x] **VoxPopuli full re-segmentation (Silero VAD) — DONE 2026-05-25.**
      `bin/resegment_voxpopuli_silero.py` over 17,292 raw EP sessions,
      8 workers, 17.4 hours wall-clock, 0 errors. Output: **4,283,766
      chunks under `processed/voxpopuli_resegmented/` (297 GB) +
      sidecar `processed/normalization/voxpopuli_resegmented.jsonl`.
      Total 17,992.84h, mean 15.12s, median 13.30s. 200ms ambient
      padding both sides. User A/B verified audibly better than old
      30s windows ("határozottan a mostani VAD jobb").**
      Snapshot: `notes/stats_snapshots/2026-05-25__vad_resegmentation_done.md`.

- [~] **Phase 3 re-run on voxpopuli_resegmented** — in progress 2026-05-26.
      Order: (1) Tier-1, (2) Tier-2 DNSMOS, (3) LID v2 (2-pass), (4) merge +
      manifest_v5 + parquet + curator. Steps 1-3 partially overlap (CPU +
      GPU parallel).
  - [x] **Tier-1** (rms/peak/clip/silence) — DONE on 4.28M new chunks
        (~30 min CPU).
  - [~] **Tier-2 DNSMOS** — in progress (CPU, 8 workers, ~36 clips/s,
        ETA ~24-26h total wall-clock from start, currently ~64% done).
  - [~] **LID v2 (2-pass design)** — in progress.
    - [x] Pass 1 (GPU, whole-clip ECAPA-TDNN on first 10s) DONE
          2026-05-26: 4,283,766 clips processed, 13.48h wall-clock at
          ~89 clips/s, 0 errors. **461,912 clips (10.78%) flagged
          for Pass 2** (HU-prob below 0.85 threshold).
    - [~] Pass 2 (windowed LID + Silero VAD silence snap, computes
          precise transition points + language_regions) — in progress
          on the 462k flagged clips, ~9.5 clips/s, ETA ~14h. Computes
          `language_regions`, `first_hu_start_sec`, `last_hu_end_sec`,
          `foreign_duration_sec` per clip. **NO TRIMMING applied yet
          — only metadata; user will spot-check in curator before
          deciding on trim policy.**
  - [x] **manifest_v5** — first build DONE 2026-05-25 (lean schema, drops
        `mosel_*` + `voxpopuli_unlabeled_gap`, adds `voxpopuli_resegmented`
        + `common_voice_25_0_hu`, adds `source_url` derived field).
        **4,359,744 rows / 18,290h.** Quality merge will re-run after
        Phase 3 completes.
  - [ ] Merge new Tier-1 + DNSMOS + LID v2 sidecars into manifest_v5
        (`bin/merge_quality_into_manifest.py --input manifest_v5.jsonl`).
  - [ ] Build manifest_v5 parquet (`bin/build_manifest_parquet.py`).
  - [ ] Build multi-source PoC parquet for curator
        (`bin/build_poc_multisource.py`, ~280 multi-source clips with
        outliers per source).
  - [ ] Restart curator: `bash bin/curator/serve.sh multi`.

- [x] **Common Voice 25.0 HU added — DONE 2026-05-25.**
      Source: Mozilla Data Collective (`mozilladatacollective.com`),
      dataset_id `cmn2g9aoi01fyo107xhdrwb5d`. New downloader
      `bin/download_common_voice_hu.py` (Bearer token auth, presigned
      Cloudflare R2 URL, Range-resume, automatic extract). Result:
      **117,503 MP3 clips / 180.85h / CC0-1.0** under
      `raw/common_voice_25_0_hu/cv-corpus-25.0-2026-03-09/hu/`.
      Splits: train 88.52h, dev 19.05h, test 20.33h, validated 129.12h
      (superset), other 45.27h, invalidated 6.45h. **Use_for designation
      (training vs eval) TBD** — older CV versions were marked
      eval_only due to inconsistent quality; re-evaluate in curator.

- [~] **Phase 2.6 cleanup attempted — Plan A (post-hoc) insufficient** —
      boundary refinement + language-purity audit completed 2026-05-24 on
      the 13,052-clip PoC sample. Tools: `bin/refine_mosel_boundaries.py`
      (12,985 / 13,052 ran without error, but most have `change_end_ms=0`
      because the structural finding below blocks rightwards extension) +
      `bin/audit_clip_language_purity.py` (with bug fix: 53.72% foreign-
      prefix, 9.76% whole-non-HU, 54.01% any-foreign-content).
      **Structural finding via clip 195 spot-check:** VoxPopuli
      unlabelled_v2.tsv.gz is 30-second sliding windows with 0s gaps,
      NOT forced alignment. Post-hoc refinement cannot fix the mid-word
      cuts. Full re-alignment (Plan B: WhisperX / MMS over 22k raw
      sessions) needed before Phase 4c. See
      `notes/JOURNEY.md` "VoxPopuli unlabelled is 30-second windows"
      entry + `notes/stats_snapshots/2026-05-24__voxpopuli_30s_window_finding.md`.
- [x] **Corpus curator (web UI for manifest exploration)** — DONE
      2026-05-23, validated 2026-05-24. Stack: Parquet + DuckDB +
      Flask + vanilla HTML/CSS/JS. Read-only browsing, column filters
      (source, has_text, lid, duration / dnsmos / vad ranges, clipped,
      hallucination, free-text search on transcripts + utterance_id),
      sortable + paginated table, inline audio playback (`preload="none"`).
      Files: `bin/build_manifest_parquet.py` + `bin/build_poc_parquet.py`
      (JSONL -> Parquet conversion, flattens nested dicts + keeps JSON
      blobs for fidelity), `bin/curator/{app.py, templates/index.html,
      static/{style.css, curator.js}, serve.sh}`. Launch:
      `bash bin/curator/serve.sh poc` ->
      `http://localhost:8002`. Phase 2.6 spot-check happens here.
      (heavy GPU+CPU); afterwards: `pip install flask duckdb pyarrow` ->
      `python bin/build_manifest_parquet.py` (~5 min) -> launch -> verify
      browser. Curation marks (drop/keep/flag) deferred to a later milestone.

## Next — phase 2: segmentation & manifest refinement

- [x] **VoxPopuli session segmentation** — DONE 2026-05-19. Used the official
      `facebookresearch/voxpopuli` `get_unlabelled_data` script's alignment
      manifest (`unlabelled_v2.tsv.gz`) but with a custom streaming `soundfile`
      backend to keep per-worker RAM constant. `bin/segment_voxpopuli.py` +
      `bin/setup_env.sh` (dedicated `hu-speech-corpus` conda env, `external/voxpopuli`
      clone with `download_url` shim). 2,312,817 per-utterance Ogg Vorbis clips
      written, 266 GB, 2h38m runtime at n_workers=16. Idempotent (sentinel +
      per-clip skip).
- [x] **YODAS2 segmentation verification** — DONE 2026-05-18. Investigated via
      `bin/investigate_yodas2.py`. Result: 3 corrupt-timestamp videos inflated
      v2 seg-sum from ~204h to 504h; v1 ⊂ v2 at video level. Decision: use v2
      as primary, drop v1, add outlier filter to manifest builder. Full
      write-up in `notes/YODAS_v1_v2_analysis.md`.
- [x] **Rebuild manifest** after segmentation — DONE 2026-05-19. `audio_path`,
      `segment_start_sec`, `segment_end_sec`, `duration_sec` populated for all
      2,312,817 voxpopuli MOSEL rows from `unlabelled_v2.tsv.gz`. YODAS2
      outlier filter + v1-drop also applied (already done 2026-05-18).
      Result: `pseudo_transcribed` total **17,762.97h** (was 5.95h before
      segmentation).

## Next — phase 2.5: audio normalization (scheduled, not yet running)

**Dependency:** every clip-level Phase 3 / Phase 4 task runs on the normalized
manifest. Detailed design in `notes/AUDIO_NORMALIZATION.md`.

Pre-execution scripts:
- [x] `bin/normalize_yodas2.py` — DONE 2026-05-20. 177,726 captions → **47,817
      merged 3-30s clips, 182.30h**. 3 outlier captions dropped, 14,221 dropped
      after merge (<3s). Sidecar: `processed/normalization/yodas2_merged.jsonl`.
- [x] `bin/extract_voxpopuli_labeled_durations.py` — DONE 2026-05-20. 20,306
      rows, **60.44h** total. 5.55% <3s (1,127), 0.67% >30s (136). Sidecar:
      `processed/normalization/voxpopuli_labeled_durations.jsonl`.
- [x] `bin/chunk_longform.py` (single script handling librivox + podcasts) —
      DONE 2026-05-20. Silero VAD based, 3-30s chunks at ~87% speech retention.
      Status: librivox + podcasts full run in progress at session creation.
- [x] `bin/chunk_voxpopuli_unlabeled_gap.py` — DONE 2026-05-20 (script written
      and verified on 3 sessions: 64 chunks, idempotent via per-session
      sentinels). Full run scheduled (~10-15 hours CPU/GPU).
- [x] Update `build_manifest.py` to load all normalization sidecars — DONE
      2026-05-20. YODAS2 reads from `yodas2_merged.jsonl`, VoxPopuli labeled
      gets durations from sidecar + filters <3s/>30s, new
      `untranscribed_chunks_rows()` reads all 3 chunk sidecars. New manifest
      `train_untranscribed_chunks.jsonl` (829,140 rows / 1,612h). Old
      `train_untranscribed.jsonl` retained at session-level for SSL use.
      Final manifest totals:
        - transcribed: 66,868 rows / 240.75h
        - pseudo_transcribed: 2,314,161 rows / 17,762.97h
        - untranscribed_chunks: 829,140 rows / 1,612.07h
        - untranscribed (session-level): 17,438 rows / 22,137.34h

Execution scheduled for when GPU is free (mostly the ~24-48 GPU-hr voxpopuli
gap chunking dominates).

## Next — phase 3: quality scoring (CPU + GPU work)

**Runs after Phase 2.5 normalization completes.** Detailed design in
`notes/QUALITY_SCORING_PHASE3.md`.

- [x] Tier-1 cheap stats (CPU, no GPU) — DONE 2026-05-20. `bin/quality_tier1.py`
      computed `rms_dbfs`, `peak_dbfs`, `is_clipped`, `silence_ratio` on
      **3,208,792 clips** (transcribed + pseudo_transcribed + untranscribed_chunks)
      in ~25 min (12 workers, ~6,000 clips/sec). Sidecar:
      `processed/quality/tier1.jsonl` (465 MB). Key findings:
        - 126,475 clipped clips (3.94%, mostly YouTube)
        - 21,132 mostly-silent (>90% silence) — drop candidates
        - 33 decode errors / 3.2M (negligible)
      Sidecar merge into `quality_flags` will be done by the (still-TODO)
      `bin/merge_quality_into_manifest.py` once DNSMOS finishes.
- [ ] Phase 3a YODAS2-only audio screen (LID + VAD + music, ~30 min) — answers
      the "how much English/music in YODAS2?" question and pre-filters before
      Phase 4 consensus.
- [ ] Tier-2 neural MOS (GPU):
  - [x] DNSMOS (P.835 SIG/BAK/OVRL) — DONE 2026-05-22. 3,189,774 rows
        scored in 24.2 h (~36 clips/s, 8 CPU workers, batch_size=32). 33
        errors (0.001%). Sidecar: `processed/quality/tier2_dnsmos.jsonl`
        (395 MB). Memory-bandwidth limited — tried 16 workers, no speedup
        (same 36 clips/s). Per-source median OVRL: librivox 3.90, mosel
        voxpopuli 3.50, voxpopuli_gap 3.26, yodas2 3.27, podcasts 2.89.
        **83% of corpus is training-grade (OVRL≥3.0)**, 1.4% drop
        candidate (OVRL<2.0).
  - [~] UTMOSv2 — **DROPPED 2026-05-21**. Per-clip `predict()` is 1.2 s/clip
        on GPU (10-clip benchmark on MOSEL voxpopuli) → 1,052 h ETA for
        3.19M rows; non-starter. DNSMOS covers the perceptual axis. See
        memory `project_tier2_utmosv2_dropped`. Custom batched inferencer
        could be ~10-30× faster but requires reverse-engineering UTMOSv2
        chunk fusion — not worth the effort for our use case.
  - [x] Silero VAD `vad_speech_ratio` — DONE 2026-05-21. `bin/quality_tier2.py
        --metric vad`, 8 workers, streaming generator, thread-pinned. 3.19M
        rows computed in 11.85h, 0 errors, 9-10G peak RAM. Sidecar:
        `processed/quality/tier2_vad.jsonl` (426 MB). Findings: 93.7% of clips
        have >70% speech ratio (training-grade); 0.59% are <30% speech
        (drop candidates); median 0.89.
  - [ ] YAMNet music detection (`music_likelihood`) — deferred; LID already
        flagged the YODAS2 contamination we cared about.
  - [x] VoxLingua107 LID — DONE 2026-05-21. `bin/quality_tier2.py --metric lid`
        with GPU-batched inference (env: `audio_ds`, batch_size=32, 1 GPU
        worker). Scope: YODAS2 + librivox + podcasts (56,707 clips, 12.8 min,
        1 error). Sidecar: `processed/quality/tier2_lid.jsonl`. Findings:
        **YODAS2 is 14% non-Hungarian** (~6,700 clips, ~25h, mostly English);
        LibriVox 100% clean, podcasts 99.4% clean. MOSEL voxpopuli and
        voxpopuli_unlabeled_gap intentionally skipped — they use the existing
        MOSEL `lid_not_hu` flag and parliamentary HU by definition respectively.
- [x] **Merge Tier-1 + Tier-2 sidecars into manifest `quality_flags`** —
      DONE 2026-05-22. `bin/merge_quality_into_manifest.py`. Run time ~50s
      (sidecar load 13s, merge 49s across 3.2M rows). Outputs:
      `train_transcribed_with_quality.jsonl` (111 MB, 66.8k rows),
      `train_pseudo_transcribed_with_quality.jsonl` (4.2 GB, 2.31M rows),
      `train_untranscribed_chunks_with_quality.jsonl` (1.3 GB, 829k rows),
      `stats_with_quality.json` (per-source quality counters).
      Missing scores omitted (not nulled). Originals untouched.
      Per-source DNSMOS≥3.0: librivox 99.9%, mosel_vp 87%, vp_gap 72.6%,
      yodas2 70.4%, podcasts 44.2%. YODAS2 `lid_not_hu`=6,678 (~14%).
- [ ] Derived `quality_tier` (GOLD/SILVER/BRONZE/REJECTED/PSEUDO) computed
      from Phase 3 audio metrics + Phase 4 consensus result (see
      `notes/QUALITY_SCORING_PHASE3.md` §6)

## Next — phase 4: multi-ASR consensus + tier-piramis

Detailed plan: `notes/HIGH_QUALITY_LABELS_STRATEGY.md`.

- [x] **Phase 4a-pre — smoke test** — DONE 2026-05-20. Ran Canary v2 +
      Qwen3-ASR FT + KenLM on voxpopuli_hu_labeled test split (1110 utts).
      Result: parliament WER ~11-12% per pillar, pairwise 10.46%, 31.3%
      exact match. Errors decorrelated. **Config 4.A viable** — proceed
      with 2-pillar (no need for Parakeet as 3rd unless 4a PoC GOLD < 3000h).
      Details: `notes/smoke_test/results.md`.
- [x] **Phase 4a — PoC on 100h** — DONE 2026-05-23. `bin/asr_consensus_poc_100h.py`
      run on 13,052 mosel clips (100.00 h, seed=42, random unbiased).
      Config: greedy both pillars, no KenLM (PoC speed; beam+KenLM CUDA-OOM
      on 30-sec mosel clips on 16 GB GPU). Runtime ~29 h wall-clock
      (Qwen 19 h, Canary 10 h, serial). **Pairwise WER 29.64% mean /
      16.67% median, exact match 3.3% → ~581 h projected GOLD.** Strategy
      doc threshold is 3,000 h GOLD; **Config 4.A insufficient**.
      Details: `notes/poc_100h/results.md`, snapshot in
      `notes/stats_snapshots/2026-05-23__phase4a_poc_done.md`.
- [x] **Phase 4b — Decision point** — DECIDED 2026-05-23. **Add Parakeet
      TDT v3 as 3rd pillar (Config 4.B).** Parakeet is fast (~0.22 s/clip
      at the 5.5× mosel length factor; adds ~1 GPU-day to Phase 4c).
      Pre-Phase-4c: re-run the same 100 h PoC sample with Parakeet to
      measure 3-pillar exact-match rate before committing the full corpus
      run. WER thresholds for SILVER/BRONZE (5%, 15%) kept as starting
      points; calibrate after 3-pillar PoC.
- [ ] **Phase 4c — Full consensus run (10-15 days local, $0).** All selected
      pillars on the full 17,757h `pseudo_transcribed` + 276h `transcribed`
      (caption cross-validation).
- [ ] **Phase 4d (optional experiment) — 4th pillar with text-prompt context.**
      Re-run Qwen FT on the existing PoC sample with `prompt = "Korábbi
      leiratozók: <canary>, <parakeet>. Adj pontos magyar átírást"`. Test
      on small PoC first (~25 GPU-h). Watch for correlated-error inheritance.
      Pre-condition: confirm Qwen3-ASR-FT supports prompt/context parameter.
      Trigger: only AFTER Plan B clean-audio Phase 4 baseline exists, so we
      can measure the marginal improvement honestly. See
      `notes/JOURNEY.md` "4th pillar with text-prompt context" entry +
      memory `project-prompted-asr-4th-pillar-idea`.
- [ ] **Phase 4d — Manifest update.** Add `quality_tier` field to every row
      (GOLD/SILVER/BRONZE/PSEUDO/REJECTED). Optionally split into filtered
      subset manifests (`train_gold.jsonl` etc.).

### Pillar candidates (config 4.A default, after 2026-05-20 benchmark)

**Default 2-pillar config (config 4.A in strategy doc):**
- **Canary 1B v2** (NeMo, Granary-trained, 1.02% YT-holdout WER, 12.34% parliament) — encoder-decoder
- **Qwen3-ASR FT + KenLM rescore** (user, yt-cleaned-v1, 9.15% CV-HU, 3.75% YT-holdout) — Speech-LLM

Architecturally diverse: enc-dec + Speech-LLM, different vendors, different
training data. If Phase 4a-pre shows pairwise WER too low → add Parakeet TDT v3
as 3rd pillar (only ~1 GPU-day extra for full run).

**Optional Tier 2 pillars:**
- Parakeet TDT 0.6B v3 (3.21% CV-HU, fastest at 0.04 s/file)
- VibeVoice-ASR LoRA v2 (~3.25% YT-holdout, slow at 12.72 s/file)
- Gemini 1.5 Pro Audio (cloud, ~$50-200 on GOLD subset only)

**Dropped from earlier plan:**
- Whisper-large-v3 — superseded by Canary v2 (better numbers + same family)
- wav2vec2-xls-r-300m-hungarian — older arch, no advantage over user models

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
- [x] Common Voice HU (v25.0) train partition — DOWNLOADED 2026-05-25
      (117,503 clips / 180.85h, CC0-1.0). Use_for designation pending
      curator review.
- [ ] Tier-3+ acoustic augmentations (noise/reverb/codec) for training.
- [ ] **Vintage public-domain HU audio (ASR diversity pilot).** Rationale:
      current corpus is acoustically biased toward clean studio audio (EP
      parliament, audiobooks, podcasts) — vintage / sávkorlátozott / zajos
      anyag ~0%. ASR robustness benefits from this distribution coverage.
      ASR-only (not TTS — stílus-mismatch).
      Sources: Filmhíradók Online (filmhiradokonline.hu / NFI Filmarchívum,
      Magyar Világhíradó 1924–1944), pre-1944 magyar játékfilmek PD szelete.
      Pilot scope:
  - [ ] Per-item jogtisztaság verifikáció (szerző + 70 év), 20–50h-os minta
        kiválasztása.
  - [ ] Download + audio demux (ffmpeg, mono 16kHz).
  - [ ] VAD szűrés (Silero, beszéd-csak részek kinyerése — várhatóan
        50–70%-os beszéd-arány).
  - [ ] ASR-konszenzus pseudolabel (Phase 4 pipeline újrahasználása, 2-pillar
        elég ennyi órán).
  - [ ] Agreement-rate kiértékelés: <60% → drop, ≥70% → scale-up a teljes
        elérhető ~500h-ra.
  - [ ] Manifest integráció: új `source_tier: vintage_noisy` flag, hogy
        TTS-tréningből kiszűrhető, ASR-tréningben súlyozható legyen.

## Open questions

- Investigate YODAS2 vs v1 utterance overlap — does v2 add real new audio
  coverage or just re-segments the same content?
- Do we need a held-out test split from our own data, or do CV-HU + FLEURS-HU
  suffice as evaluation?
- For TTS phase: F5-TTS-hungarian (community) vs CosyVoice 2 HU (Alibaba) —
  benchmark both on a small subset before committing.
