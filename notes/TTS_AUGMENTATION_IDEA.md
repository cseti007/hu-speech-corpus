# TTS-augmented ASR training — idea note

**Date:** 2026-05-17
**Status:** Proposed approach; pending baseline ASR results to decide if worth pursuing.

## The hypothesis

Domain bias is the central risk of using ~17,700h of EP-parliamentary VoxPopuli HU
as the bulk of training data. A reference-conditioned TTS pipeline could
synthesize **targeted diverse data** to fill the gaps (spontaneous conversation,
casual register, telephony, dialects, etc.) without needing to scrape more raw
audio.

## Why reference-conditioned TTS is the right form

Naive form: "train a TTS on noisy audio → it produces noisy speech."
Problem: TTS quality degrades with bad training data — phonetic accuracy
and prosody suffer. Garbage in → garbage out.

Modern form (2026 SOTA): **reference-conditioned synthesis** in F5-TTS,
CosyVoice 2, NaturalSpeech-3, VALL-E 2. A short reference clip (3-10 sec)
conditions the output to inherit:
- Voice timbre (speaker)
- Speaking style (pace, intonation)
- Acoustic conditions (background noise, reverb, codec)

So you train (or reuse) one strong multi-domain TTS, then **swap references**
at inference time to get diverse stylistic outputs.

## Concrete pipeline

```
Phase B: Reference pool (~500 diverse 5-10s clips, labeled by domain)
  - clean: LibriVox audiobook chunks
  - spontaneous: Hetvegi Kotekedo podcast snippets
  - interview: Szabad Europa Podcast slices
  - news: archive.org HU news clips
  - phone: synthesized telephony degradation of clean refs
  - noisy: street/cafe/office background mixed clean refs

Phase C: HU text corpus (millions of sentences)
  - Wikipedia HU (~600k articles)
  - Magyar Nyelvtechnologiai Korpusz (MaNyTI)
  - News crawls (RSS aggregations)
  - Forum/blog scrapes (where CC or licensed appropriately)

Phase D: Cross-product synthesis
  - For each text segment, sample N references
  - F5-TTS-hungarian or CosyVoice 2 HU as base
  - Output: text × reference matrix → diverse synthetic audio

Phase E: Real-world augmentation on top
  - MUSAN/DEMAND noise injection
  - Room IR convolution
  - Codec simulation (telephony G.711, MP3 64kbps, Opus 16kbps)
  - Speed/pitch perturbation

Phase F: ASR training on mix
  - Real (17,933h) + Synthetic-aug (target 10,000-30,000h)
  - Aim for <30% synthetic ratio (research shows higher hurts)
  - Domain-balanced batch sampling
```

## Cost estimate

- TTS model: F5-TTS-hungarian or CosyVoice 2 HU (already available or fine-tuneable)
- Synthesis: ~0.05x realtime on H100 → 30k h synth = 1500 GPU-h ≈ $300-600
- Plus ASR training compute on the bigger dataset (+30-50% over base)
- **Total path: +1-2 weeks effort, +$400-800 compute** vs baseline alone

## When to actually do this

**Not blindly first.** Order of operations:
1. ASR baseline on real data (Canary-1b-v2 fine-tune on 230h)
2. Measure per-domain failure (CV HU, FLEURS HU, custom held-out)
3. IF specific domain gaps identified → TTS-aug targeted at those gaps
4. IF general weakness → continual pretraining on 17,700h first

The user-proposed approach is valid and aligns with 2026 SOTA practice
(F5-TTS / CosyVoice 2 reference-condition transfer is exactly the
mechanism that closes the sim2real gap).

## Key reference for reference-conditioned acoustic transfer

CosyVoice 2 specifically excels at acoustic environment cloning — not just
voice timbre. F5-TTS uses flow matching which preserves reference style
including noise characteristics.

## TODO if this path is taken

- [ ] Build reference pool (500 labeled clips from existing sources)
- [ ] Decide TTS base: F5-TTS-hungarian vs CosyVoice 2 HU
- [ ] HU text corpus assembly (Wikipedia + news scrape + others)
- [ ] Synthesis script + augmentation pipeline
- [ ] Iteration loop: train ASR → measure → re-synthesize targeted

Related: [[feedback_hungarian_speech_datasets]]
