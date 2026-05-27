#!/usr/bin/env python3
"""Investigate KenLM-adjudicated ASR consensus on smoke set disagreements.

For every smoke clip where the two pillars disagree, run word-level
alignment (Python `difflib.SequenceMatcher.get_opcodes()`), find each
disagreement span, and score each pillar's candidate via KenLM. Print a
side-by-side view that a human can spot-check before committing to a full
pipeline integration.

The KenLM model is the 6-gram Hungarian model we already use for Qwen FT
rescoring (see project memory `phase4-toolchain`).

Run with the qwen3-asr env (where `kenlm` Python bindings live):
  /media/cseti/datassd/conda/miniconda3/envs/qwen3-asr/bin/python \
      bin/investigate_kenlm_adjudication.py [--limit 20]

Output goes to stdout; spot-check by reading.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
KENLM_PATH = (
    "/home/cseti/data2/AI/models/hub/"
    "models--sarpba--hungarian_kenlm_models/snapshots/"
    "b76549cbf67e75325ede3c555cebd2fd13261262/magyar_hplt_lm_6gram.kenlm"
)
DEFAULT_SET = "smoke"
CONTEXT_WORDS = 5      # words of context each side of a disagreement span
MIN_SCORE_DELTA = 1.0  # log10 difference to call a "clear" winner


def load_pillar(path: Path) -> dict[str, list[str]]:
    """Read per-pillar sidecar → {utterance_id: normalized words}."""
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["utterance_id"]] = (r["normalized"] or "").split()
    return out


def score_in_context(model, before: list[str], candidate: list[str],
                     after: list[str]) -> float:
    """KenLM log10 prob of `candidate` words placed between context words.

    KenLM scores entire sentences; we score the sequence
    `before + candidate + after` so the n-gram window can see candidate
    in context. Returning the FULL sentence's score isn't fair across
    candidates of different word counts (insertions vs same-length subs),
    so we subtract the context-only score → returns the marginal
    contribution of the candidate."""
    full = " ".join(before + candidate + after)
    base = " ".join(before + after)
    if not full.strip():
        return 0.0
    full_score = model.score(full, bos=True, eos=True)
    base_score = model.score(base, bos=True, eos=True) if base.strip() else 0.0
    return full_score - base_score


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default=DEFAULT_SET,
                   help="Which set to investigate (default: smoke).")
    p.add_argument("--limit", type=int, default=15,
                   help="Maximum disagreement spans to print (default 15).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    try:
        import kenlm
    except ImportError:
        print("[error] `kenlm` not importable. Run with qwen3-asr env:",
              file=sys.stderr)
        print("  /media/cseti/datassd/conda/miniconda3/envs/qwen3-asr/bin/python "
              "bin/investigate_kenlm_adjudication.py", file=sys.stderr)
        return 2

    print(f"[init] loading KenLM ({KENLM_PATH.split('/')[-1]})...",
          file=sys.stderr, flush=True)
    model = kenlm.Model(KENLM_PATH)

    canary_path = DATA_ROOT / "processed" / "asr" / f"canary_v2_{args.set}.jsonl"
    qwen_path = DATA_ROOT / "processed" / "asr" / f"qwen_ft_{args.set}.jsonl"
    print(f"[init] loading pillars from {canary_path.parent}",
          file=sys.stderr)
    canary = load_pillar(canary_path)
    qwen = load_pillar(qwen_path)
    common = sorted(set(canary) & set(qwen))
    print(f"[init] {len(common)} clips in both pillars", file=sys.stderr)

    # Collect disagreement spans across all clips.
    spans: list[dict] = []
    for uid in common:
        a = canary[uid]
        b = qwen[uid]
        if a == b:
            continue
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            # Context (use the matched portion at the boundaries, from
            # either side — they're equal there).
            before_start = max(0, i1 - CONTEXT_WORDS)
            after_end_a = min(len(a), i2 + CONTEXT_WORDS)
            after_end_b = min(len(b), j2 + CONTEXT_WORDS)
            # Use the AGREED context (same in both since we're outside the
            # disagreement span). Pull from `a` for simplicity.
            before = a[before_start:i1]
            # After-context might differ if there are further disagreements;
            # pick the shorter to stay aligned.
            after_a = a[i2:after_end_a]
            after_b = b[j2:after_end_b]
            # Aligned context: take the longest common prefix of the two
            # after-contexts so KenLM sees a consistent base sentence.
            after = []
            for x, y in zip(after_a, after_b):
                if x != y:
                    break
                after.append(x)
            spans.append({
                "utterance_id": uid,
                "op": tag,
                "before": before,
                "cand_canary": a[i1:i2],
                "cand_qwen": b[j1:j2],
                "after": after,
            })

    print(f"[init] {len(spans)} total disagreement spans found",
          file=sys.stderr)

    import random
    rng = random.Random(args.seed)
    rng.shuffle(spans)
    spans = spans[:args.limit]

    # Score each span via KenLM, print side-by-side.
    print()
    print("=" * 90)
    print("KenLM-adjudicated disagreement spans (random sample)")
    print("=" * 90)
    n_canary_wins = 0
    n_qwen_wins = 0
    n_tie = 0
    for i, s in enumerate(spans, 1):
        ctx_b = " ".join(s["before"]) or "<bos>"
        ctx_a = " ".join(s["after"]) or "<eos>"
        cand_c = " ".join(s["cand_canary"]) or "<EMPTY>"
        cand_q = " ".join(s["cand_qwen"]) or "<EMPTY>"
        sc_c = score_in_context(model, s["before"], s["cand_canary"], s["after"])
        sc_q = score_in_context(model, s["before"], s["cand_qwen"], s["after"])
        delta = sc_q - sc_c
        if abs(delta) < MIN_SCORE_DELTA:
            winner = "(too close)"
            n_tie += 1
        elif delta > 0:
            winner = "Qwen"
            n_qwen_wins += 1
        else:
            winner = "Canary"
            n_canary_wins += 1
        print()
        print(f"[{i}] uid: {s['utterance_id']}  op={s['op']}")
        print(f"    BEFORE: ...{ctx_b}")
        print(f"    Canary: {cand_c!r}      (log10 P = {sc_c:+.2f})")
        print(f"    Qwen:   {cand_q!r}      (log10 P = {sc_q:+.2f})")
        print(f"    AFTER:  {ctx_a}...")
        print(f"    Δ (Qwen − Canary) = {delta:+.2f}  →  winner: {winner}")

    print()
    print("=" * 90)
    print(f"Summary over {len(spans)} spans:")
    print(f"  Canary wins (Δ < -{MIN_SCORE_DELTA}):  {n_canary_wins}")
    print(f"  Qwen wins   (Δ > +{MIN_SCORE_DELTA}):  {n_qwen_wins}")
    print(f"  Too close to call:                    {n_tie}")
    print()
    print("Spot-check: do the KenLM winners look right? If yes, integrate via")
    print("bin/asr_consensus_kenlm.py (TODO) → new GOLD-via-kenlm tier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
