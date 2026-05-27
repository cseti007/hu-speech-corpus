#!/usr/bin/env python3
"""KenLM-adjudicated consensus on Phase 4 disagreement spans.

For every clip where the two pillars (Canary v2 + Qwen FT) disagree on at
least one word, run word-level alignment (`difflib.SequenceMatcher`) to
find each disagreement span, then score each pillar's candidate against
the 6-gram Hungarian KenLM (see project memory `phase4-toolchain`). The
winner of each span — when the log10 score difference is large enough —
gets stitched into a unified `kenlm_consensus_text`. Clips whose every
span has a clear KenLM winner are promoted to a new `GOLD-via-kenlm` tier
even though pairwise WER alone would mark them MEDIUM/LOW.

This is the length-bias mitigation we designed 2026-05-27: long clips
with a handful of isolated disagreements are rescued instead of dropped.

Inputs (under `processed/asr/`):
  canary_v2_<set>.jsonl    per-clip Canary output (raw + normalized)
  qwen_ft_<set>.jsonl      per-clip Qwen output (raw + normalized)
  consensus_<set>.jsonl    per-clip pairwise_wer / consensus_tier (raw)

Output:
  consensus_kenlm_<set>.jsonl   per-clip KenLM-adjudicated tier + fields

Run with the qwen3-asr env (where `kenlm` Python bindings live):
  /media/cseti/datassd/conda/miniconda3/envs/qwen3-asr/bin/python \
      bin/asr_consensus_kenlm.py --set smoke
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
ASR_DIR = DATA_ROOT / "processed" / "asr"
KENLM_PATH = (
    "/home/cseti/data2/AI/models/hub/"
    "models--sarpba--hungarian_kenlm_models/snapshots/"
    "b76549cbf67e75325ede3c555cebd2fd13261262/magyar_hplt_lm_6gram.kenlm"
)

# Tunable thresholds
CONTEXT_WORDS = 5      # words of context on each side of a disagreement span
MIN_SCORE_DELTA = 1.0  # log10 difference to call a clear winner (else "too close")
LOW_THRESHOLD = 0.50   # if raw pairwise_wer is above this, stay LOW regardless


# ============================================================
# Sidecar I/O
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ============================================================
# Word-level alignment + KenLM adjudication
# ============================================================

def score_in_context(model, before: list[str], candidate: list[str],
                     after: list[str]) -> float:
    """KenLM marginal log10 prob of `candidate` placed in context.

    Returns score(before+candidate+after) - score(before+after) so the
    comparison is fair across candidates of different word counts."""
    full = " ".join(before + candidate + after)
    base = " ".join(before + after)
    if not full.strip():
        return 0.0
    full_score = model.score(full, bos=True, eos=True)
    base_score = model.score(base, bos=True, eos=True) if base.strip() else 0.0
    return full_score - base_score


def adjudicate(canary_words: list[str], qwen_words: list[str],
               model) -> dict:
    """Word-align canary vs qwen, KenLM-adjudicate each disagreement span.

    Returns a dict:
      consensus_words   list[str] — reconstructed text (agreed + winners)
      n_resolved        int — spans where KenLM picked a clear winner
      n_unresolved      int — spans where |Δ| < MIN_SCORE_DELTA
      canary_wins       int
      qwen_wins         int
      details           list of per-span info (op, candidates, scores, winner)
    """
    sm = difflib.SequenceMatcher(a=canary_words, b=qwen_words, autojunk=False)
    consensus_words: list[str] = []
    n_resolved = n_unresolved = canary_wins = qwen_wins = 0
    details: list[dict] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            consensus_words.extend(canary_words[i1:i2])
            continue
        # Build context: agreed words BEFORE the span (look back into Canary's
        # array — outside the span the two arrays match) and the matching
        # prefix of AFTER words.
        before = canary_words[max(0, i1 - CONTEXT_WORDS):i1]
        after_c = canary_words[i2:i2 + CONTEXT_WORDS]
        after_q = qwen_words[j2:j2 + CONTEXT_WORDS]
        after: list[str] = []
        for x, y in zip(after_c, after_q):
            if x != y:
                break
            after.append(x)
        cand_canary = canary_words[i1:i2]
        cand_qwen = qwen_words[j1:j2]
        sc_c = score_in_context(model, before, cand_canary, after)
        sc_q = score_in_context(model, before, cand_qwen, after)
        delta = sc_q - sc_c
        winner = "tie"
        if abs(delta) < MIN_SCORE_DELTA:
            winner = "tie"
            n_unresolved += 1
            # Pick canary as a deterministic tie-breaker so consensus_words
            # always has SOMETHING coherent; the tier code knows there's an
            # unresolved span and won't promote the clip to GOLD-via-kenlm.
            consensus_words.extend(cand_canary)
        elif delta > 0:
            winner = "qwen"
            qwen_wins += 1
            n_resolved += 1
            consensus_words.extend(cand_qwen)
        else:
            winner = "canary"
            canary_wins += 1
            n_resolved += 1
            consensus_words.extend(cand_canary)
        details.append({
            "op": tag,
            "cand_canary": " ".join(cand_canary),
            "cand_qwen": " ".join(cand_qwen),
            "score_canary": round(sc_c, 3),
            "score_qwen": round(sc_q, 3),
            "delta": round(delta, 3),
            "winner": winner,
        })

    return {
        "consensus_words": consensus_words,
        "n_resolved": n_resolved,
        "n_unresolved": n_unresolved,
        "canary_wins": canary_wins,
        "qwen_wins": qwen_wins,
        "details": details,
    }


def kenlm_tier(pairwise_wer: float, n_resolved: int, n_unresolved: int) -> str:
    if pairwise_wer == 0:
        return "GOLD"
    if pairwise_wer > LOW_THRESHOLD:
        return "LOW"
    if n_unresolved == 0 and n_resolved > 0:
        return "GOLD-via-kenlm"
    if n_unresolved == 1:
        return "HIGH"
    if n_unresolved >= 2:
        return "MEDIUM"
    return "LOW"


# ============================================================
# Main
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="smoke",
                   help="smoke / dev / test (default smoke).")
    p.add_argument("--keep-details", action="store_true",
                   help="Include per-span details in the sidecar (default off — "
                        "details are useful for debugging but bloat the file).")
    args = p.parse_args()

    set_name = args.set
    canary_path = ASR_DIR / f"canary_v2_{set_name}.jsonl"
    qwen_path = ASR_DIR / f"qwen_ft_{set_name}.jsonl"
    cons_path = ASR_DIR / f"consensus_{set_name}.jsonl"
    out_path = ASR_DIR / f"consensus_kenlm_{set_name}.jsonl"

    for p_ in (canary_path, qwen_path, cons_path):
        if not p_.is_file():
            print(f"[error] missing input: {p_}", file=sys.stderr)
            return 2

    try:
        import kenlm
    except ImportError:
        print("[error] `kenlm` not importable. Run with qwen3-asr env.",
              file=sys.stderr)
        return 2

    print(f"[init] loading KenLM ({KENLM_PATH.split('/')[-1]})...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    model = kenlm.Model(KENLM_PATH)
    print(f"[init] KenLM loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    canary = {r["utterance_id"]: r for r in load_jsonl(canary_path)}
    qwen = {r["utterance_id"]: r for r in load_jsonl(qwen_path)}
    cons = {r["utterance_id"]: r for r in load_jsonl(cons_path)}
    print(f"[init] {len(canary)} canary / {len(qwen)} qwen / {len(cons)} "
          f"consensus rows", file=sys.stderr)

    out_rows = []
    t0 = time.time()
    n_done = 0
    for uid, c_row in canary.items():
        q_row = qwen.get(uid)
        cons_row = cons.get(uid)
        if q_row is None or cons_row is None:
            continue
        c_words = (c_row["normalized"] or "").split()
        q_words = (q_row["normalized"] or "").split()
        adj = adjudicate(c_words, q_words, model)
        pw = cons_row.get("pairwise_wer") or 0.0
        tier = kenlm_tier(pw, adj["n_resolved"], adj["n_unresolved"])
        out = {
            "utterance_id": uid,
            "source": cons_row.get("source"),
            "pairwise_wer": pw,
            "consensus_tier": cons_row.get("consensus_tier"),
            "duration_bucket": cons_row.get("duration_bucket"),
            "kenlm_resolved_count": adj["n_resolved"],
            "kenlm_unresolved_count": adj["n_unresolved"],
            "kenlm_canary_wins": adj["canary_wins"],
            "kenlm_qwen_wins": adj["qwen_wins"],
            "kenlm_consensus_text": " ".join(adj["consensus_words"]),
            "kenlm_tier": tier,
        }
        if args.keep_details:
            out["kenlm_details"] = adj["details"]
        out_rows.append(out)
        n_done += 1
        if n_done % 200 == 0:
            print(f"  [...] {n_done}/{len(canary)} done", file=sys.stderr)

    print(f"[done] adjudication: {n_done} clips in {time.time() - t0:.1f}s",
          file=sys.stderr)
    write_jsonl(out_path, out_rows)
    print(f"[write] {out_path}", file=sys.stderr)

    # ----- Summary -----
    print()
    print("=" * 72)
    print(f"KenLM-adjudicated consensus summary — set={set_name}")
    print("=" * 72)
    print(f"\nTier (raw pairwise WER) → tier (KenLM-adjudicated):")
    transition = Counter()
    for r in out_rows:
        transition[(r["consensus_tier"], r["kenlm_tier"])] += 1
    raw_tiers = ["GOLD", "HIGH", "MEDIUM", "LOW"]
    kenlm_tiers = ["GOLD", "GOLD-via-kenlm", "HIGH", "MEDIUM", "LOW"]
    header = f"{'raw \\ kenlm':<14}" + "".join(f"{t:>16s}" for t in kenlm_tiers)
    print(header)
    for raw in raw_tiers:
        row = [f"{raw:<14}"]
        for kt in kenlm_tiers:
            n = transition.get((raw, kt), 0)
            row.append(f"{n:>16}")
        print("".join(row))

    print(f"\nTier distribution comparison:")
    print(f"  raw                  ", end="")
    raw_counts = Counter(r["consensus_tier"] for r in out_rows)
    for t in raw_tiers:
        n = raw_counts.get(t, 0)
        print(f"{t}={n}  ", end="")
    print()
    print(f"  kenlm-adjudicated    ", end="")
    kenlm_counts = Counter(r["kenlm_tier"] for r in out_rows)
    for t in kenlm_tiers:
        n = kenlm_counts.get(t, 0)
        print(f"{t}={n}  ", end="")
    print()

    print(f"\nGOLD%% by duration bucket (raw → kenlm-trainable):")
    by_bucket = defaultdict(lambda: {"total": 0, "raw_gold": 0, "kenlm_gold": 0})
    for r in out_rows:
        b = by_bucket[r["duration_bucket"]]
        b["total"] += 1
        if r["consensus_tier"] == "GOLD":
            b["raw_gold"] += 1
        if r["kenlm_tier"] in ("GOLD", "GOLD-via-kenlm"):
            b["kenlm_gold"] += 1
    for bucket in ("<5s", "5-10s", "10-20s", "20-30s", "unknown"):
        b = by_bucket.get(bucket)
        if not b or b["total"] == 0:
            continue
        raw_pct = b["raw_gold"] / b["total"] * 100
        kenlm_pct = b["kenlm_gold"] / b["total"] * 100
        delta = kenlm_pct - raw_pct
        print(f"  {bucket:8s}  raw={raw_pct:5.1f}%  →  kenlm={kenlm_pct:5.1f}%  "
              f"(+{delta:.1f}pp,  n={b['total']})")

    n_promoted = sum(1 for r in out_rows
                     if r["kenlm_tier"] == "GOLD-via-kenlm"
                     and r["consensus_tier"] != "GOLD")
    print(f"\nClips promoted to GOLD-via-kenlm: {n_promoted} "
          f"({n_promoted/max(1,len(out_rows))*100:.1f}% of total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
