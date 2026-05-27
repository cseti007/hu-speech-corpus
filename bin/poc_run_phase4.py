#!/usr/bin/env python3
"""Phase 4 ASR consensus runner — Config 4.A (Canary v2 + Qwen FT FT).

Per Rule 8 ladder (smoke → dev → full), drives via `--set`:
  smoke  → processed/parquets/smoke.parquet   (~300 clips, validation)
  dev    → processed/parquets/dev.parquet     (~36k clips / 97h, calibration)
  test   → processed/parquets/test.parquet    (1,022 clips / 3h, eval)
  (full corpus deferred to sponsor — too much GPU on local hw.)

Pipeline:
  1. Read the set's parquet, symlink each clip into a temp audio dir,
     write ground_truth.jsonl with source_caption (used as reference for
     per-pillar ref_wer where available; toolkit also reports its own WER).
  2. Subprocess → Canary v2 inference (env: audio_ds, greedy decode).
  3. Subprocess → Qwen FT inference (env: qwen3-asr, KenLM rescore + beam=5).
  4. Read each pillar's transcriptions_*.csv, re-normalize via
     `normalize_hu()` (canonical, NOT toolkit's older normalize_text), and
     write three sidecars:

     processed/asr/canary_v2_<set>.jsonl    — per-clip canary output
     processed/asr/qwen_ft_<set>.jsonl      — per-clip qwen output
     processed/asr/consensus_<set>.jsonl    — per-clip consensus

  5. Print a summary table (per-pillar median ref_wer, pairwise WER stats,
     consensus tier distribution, duration-bucket GOLD%, etc.).

Each per-pillar sidecar row:
  { utterance_id, raw, normalized, ref_wer | null, runtime_sec | null }

Each consensus sidecar row:
  { utterance_id, pairwise_wer, consensus_tier, duration_bucket,
    consensus_text | null }

Idempotent flags:
  --skip-prep      reuse the temp audio dir + ground_truth.jsonl
  --skip-canary    skip the Canary subprocess (reuse output dir)
  --skip-qwen      skip the Qwen subprocess (reuse output dir)
  --skip-consensus skip the final WER + tier pass

Run with the BASE conda env (subprocess switches to pillar-specific envs):
  /media/cseti/datassd/conda/miniconda3/bin/python bin/poc_run_phase4.py --set smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

# ============================================================
# Paths + config (mostly mirrors project memory `phase4-toolchain`)
# ============================================================

DATA_ROOT = Path("/home/cseti/datassd2/hu-speech-corpus")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SET_PARQUETS = {
    "smoke": DATA_ROOT / "processed" / "parquets" / "smoke.parquet",
    "dev":   DATA_ROOT / "processed" / "parquets" / "dev.parquet",
    "test":  DATA_ROOT / "processed" / "parquets" / "test.parquet",
}
ASR_OUT_DIR = DATA_ROOT / "processed" / "asr"
WORK_DIR_TEMPLATE = "/tmp/phase4_{set}"

# Pillar config
QWEN_FT_PATH = (
    "/home/cseti/data2/AI/training/Audio/output/qwen-asr/"
    "982457_qwen3-asr_yt-cleaned-v1/Qwen3-ASR_Hungarian_v1_17000"
)
KENLM_PATH = (
    "/home/cseti/data2/AI/models/hub/"
    "models--sarpba--hungarian_kenlm_models/snapshots/"
    "b76549cbf67e75325ede3c555cebd2fd13261262/magyar_hplt_lm_6gram.kenlm"
)
CANARY_MODEL = "nvidia/canary-1b-v2"

QWEN_ENV_PY = "/media/cseti/datassd/conda/miniconda3/envs/qwen3-asr/bin/python"
NEMO_ENV_PY = "/media/cseti/datassd/conda/miniconda3/envs/audio_ds/bin/python"
TOOLKIT_DIR = Path("/home/cseti/data2/Develop/Github-cseti/asr-eval-toolkit")

# Anti-fragmentation env extras for long parliament clips on 16 GB GPU.
SUBPROCESS_ENV_EXTRAS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

# WER-based tier thresholds (locked in 2026-05-27, see project memory
# `phase4-toolchain`). Length-normalized so longer clips aren't biased
# out (exact-match would penalize them).
TIER_THRESHOLDS = [
    ("GOLD",   0.02),
    ("HIGH",   0.05),
    ("MEDIUM", 0.10),
    ("LOW",    float("inf")),
]


def assign_tier(wer: float) -> str:
    for label, threshold in TIER_THRESHOLDS:
        if wer <= threshold:
            return label
    return "LOW"


def duration_bucket(duration_sec: float | None) -> str:
    if duration_sec is None:
        return "unknown"
    if duration_sec < 5:
        return "<5s"
    if duration_sec < 10:
        return "5-10s"
    if duration_sec < 20:
        return "10-20s"
    return "20-30s"


# ============================================================
# normalize_hu (vendored via import from build_manifest_v5)
# ============================================================

def _load_normalize_hu():
    """Import `normalize_hu` from build_manifest_v5 — single source of truth."""
    sys.path.insert(0, str(PROJECT_ROOT / "bin"))
    from build_manifest_v5 import normalize_hu  # noqa: WPS433
    return normalize_hu


# ============================================================
# Audio prep — symlink set's clips into a temp dir, write ground_truth.jsonl
# ============================================================

_FNAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(utt: str) -> str:
    return _FNAME_SAFE.sub("_", utt)


def read_set_rows(set_name: str) -> Iterable[dict]:
    """Yield dict rows from the named set's parquet via pyarrow."""
    import pyarrow.parquet as pq
    table = pq.read_table(str(SET_PARQUETS[set_name]),
                          columns=["utterance_id", "audio_path",
                                   "duration_sec", "source", "transcripts"])
    cols = table.column_names
    for batch in table.to_batches():
        d = batch.to_pydict()
        for i in range(len(d[cols[0]])):
            tr = d["transcripts"][i] or {}
            yield {
                "utterance_id": d["utterance_id"][i],
                "audio_path": d["audio_path"][i],
                "duration_sec": d["duration_sec"][i],
                "source": d["source"][i],
                "raw": tr.get("source_caption"),
                "norm": tr.get("source_caption_normalized"),
            }


def prepare_audio_dir(set_name: str, work_dir: Path) -> dict[str, dict]:
    """Symlink each clip's audio into `work_dir` and write ground_truth.jsonl.

    Returns a `{filename: {utterance_id, duration_sec, raw_caption,
    normalized_caption}}` mapping for downstream join."""
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_set_rows(set_name))

    fname_to_meta: dict[str, dict] = {}
    n_linked = 0
    n_skip_audio = 0
    gt_entries = []
    for row in rows:
        src = Path(row["audio_path"])
        if not src.is_file():
            n_skip_audio += 1
            continue
        ext = src.suffix or ".ogg"
        fname = safe_filename(row["utterance_id"]) + ext
        dst = work_dir / fname
        if dst.is_symlink() or dst.exists():
            try:
                dst.unlink()
            except OSError:
                pass
        try:
            dst.symlink_to(src)
        except OSError:
            n_skip_audio += 1
            continue
        n_linked += 1
        # Use raw caption as the toolkit's ground_truth so it can report its
        # own per-row WER side-by-side; the toolkit applies its own
        # normalize_text. We additionally compute our own ref_wer via
        # normalize_hu in the post-process step.
        gt_text = (row["raw"] or "").strip()
        gt_entries.append({"audio": fname, "text": gt_text})
        fname_to_meta[fname] = {
            "utterance_id": row["utterance_id"],
            "source": row["source"],
            "duration_sec": row["duration_sec"],
            "raw_caption": row["raw"],
            "normalized_caption": row["norm"],
        }

    (work_dir / "ground_truth.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in gt_entries) + "\n",
        encoding="utf-8",
    )
    print(f"[prep] {n_linked:,} clips linked into {work_dir} "
          f"({n_skip_audio} skipped — audio missing)", file=sys.stderr)
    return fname_to_meta


# ============================================================
# Pillar inference (subprocess to toolkit/batch_evaluate.py)
# ============================================================

def _subprocess_env() -> dict:
    env = os.environ.copy()
    env.update(SUBPROCESS_ENV_EXTRAS)
    return env


def run_canary(work_dir: Path, out_dir: Path) -> float:
    """Greedy Canary v2 via NeMo. Env: audio_ds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        NEMO_ENV_PY, "batch_evaluate.py", "--model-type", "nemo",
        "-i", str(work_dir),
        "-g", str(work_dir / "ground_truth.jsonl"),
        "-m", CANARY_MODEL,
        "--nemo-language", "hu",
        "-o", str(out_dir),
    ]
    print(f"[canary] running batch_evaluate (greedy, NeMo)...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=TOOLKIT_DIR, check=True, env=_subprocess_env())
    return time.time() - t0


def run_qwen(work_dir: Path, out_dir: Path) -> float:
    """Qwen FT + KenLM rescore + beam=5. Env: qwen3-asr."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        QWEN_ENV_PY, "batch_evaluate.py", "--model-type", "qwen",
        "-i", str(work_dir),
        "-g", str(work_dir / "ground_truth.jsonl"),
        "-m", QWEN_FT_PATH,
        "--language", "Hungarian",
        "--kenlm-model", KENLM_PATH,
        "--num-beams", "5",
        "-o", str(out_dir),
    ]
    print(f"[qwen] running batch_evaluate (KenLM + beam=5)...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=TOOLKIT_DIR, check=True, env=_subprocess_env())
    return time.time() - t0


# ============================================================
# Read toolkit CSV outputs
# ============================================================

def load_transcriptions(out_dir: Path) -> dict[str, dict]:
    """Read transcriptions_*.csv from a pillar's output dir.

    Returns {filename: {raw, ground_truth}}. We pick the first available
    of `transcription_kenlm` / `transcription_beam` / `transcription` —
    the toolkit emits whichever matches the run config."""
    csvs = list(out_dir.glob("transcriptions_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no transcriptions_*.csv in {out_dir}")
    csv_path = csvs[0]
    result = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("transcription_kenlm")
                   or row.get("transcription_beam")
                   or row.get("transcription_greedy")
                   or row.get("transcription")
                   or "")
            result[row["file"]] = {
                "raw": raw,
                "ground_truth": row.get("ground_truth", ""),
            }
    print(f"[load] {csv_path.name}: {len(result)} rows", file=sys.stderr)
    return result


# ============================================================
# WER (Levenshtein-based, jiwer-compatible)
# ============================================================

def wer(ref_words: list[str], hyp_words: list[str]) -> float:
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    m, n = len(ref_words), len(hyp_words)
    if n == 0:
        return 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / m


# ============================================================
# Sidecar writers + consensus
# ============================================================

def write_pillar_sidecar(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build_pillar_sidecar(pillar_name: str, transcriptions: dict[str, dict],
                         fname_to_meta: dict[str, dict],
                         normalize_hu) -> list[dict]:
    """Per-clip pillar output with our normalize_hu + ref_wer."""
    out = []
    for fname, t in transcriptions.items():
        meta = fname_to_meta.get(fname)
        if meta is None:
            # Toolkit emitted output for a file we don't know — skip.
            continue
        raw = t["raw"]
        normalized = normalize_hu(raw) or ""
        ref = meta.get("normalized_caption")
        if ref:
            ref_wer = wer(ref.split(), normalized.split())
        else:
            ref_wer = None
        out.append({
            "utterance_id": meta["utterance_id"],
            "source": meta["source"],
            "raw": raw,
            "normalized": normalized,
            "ref_wer": ref_wer,
        })
    return out


def build_consensus_sidecar(canary_rows: list[dict], qwen_rows: list[dict],
                            fname_to_meta: dict[str, dict]) -> list[dict]:
    by_uid_canary = {r["utterance_id"]: r for r in canary_rows}
    by_uid_qwen = {r["utterance_id"]: r for r in qwen_rows}
    common = sorted(set(by_uid_canary) & set(by_uid_qwen))
    out = []
    duration_by_uid = {m["utterance_id"]: m["duration_sec"]
                       for m in fname_to_meta.values()}
    for uid in common:
        c_norm = by_uid_canary[uid]["normalized"].split()
        q_norm = by_uid_qwen[uid]["normalized"].split()
        if not c_norm and not q_norm:
            pw = 0.0  # both empty — trivially identical
        elif not c_norm or not q_norm:
            pw = 1.0  # one is empty, other not → max disagreement
        else:
            pw = wer(c_norm, q_norm)
        tier = assign_tier(pw)
        # Consensus text only when GOLD (tight agreement). Canary tends to
        # be better on parliamentary register; pick canary's normalized
        # form as the canonical when they agree.
        consensus_text = by_uid_canary[uid]["normalized"] if tier == "GOLD" else None
        out.append({
            "utterance_id": uid,
            "source": by_uid_canary[uid]["source"],
            "pairwise_wer": round(pw, 4),
            "consensus_tier": tier,
            "duration_bucket": duration_bucket(duration_by_uid.get(uid)),
            "consensus_text": consensus_text,
        })
    return out


# ============================================================
# Summary reporting
# ============================================================

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = int(round((p / 100.0) * (len(vs) - 1)))
    return vs[k]


def print_summary(set_name: str, canary: list[dict], qwen: list[dict],
                  consensus: list[dict], runtimes: dict[str, float]) -> None:
    print("\n" + "=" * 70)
    print(f"Phase 4 SUMMARY — set={set_name}")
    print("=" * 70)
    # Runtimes
    print("\nRuntime:")
    for k, v in runtimes.items():
        print(f"  {k:10s}  {v/60:6.1f} min")

    # Per-pillar ref_wer
    for name, rows in (("Canary v2", canary), ("Qwen FT", qwen)):
        ref_wers = [r["ref_wer"] for r in rows if r["ref_wer"] is not None]
        by_source = defaultdict(list)
        for r in rows:
            if r["ref_wer"] is not None:
                by_source[r["source"]].append(r["ref_wer"])
        print(f"\n{name} — ref_wer vs source_caption_normalized "
              f"({len(ref_wers)} clips with caption):")
        if ref_wers:
            print(f"  overall   median={percentile(ref_wers, 50)*100:.2f}%  "
                  f"p25={percentile(ref_wers, 25)*100:.2f}%  "
                  f"p75={percentile(ref_wers, 75)*100:.2f}%  "
                  f"mean={sum(ref_wers)/len(ref_wers)*100:.2f}%")
            for src in sorted(by_source):
                v = by_source[src]
                print(f"  {src:24s} median={percentile(v, 50)*100:.2f}%  n={len(v)}")

    # Pairwise + tier
    pws = [r["pairwise_wer"] for r in consensus]
    print(f"\nPairwise WER (Qwen ↔ Canary) over {len(pws)} clips:")
    if pws:
        print(f"  median={percentile(pws, 50)*100:.2f}%  "
              f"p25={percentile(pws, 25)*100:.2f}%  "
              f"p75={percentile(pws, 75)*100:.2f}%  "
              f"mean={sum(pws)/len(pws)*100:.2f}%")

    print(f"\nTier distribution (GOLD ≤ 0.02 / HIGH ≤ 0.05 / "
          f"MEDIUM ≤ 0.10 / LOW > 0.10):")
    tier_counts = Counter(r["consensus_tier"] for r in consensus)
    total = max(1, sum(tier_counts.values()))
    for tier in ("GOLD", "HIGH", "MEDIUM", "LOW"):
        n = tier_counts.get(tier, 0)
        print(f"  {tier:8s} {n:>5,}  {n/total*100:5.1f}%")

    print(f"\nGOLD% by duration bucket (length-bias diagnostic):")
    by_bucket = defaultdict(list)
    for r in consensus:
        by_bucket[r["duration_bucket"]].append(r["consensus_tier"])
    for bucket in ("<5s", "5-10s", "10-20s", "20-30s", "unknown"):
        tiers = by_bucket.get(bucket, [])
        if not tiers:
            continue
        gold = sum(1 for t in tiers if t == "GOLD")
        print(f"  {bucket:8s} {gold:>4,}/{len(tiers):<5,}  "
              f"GOLD={gold/len(tiers)*100:5.1f}%")
    print()


# ============================================================
# Main
# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["smoke", "dev", "test"], default="smoke",
                   help="Which parquet set to run consensus on (default: smoke).")
    p.add_argument("--skip-prep", action="store_true",
                   help="Reuse the existing temp audio dir + ground_truth.")
    p.add_argument("--skip-canary", action="store_true",
                   help="Skip Canary subprocess (reuse output dir).")
    p.add_argument("--skip-qwen", action="store_true",
                   help="Skip Qwen subprocess (reuse output dir).")
    p.add_argument("--skip-consensus", action="store_true",
                   help="Skip the consensus + sidecar write pass.")
    args = p.parse_args()

    set_name = args.set
    if not SET_PARQUETS[set_name].is_file():
        print(f"[error] missing parquet: {SET_PARQUETS[set_name]}", file=sys.stderr)
        return 2

    work_dir = Path(WORK_DIR_TEMPLATE.format(set=set_name))
    canary_out = work_dir / "canary_raw"
    qwen_out = work_dir / "qwen_raw"

    # ---- 1. Audio prep ----
    fname_to_meta_cache = work_dir / "fname_to_meta.json"
    if args.skip_prep and fname_to_meta_cache.is_file():
        with fname_to_meta_cache.open(encoding="utf-8") as f:
            fname_to_meta = json.load(f)
        print(f"[prep] reused {len(fname_to_meta)} entries from cache",
              file=sys.stderr)
    else:
        fname_to_meta = prepare_audio_dir(set_name, work_dir)
        fname_to_meta_cache.write_text(
            json.dumps(fname_to_meta, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    runtimes = {}

    # ---- 2. Canary inference ----
    if not args.skip_canary:
        runtimes["canary"] = run_canary(work_dir, canary_out)
    else:
        print("[canary] skipped", file=sys.stderr)

    # ---- 3. Qwen inference ----
    if not args.skip_qwen:
        runtimes["qwen"] = run_qwen(work_dir, qwen_out)
    else:
        print("[qwen] skipped", file=sys.stderr)

    if args.skip_consensus:
        print("[consensus] skipped", file=sys.stderr)
        return 0

    # ---- 4. Post-process: normalize, sidecars ----
    normalize_hu = _load_normalize_hu()

    canary_t = load_transcriptions(canary_out)
    qwen_t = load_transcriptions(qwen_out)
    canary_rows = build_pillar_sidecar("canary_v2", canary_t,
                                       fname_to_meta, normalize_hu)
    qwen_rows = build_pillar_sidecar("qwen_ft", qwen_t,
                                     fname_to_meta, normalize_hu)
    consensus_rows = build_consensus_sidecar(canary_rows, qwen_rows, fname_to_meta)

    canary_side = ASR_OUT_DIR / f"canary_v2_{set_name}.jsonl"
    qwen_side = ASR_OUT_DIR / f"qwen_ft_{set_name}.jsonl"
    consensus_side = ASR_OUT_DIR / f"consensus_{set_name}.jsonl"
    write_pillar_sidecar(canary_side, canary_rows)
    write_pillar_sidecar(qwen_side, qwen_rows)
    write_pillar_sidecar(consensus_side, consensus_rows)

    print(f"\nSidecars written:")
    print(f"  {canary_side}    ({len(canary_rows)} rows)")
    print(f"  {qwen_side}      ({len(qwen_rows)} rows)")
    print(f"  {consensus_side} ({len(consensus_rows)} rows)")

    # ---- 5. Summary ----
    print_summary(set_name, canary_rows, qwen_rows, consensus_rows, runtimes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
