#!/usr/bin/env python3
"""Curator backend: Flask + DuckDB over manifest.parquet, plus audio streaming.

Serves a single-page UI for filtering / sorting / paginating manifest rows
and playing the per-row audio inline. Read-only — no curation marks yet.

Endpoints:
  GET /                        single-page UI (templates/index.html)
  GET /api/stats               { total_rows, total_hours }
  GET /api/sources             distinct sources + per-source row count
  GET /api/lid_values          distinct qf_lid values + counts (for the dropdown)
  GET /api/rows?<filters>      filtered + sorted + paginated rows
  GET /api/row/<utterance_id>  full row detail (all columns + JSON blobs)
  GET /audio/<rel_path>        stream audio file with HTTP Range support

Run via the launcher (recommended):
  bash bin/curator/serve.sh
  # browser: http://localhost:8002

Or directly:
  python bin/curator/app.py

Dependencies: flask, duckdb, pyarrow. Install in your environment of choice.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

# DuckDB is imported lazily after path validation to give a clean error if
# the dependency isn't installed.
_DATA_ROOT_ENV = os.environ.get("HU_CORPUS_ROOT")
if not _DATA_ROOT_ENV:
    raise SystemExit(
        "HU_CORPUS_ROOT env var is not set. Export it to your corpus storage "
        "root before running the curator."
    )
DATA_ROOT = Path(_DATA_ROOT_ENV)
# CURATOR_PARQUET overrides the default; set by bin/curator/serve.sh based
# on the optional CLI alias (e.g. `serve.sh poc` -> manifest_poc_100h.parquet).
MANIFEST_PARQUET = Path(os.environ.get(
    "CURATOR_PARQUET",
    str(DATA_ROOT / "processed" / "manifests" / "manifest.parquet"),
))

PORT = int(os.environ.get("CURATOR_PORT", "8002"))
HOST = os.environ.get("CURATOR_HOST", "127.0.0.1")
MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 50

# Union of every column we might want to sort by. The actual ALLOWED_SORT_COLUMNS
# is the intersection of this set with the loaded parquet's schema (computed at
# startup). Anything not in the set OR not in the parquet falls back to
# utterance_id to prevent injection-via-ORDER-BY.
ALLOWED_SORT_COLUMNS_BASE = {
    "utterance_id",
    "source",
    "duration_sec",
    "has_text",
    "qf_dnsmos_ovrl",
    "qf_vad_speech_ratio",
    "qf_lid",
    "qf_lid_is_hu_prob",
    "qf_silence_ratio",
    "qf_is_clipped",
    # PoC parquet (manifest_poc_100h.parquet) additions:
    "qc_pairwise_wer",
    "qc_exact_match",
}

# Union of every column we might want in the list view. Filtered to what the
# loaded parquet actually has at startup.
LIST_COLUMNS_BASE = [
    "utterance_id",
    "source",
    "duration_sec",
    "has_text",
    "text_preview",
    "qf_dnsmos_ovrl",
    "qf_vad_speech_ratio",
    "qf_lid",
    "qf_lid_is_hu_prob",
    "qf_is_clipped",
    "relative_audio_path",
    "audio_path",
    # Phase 2.6 boundary refinement: absolute path; we derive a relative
    # form server-side (see _add_refined_relative).
    "refined_audio_path",
    # PoC parquet additions — included unconditionally; the runtime filter
    # drops them on parquets that don't have them. The frontend's renderRow
    # ignores fields it doesn't expect, so this is safe.
    "qc_pairwise_wer",
    "qc_exact_match",
    "text_qwen_ft_greedy",
    "text_canary_v2_greedy",
]


def _add_refined_relative(row: dict) -> dict:
    """If refined_audio_path is present (absolute under DATA_ROOT), derive
    refined_relative_audio_path so the /audio/<rel_path> endpoint can serve it.
    Mutates and returns the row dict."""
    abs_path = row.get("refined_audio_path")
    if abs_path:
        root_str = str(DATA_ROOT) + "/"
        if abs_path.startswith(root_str):
            row["refined_relative_audio_path"] = abs_path[len(root_str):]
    return row


def _open_duckdb():
    """Open a DuckDB connection over the manifest parquet as a view."""
    try:
        import duckdb  # noqa: WPS433
    except ImportError:
        print("[error] duckdb not installed. Run:\n"
              "  pip install duckdb flask pyarrow",
              file=sys.stderr)
        raise

    if not MANIFEST_PARQUET.exists():
        print(f"[error] manifest parquet missing: {MANIFEST_PARQUET}\n"
              f"  Run: python bin/build_manifest_parquet.py first.",
              file=sys.stderr)
        raise SystemExit(2)

    con = duckdb.connect(":memory:")
    # DuckDB's read_parquet handles the path as a literal — no injection risk
    # since MANIFEST_PARQUET is a server-controlled constant.
    con.execute(
        f"CREATE VIEW manifest AS SELECT * FROM read_parquet('{MANIFEST_PARQUET}')"
    )
    return con


app = Flask(__name__, template_folder="templates", static_folder="static")
con = _open_duckdb()

# Discover which columns the loaded parquet actually has, then filter the
# allowed-sort and list-projection sets to that intersection. Lets the same
# Flask app serve both the base manifest and the PoC variant cleanly.
_present_columns = {
    d[0] for d in con.execute("SELECT * FROM manifest LIMIT 0").description
}
ALLOWED_SORT_COLUMNS = ALLOWED_SORT_COLUMNS_BASE & _present_columns
LIST_COLUMNS = [c for c in LIST_COLUMNS_BASE if c in _present_columns]
print(f"[curator] {len(_present_columns)} columns in parquet; "
      f"{len(LIST_COLUMNS)} list-view columns, "
      f"{len(ALLOWED_SORT_COLUMNS)} sortable columns",
      file=sys.stderr)


# ============================================================
# Filter builder — turns query params into a parameterised WHERE clause.
# ============================================================

def _build_where(args) -> tuple[str, list]:
    """Parse Flask request.args into (where_sql, params). Returns ('TRUE', [])
    when no filters are active."""
    where: list[str] = []
    params: list = []

    sources = args.getlist("source")
    if sources:
        placeholders = ",".join("?" for _ in sources)
        where.append(f"source IN ({placeholders})")
        params.extend(sources)

    has_text = args.get("has_text")
    if has_text == "yes":
        where.append("has_text = TRUE")
    elif has_text == "no":
        where.append("has_text = FALSE")

    for col, qmin, qmax in (
        ("duration_sec", "duration_min", "duration_max"),
        ("qf_dnsmos_ovrl", "dnsmos_min", "dnsmos_max"),
        ("qf_vad_speech_ratio", "vad_min", "vad_max"),
        ("qf_lid_is_hu_prob", "lid_hu_min", "lid_hu_max"),
        # PoC parquet additions; harmless on parquets that don't have these
        # columns IF the user doesn't submit the corresponding params.
        ("qc_pairwise_wer", "qc_wer_min", "qc_wer_max"),
    ):
        # Skip filters on columns the loaded parquet doesn't have.
        if col not in _present_columns:
            continue
        lo = args.get(qmin, type=float)
        hi = args.get(qmax, type=float)
        if lo is not None:
            where.append(f"{col} >= ?")
            params.append(lo)
        if hi is not None:
            where.append(f"{col} <= ?")
            params.append(hi)

    if "qc_exact_match" in _present_columns:
        qc_exact = args.get("qc_exact")
        if qc_exact == "yes":
            where.append("qc_exact_match = TRUE")
        elif qc_exact == "no":
            where.append("(qc_exact_match = FALSE OR qc_exact_match IS NULL)")

    lid = args.get("lid")
    if lid == "hu":
        where.append("qf_lid = 'hu'")
    elif lid == "non_hu":
        where.append("qf_lid IS NOT NULL AND qf_lid != 'hu'")
    elif lid == "unknown":
        where.append("qf_lid IS NULL")

    is_clipped = args.get("is_clipped")
    if is_clipped == "yes":
        where.append("qf_is_clipped = TRUE")
    elif is_clipped == "no":
        where.append("qf_is_clipped = FALSE")

    halluc = args.get("halluc")
    if halluc == "yes":
        where.append("qf_any_hallucination_flag = TRUE")
    elif halluc == "no":
        where.append("(qf_any_hallucination_flag = FALSE OR qf_any_hallucination_flag IS NULL)")

    # Phase 2.6 foreign-content filter. Combines two underlying flags:
    #   qf_foreign_prefix_sec > 0  -> leading non-HU before HU
    #   qf_whole_non_hu = TRUE     -> whole-clip LID lands on non-HU
    foreign = args.get("foreign")
    if foreign and "qf_foreign_prefix_sec" in _present_columns:
        if foreign == "with_prefix":
            where.append("qf_foreign_prefix_sec > 0")
        elif foreign == "whole_non_hu":
            where.append("qf_whole_non_hu = TRUE")
        elif foreign == "any_foreign":
            where.append("(qf_foreign_prefix_sec > 0 OR qf_whole_non_hu = TRUE)")
        elif foreign == "clean_hu":
            where.append("(qf_foreign_prefix_sec IS NULL OR qf_foreign_prefix_sec = 0) "
                         "AND (qf_whole_non_hu IS NULL OR qf_whole_non_hu = FALSE)")

    # Optional: minimum foreign-prefix length in seconds (e.g. 2 -> only
    # clips with ≥2 sec of leading foreign speech).
    if "qf_foreign_prefix_sec" in _present_columns:
        fp_min = args.get("foreign_prefix_min", type=float)
        if fp_min is not None:
            where.append("qf_foreign_prefix_sec >= ?")
            params.append(fp_min)

    q = (args.get("q") or "").strip()
    if q:
        like = f"%{q}%"
        where.append("(utterance_id ILIKE ? OR text_source_caption ILIKE ? "
                     "OR text_whisper_large_v3_pseudo ILIKE ?)")
        params.extend([like, like, like])

    return (" AND ".join(where) if where else "TRUE"), params


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    row = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(duration_sec), 0) / 3600 FROM manifest"
    ).fetchone()
    return jsonify({
        "total_rows": int(row[0]),
        "total_hours": round(float(row[1]), 2),
        "parquet_name": MANIFEST_PARQUET.name,
        # Used by the frontend to show/hide PoC-specific filter widgets.
        "has_qc_columns": "qc_exact_match" in _present_columns,
    })


@app.route("/api/sources")
def api_sources():
    rows = con.execute(
        "SELECT source, COUNT(*) AS n FROM manifest GROUP BY source ORDER BY source"
    ).fetchall()
    return jsonify([{"source": r[0], "n": int(r[1])} for r in rows])


@app.route("/api/lid_values")
def api_lid_values():
    """For UI: the actual LID codes present in the corpus, with counts."""
    rows = con.execute(
        "SELECT qf_lid, COUNT(*) FROM manifest "
        "WHERE qf_lid IS NOT NULL GROUP BY qf_lid ORDER BY 2 DESC"
    ).fetchall()
    return jsonify([{"lid": r[0], "n": int(r[1])} for r in rows])


@app.route("/api/rows")
def api_rows():
    page = max(1, request.args.get("page", default=1, type=int) or 1)
    page_size = min(
        MAX_PAGE_SIZE,
        max(1, request.args.get("page_size", default=DEFAULT_PAGE_SIZE, type=int) or DEFAULT_PAGE_SIZE),
    )

    sort_col = request.args.get("sort") or "utterance_id"
    if sort_col not in ALLOWED_SORT_COLUMNS:
        sort_col = "utterance_id"
    sort_dir = (request.args.get("dir") or "asc").upper()
    if sort_dir not in ("ASC", "DESC"):
        sort_dir = "ASC"
    # NULLS LAST for DESC, NULLS LAST for ASC too — sparse quality columns
    # would otherwise dominate the first page with empty rows.
    nulls = "NULLS LAST"

    where_sql, params = _build_where(request.args)

    total = con.execute(
        f"SELECT COUNT(*) FROM manifest WHERE {where_sql}", params
    ).fetchone()[0]

    offset = (page - 1) * page_size
    cols_sql = ", ".join(LIST_COLUMNS)
    rows = con.execute(
        f"SELECT {cols_sql} FROM manifest WHERE {where_sql} "
        f"ORDER BY {sort_col} {sort_dir} {nulls} LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()

    return jsonify({
        "total": int(total),
        "page": page,
        "page_size": page_size,
        "sort": sort_col,
        "dir": sort_dir.lower(),
        "rows": [_add_refined_relative(dict(zip(LIST_COLUMNS, r))) for r in rows],
    })


@app.route("/api/row/<path:utterance_id>")
def api_row_detail(utterance_id: str):
    # path: converter allows slashes — utterance_ids like
    # "mosel/20090112-0900-PLENARY-10_hu_50" contain them.
    cursor = con.execute(
        "SELECT * FROM manifest WHERE utterance_id = ? LIMIT 1", [utterance_id]
    )
    cols = [d[0] for d in cursor.description]
    row = cursor.fetchone()
    if not row:
        abort(404)
    return jsonify(_add_refined_relative(dict(zip(cols, row))))


@app.route("/audio/<path:rel_path>")
def audio(rel_path: str):
    """Stream an audio file under DATA_ROOT. Uses Flask's send_file with
    conditional=True for HTTP Range request support (browser seekable audio)."""
    # Strip any leading slashes to keep the join inside DATA_ROOT.
    clean = rel_path.lstrip("/")
    candidate = (DATA_ROOT / clean).resolve()
    try:
        candidate.relative_to(DATA_ROOT.resolve())
    except ValueError:
        abort(403)
    if not candidate.is_file():
        abort(404)
    # Flask infers mime from extension via mimetypes — handles .ogg/.wav/.mp3/.flac.
    return send_file(str(candidate), conditional=True)


def main() -> int:
    print(f"[curator] parquet: {MANIFEST_PARQUET}", file=sys.stderr)
    print(f"[curator] data root: {DATA_ROOT}", file=sys.stderr)
    print(f"[curator] http://{HOST}:{PORT}", file=sys.stderr)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
