#!/usr/bin/env python3
"""Generic curator backend: Flask + DuckDB over any v5-schema parquet.

Rewrite of the original curator (`app_legacy.py`) that hardcoded the v4
flat `qf_*` schema. This version supports any parquet:
  - Lists available parquets from processed/parquets/ and processed/manifests/
  - Returns the full schema (flat columns + STRUCT subfields via dot-notation)
  - Accepts dynamic ?columns=col1,quality_flags.dnsmos_ovrl,... in /api/rows
  - Switches active parquet at runtime via ?parquet=name URL parameter

Endpoints:
  GET /                        single-page UI
  GET /api/parquets            list of available parquets in known dirs
  GET /api/schema              columns + STRUCT subfields for active parquet
  GET /api/stats               { total_rows, total_hours, parquet_name }
  GET /api/sources             distinct values of `source` (if present)
  GET /api/rows?columns=...&...filters    paginated rows w/ requested cols
  GET /api/row/<utterance_id>  full row detail (every column + struct)
  GET /audio/<rel_path>        stream audio file with HTTP Range support

Run via the launcher:
  bash bin/curator/serve.sh
  # browser: http://localhost:8002

Dependencies: flask, duckdb, pyarrow. The hu-speech-corpus conda env has these.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import RLock

from flask import Flask, abort, jsonify, render_template, request, send_file

_DATA_ROOT_ENV = os.environ.get("HU_CORPUS_ROOT")
if not _DATA_ROOT_ENV:
    raise SystemExit(
        "HU_CORPUS_ROOT env var is not set. Export it to your corpus storage "
        "root before running the curator."
    )
DATA_ROOT = Path(_DATA_ROOT_ENV)

# Where we scan for parquets. Two directories: the curated parquets
# (smoke, dev, future test) and the canonical manifest parquets.
PARQUET_DIRS = [
    DATA_ROOT / "processed" / "parquets",
    DATA_ROOT / "processed" / "manifests",
]

# Default parquet on startup. Can be overridden by CURATOR_PARQUET env var
# or by selecting a different parquet via ?parquet= or POST /api/load.
DEFAULT_PARQUET = Path(os.environ.get(
    "CURATOR_PARQUET",
    str(DATA_ROOT / "processed" / "parquets" / "smoke.parquet"),
))

PORT = int(os.environ.get("CURATOR_PORT", "8002"))
HOST = os.environ.get("CURATOR_HOST", "127.0.0.1")
MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 50

# Default columns shown on first load (overridable via Columns dialog;
# persisted per-parquet in browser localStorage). These should resolve on
# most v5-schema parquets; missing ones are silently skipped.
DEFAULT_COLUMNS = [
    "utterance_id",
    "source",
    "duration_sec",
    "audio_path",
    "transcripts.source_caption",
]

# Columns that must always be fetched server-side even if the user didn't
# pick them, because cell rendering needs them (e.g. audio playback needs
# audio_path / source / parquet_row_index for the inline player dispatcher).
AUDIO_HELPER_COLUMNS = [
    "audio_path", "refined_audio_path", "source",
    "parquet_row_index",  # vp_labeled rows decode via /audio_parquet
]


# ============================================================
# State: parquet-keyed cache of DuckDB connections + schemas
# ============================================================

class CuratorState:
    """Per-parquet DuckDB connection cache. Lets the UI switch parquets at
    runtime without reopening the connection on every request."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._cache: dict[str, dict] = {}  # path -> {con, schema, columns}
        self._active: str | None = None

    def list_parquets(self) -> list[dict]:
        """Scan known dirs, return sorted list of parquet metadata."""
        out: list[dict] = []
        for d in PARQUET_DIRS:
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.parquet")):
                out.append({
                    "name": p.name,
                    "path": str(p),
                    "dir": d.name,
                    "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
                })
        return out

    def open(self, path: Path) -> dict:
        """Open (or fetch cached) handle for `path`. Returns {con, schema, columns}."""
        with self._lock:
            key = str(path.resolve())
            if key in self._cache:
                return self._cache[key]
            if not path.exists():
                raise FileNotFoundError(path)
            import duckdb
            con = duckdb.connect(":memory:")
            con.execute(
                "CREATE VIEW manifest AS "
                f"SELECT * FROM read_parquet('{path}')"
            )
            schema = self._discover_schema(con)
            entry = {"con": con, "schema": schema, "path": path}
            self._cache[key] = entry
            return entry

    def _discover_schema(self, con) -> list[dict]:
        """Return list of {name, type, subfields?} for every top-level column.

        STRUCT columns get a `subfields` list of {name, type} entries — we
        surface those via dot notation in the column picker."""
        desc = con.execute("SELECT * FROM manifest LIMIT 0").description
        out: list[dict] = []
        for d in desc:
            col_name = d[0]
            # DuckDB returns DuckDBPyType objects, not str; coerce to text.
            col_type_str = str(d[1])
            entry: dict = {"name": col_name, "type": col_type_str}
            if col_type_str.upper().startswith("STRUCT("):
                sub = con.execute(
                    f"DESCRIBE SELECT {col_name}.* FROM manifest LIMIT 0"
                ).fetchall()
                entry["subfields"] = [
                    {"name": r[0], "type": str(r[1])} for r in sub
                ]
            out.append(entry)
        return out

    def set_active(self, path: Path) -> dict:
        entry = self.open(path)
        with self._lock:
            self._active = str(path.resolve())
        return entry

    def active(self) -> dict:
        with self._lock:
            if self._active is None:
                raise RuntimeError("no active parquet")
            return self._cache[self._active]

    def active_path(self) -> Path:
        return self.active()["path"]


STATE = CuratorState()


def _resolve_parquet_arg(arg: str | None) -> Path:
    """Resolve a ?parquet= request arg to a concrete path.

    Accepts either a bare filename (looked up in PARQUET_DIRS) or an absolute
    path (must be inside one of the known dirs, no escape). None falls back
    to the currently-active path."""
    if not arg:
        return STATE.active_path()
    p = Path(arg)
    if not p.is_absolute():
        for d in PARQUET_DIRS:
            cand = d / arg
            if cand.is_file():
                return cand.resolve()
        abort(404, description=f"parquet not found: {arg}")
    p = p.resolve()
    if not any(str(p).startswith(str(d.resolve()) + "/") for d in PARQUET_DIRS):
        abort(403, description=f"parquet outside allowed dirs: {p}")
    if not p.is_file():
        abort(404)
    return p


def _entry_for_request(args) -> dict:
    """Resolve the parquet specified by ?parquet=, open if needed, return state."""
    p = _resolve_parquet_arg(args.get("parquet"))
    return STATE.open(p)


# ============================================================
# Filter builder — generic, applies only when columns exist
# ============================================================

def _build_where(entry: dict, args) -> tuple[str, list]:
    """Parse request.args into (where_sql, params).

    Only references columns that exist in the active parquet's schema
    (top-level or STRUCT subfield). Anything missing is silently skipped."""
    schema = entry["schema"]
    flat_cols = {c["name"] for c in schema}
    struct_subfields: dict[str, set] = {}
    for c in schema:
        if c.get("subfields"):
            struct_subfields[c["name"]] = {s["name"] for s in c["subfields"]}

    def resolve(col_path: str) -> str | None:
        """Resolve a column path ('source', 'quality_flags.dnsmos_ovrl') to
        a DuckDB-quoted expression. Returns None if the column doesn't exist."""
        if "." in col_path:
            top, sub = col_path.split(".", 1)
            if top in struct_subfields and sub in struct_subfields[top]:
                return f"{top}.{sub}"
            return None
        if col_path in flat_cols:
            return col_path
        return None

    where: list[str] = []
    params: list = []

    sources = args.getlist("source")
    if sources and "source" in flat_cols:
        placeholders = ",".join("?" for _ in sources)
        where.append(f"source IN ({placeholders})")
        params.extend(sources)

    # Free text search across utterance_id and any string text-preview column.
    q = (args.get("q") or "").strip()
    if q:
        like = f"%{q}%"
        text_targets = []
        if "utterance_id" in flat_cols:
            text_targets.append("utterance_id ILIKE ?")
            params.append(like)
        # Also search transcripts.* subfields if present
        for c in schema:
            if c["name"] == "transcripts" and c.get("subfields"):
                for sf in c["subfields"]:
                    text_targets.append(f"transcripts.{sf['name']} ILIKE ?")
                    params.append(like)
                break
        if text_targets:
            where.append("(" + " OR ".join(text_targets) + ")")

    # Generic per-column range filters via ?filter_min[<col>]=...&filter_max[<col>]=...
    # E.g. filter_min[duration_sec]=3 & filter_max[duration_sec]=30
    for key, value in args.items():
        if not value:
            continue
        if key.startswith("filter_min[") and key.endswith("]"):
            col = key[len("filter_min["):-1]
            expr = resolve(col)
            if expr is not None:
                where.append(f"{expr} >= ?")
                params.append(float(value))
        elif key.startswith("filter_max[") and key.endswith("]"):
            col = key[len("filter_max["):-1]
            expr = resolve(col)
            if expr is not None:
                where.append(f"{expr} <= ?")
                params.append(float(value))
        elif key.startswith("filter_eq[") and key.endswith("]"):
            col = key[len("filter_eq["):-1]
            expr = resolve(col)
            if expr is not None:
                if value.lower() in ("true", "false"):
                    where.append(f"{expr} = {value.upper()}")
                else:
                    where.append(f"{expr} = ?")
                    params.append(value)

    return (" AND ".join(where) if where else "TRUE"), params


def _is_sortable_type(t: str) -> bool:
    """Boolean / numeric / string types are sortable. STRUCT / LIST / MAP are not."""
    if not isinstance(t, str):
        return False
    tu = t.upper()
    for bad in ("STRUCT(", "LIST(", "MAP(", "[]"):
        if bad in tu:
            return False
    return True


def _resolve_sort(entry: dict, sort_arg: str) -> str:
    """Validate sort column. Default to utterance_id if invalid / missing."""
    schema = entry["schema"]
    flat = {c["name"]: c for c in schema}
    if sort_arg in flat and _is_sortable_type(flat[sort_arg]["type"]):
        return sort_arg
    if "." in sort_arg:
        top, sub = sort_arg.split(".", 1)
        if top in flat and flat[top].get("subfields"):
            for sf in flat[top]["subfields"]:
                if sf["name"] == sub and _is_sortable_type(sf["type"]):
                    return f"{top}.{sub}"
    if "utterance_id" in flat:
        return "utterance_id"
    return schema[0]["name"]


# ============================================================
# Flask app
# ============================================================

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.before_request
def _ensure_active():
    """First request: activate DEFAULT_PARQUET if no parquet is loaded yet.
    Cheap no-op on subsequent requests (cached in STATE)."""
    try:
        STATE.active()
    except RuntimeError:
        if DEFAULT_PARQUET.exists():
            STATE.set_active(DEFAULT_PARQUET)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/parquets")
def api_parquets():
    """Return discovered parquets + active one's path."""
    return jsonify({
        "parquets": STATE.list_parquets(),
        "active": str(STATE.active_path()) if STATE._active else None,
    })


@app.route("/api/load")
def api_load():
    """Activate a parquet by name (filename within PARQUET_DIRS) or absolute path.

    GET so we can use a simple URL like /api/load?parquet=smoke.parquet."""
    p = _resolve_parquet_arg(request.args.get("parquet"))
    STATE.set_active(p)
    return jsonify({"active": str(p), "schema": STATE.active()["schema"]})


@app.route("/api/schema")
def api_schema():
    entry = _entry_for_request(request.args)
    return jsonify({
        "parquet": str(entry["path"]),
        "columns": entry["schema"],
        "default_columns": DEFAULT_COLUMNS,
    })


@app.route("/api/stats")
def api_stats():
    entry = _entry_for_request(request.args)
    con = entry["con"]
    flat_cols = {c["name"] for c in entry["schema"]}
    has_duration = "duration_sec" in flat_cols
    sel = "COUNT(*)"
    if has_duration:
        sel += ", COALESCE(SUM(duration_sec), 0) / 3600.0"
    row = con.execute(f"SELECT {sel} FROM manifest").fetchone()
    out = {
        "total_rows": int(row[0]),
        "parquet_name": entry["path"].name,
        "parquet_path": str(entry["path"]),
    }
    if has_duration:
        out["total_hours"] = round(float(row[1]), 2)
    return jsonify(out)


@app.route("/api/sources")
def api_sources():
    entry = _entry_for_request(request.args)
    flat_cols = {c["name"] for c in entry["schema"]}
    if "source" not in flat_cols:
        return jsonify([])
    rows = entry["con"].execute(
        "SELECT source, COUNT(*) FROM manifest GROUP BY source ORDER BY source"
    ).fetchall()
    return jsonify([{"source": r[0], "n": int(r[1])} for r in rows])


@app.route("/api/rows")
def api_rows():
    entry = _entry_for_request(request.args)
    con = entry["con"]
    schema = entry["schema"]
    flat_cols = {c["name"]: c for c in schema}
    struct_subfields: dict[str, set] = {
        c["name"]: {s["name"] for s in c.get("subfields") or []}
        for c in schema if c.get("subfields")
    }

    page = max(1, request.args.get("page", default=1, type=int) or 1)
    page_size = min(
        MAX_PAGE_SIZE,
        max(1, request.args.get("page_size", default=DEFAULT_PAGE_SIZE,
                                type=int) or DEFAULT_PAGE_SIZE),
    )

    # Columns: user-requested ?columns=col1,col2,quality_flags.dnsmos_ovrl,...
    cols_param = request.args.get("columns") or ""
    requested = [c.strip() for c in cols_param.split(",") if c.strip()]
    if not requested:
        requested = list(DEFAULT_COLUMNS)
    # Always include audio helpers (for inline player) when present.
    for h in AUDIO_HELPER_COLUMNS:
        if h in flat_cols and h not in requested:
            requested.append(h)
    # Always include utterance_id for row-detail / focus.
    if "utterance_id" in flat_cols and "utterance_id" not in requested:
        requested.insert(0, "utterance_id")

    # Build SELECT list. STRUCT subfields get aliased to the dotted name.
    select_exprs: list[str] = []
    out_cols: list[str] = []
    for path in requested:
        if "." in path:
            top, sub = path.split(".", 1)
            if top in struct_subfields and sub in struct_subfields[top]:
                select_exprs.append(f'{top}.{sub} AS "{path}"')
                out_cols.append(path)
                continue
        if path in flat_cols:
            select_exprs.append(path)
            out_cols.append(path)
        # else: silently drop unknown paths
    if not select_exprs:
        abort(400, description="no resolvable columns requested")

    sort_col = _resolve_sort(entry, request.args.get("sort") or "utterance_id")
    sort_dir = (request.args.get("dir") or "asc").upper()
    if sort_dir not in ("ASC", "DESC"):
        sort_dir = "ASC"

    where_sql, params = _build_where(entry, request.args)

    total = con.execute(
        f"SELECT COUNT(*) FROM manifest WHERE {where_sql}", params
    ).fetchone()[0]

    offset = (page - 1) * page_size
    sql = (
        f"SELECT {', '.join(select_exprs)} FROM manifest "
        f"WHERE {where_sql} "
        f"ORDER BY {sort_col} {sort_dir} NULLS LAST LIMIT ? OFFSET ?"
    )
    cursor = con.execute(sql, params + [page_size, offset])
    cursor_cols = [d[0] for d in cursor.description]
    rows_raw = cursor.fetchall()

    rows_out = []
    for r in rows_raw:
        row = dict(zip(cursor_cols, r))
        # If source + audio_path present, derive a relative audio path
        # under DATA_ROOT so the /audio/<rel_path> endpoint can serve it.
        ap = row.get("audio_path")
        if ap:
            root_str = str(DATA_ROOT) + "/"
            if isinstance(ap, str) and ap.startswith(root_str):
                row["_relative_audio_path"] = ap[len(root_str):]
        ra = row.get("refined_audio_path")
        if ra:
            root_str = str(DATA_ROOT) + "/"
            if isinstance(ra, str) and ra.startswith(root_str):
                row["_relative_refined_audio_path"] = ra[len(root_str):]
        rows_out.append(row)

    return jsonify({
        "total": int(total),
        "page": page,
        "page_size": page_size,
        "sort": sort_col,
        "dir": sort_dir.lower(),
        "columns": out_cols,
        "rows": rows_out,
    })


@app.route("/api/row/<path:utterance_id>")
def api_row_detail(utterance_id: str):
    entry = _entry_for_request(request.args)
    con = entry["con"]
    cursor = con.execute(
        "SELECT * FROM manifest WHERE utterance_id = ? LIMIT 1", [utterance_id]
    )
    cols = [d[0] for d in cursor.description]
    row = cursor.fetchone()
    if not row:
        abort(404)
    return jsonify(dict(zip(cols, row)))


@app.route("/audio/<path:rel_path>")
def audio(rel_path: str):
    """Stream an audio file under DATA_ROOT (HTTP Range supported)."""
    clean = rel_path.lstrip("/")
    candidate = (DATA_ROOT / clean).resolve()
    try:
        candidate.relative_to(DATA_ROOT.resolve())
    except ValueError:
        abort(403)
    if not candidate.is_file():
        abort(404)
    return send_file(str(candidate), conditional=True)


# Cache parquet shards we've read for audio extraction (vp_labeled).
# Keyed by absolute path; each shard is ~600 MB so we keep at most a few.
import io as _io
from collections import OrderedDict
_PARQUET_AUDIO_CACHE: "OrderedDict[str, list]" = OrderedDict()
_PARQUET_AUDIO_CACHE_MAX = 4


def _parquet_audio_col(shard_path: Path) -> list:
    """Return the cached 'audio' column (list of {bytes, path}) for a shard."""
    key = str(shard_path)
    if key in _PARQUET_AUDIO_CACHE:
        _PARQUET_AUDIO_CACHE.move_to_end(key)
        return _PARQUET_AUDIO_CACHE[key]
    import pyarrow.parquet as pq
    table = pq.read_table(shard_path, columns=["audio"])
    col = table.column("audio").to_pylist()
    _PARQUET_AUDIO_CACHE[key] = col
    while len(_PARQUET_AUDIO_CACHE) > _PARQUET_AUDIO_CACHE_MAX:
        _PARQUET_AUDIO_CACHE.popitem(last=False)
    return col


@app.route("/audio_parquet")
def audio_parquet():
    """Serve audio bytes extracted from a parquet shard (vp_labeled rows).

    Required query params:
      path        absolute path to the parquet shard (must be under DATA_ROOT)
      row_index   row index within the shard's `audio` column

    Cached shard reads (LRU, up to 4 shards) — first hit on a shard is slow
    (~600 MB read), subsequent rows are instant."""
    shard_str = request.args.get("path")
    row_index_s = request.args.get("row_index")
    if not shard_str or row_index_s is None:
        abort(400, description="path and row_index required")
    try:
        row_index = int(float(row_index_s))
    except (TypeError, ValueError):
        abort(400, description="row_index must be an integer")
    shard = Path(shard_str).resolve()
    try:
        shard.relative_to(DATA_ROOT.resolve())
    except ValueError:
        abort(403)
    if not shard.is_file():
        abort(404)
    col = _parquet_audio_col(shard)
    if row_index < 0 or row_index >= len(col):
        abort(404, description=f"row_index {row_index} out of range (0..{len(col)-1})")
    entry = col[row_index]
    if entry is None or "bytes" not in entry:
        abort(404, description="row has no audio bytes")
    audio_bytes = entry["bytes"]
    # vp_labeled is WAV (RIFF magic). Set Content-Type accordingly so the
    # browser knows how to decode without sniffing.
    return send_file(
        _io.BytesIO(audio_bytes),
        mimetype="audio/wav",
        conditional=False,  # in-memory, no Range needed for small clips
    )


def main() -> int:
    print(f"[curator] default parquet: {DEFAULT_PARQUET}", file=sys.stderr)
    print(f"[curator] data root: {DATA_ROOT}", file=sys.stderr)
    print(f"[curator] http://{HOST}:{PORT}", file=sys.stderr)
    if DEFAULT_PARQUET.exists():
        STATE.set_active(DEFAULT_PARQUET)
        print(f"[curator] activated default parquet ({len(STATE.active()['schema'])} cols)",
              file=sys.stderr)
    else:
        print(f"[curator] default parquet missing; pick one via UI", file=sys.stderr)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
