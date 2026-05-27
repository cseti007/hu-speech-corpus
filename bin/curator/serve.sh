#!/usr/bin/env bash
# Launch the corpus curator (Flask + DuckDB over a parquet file).
#
# Usage:
#   bash bin/curator/serve.sh                    # default: manifest.parquet
#   bash bin/curator/serve.sh poc                # alias: manifest_poc_100h.parquet
#   bash bin/curator/serve.sh some_subset.parquet  # relative to manifests/ dir
#   bash bin/curator/serve.sh /abs/path/file.parquet  # absolute path
#
# Env vars:
#   CURATOR_PORT      listen port (default 8002)
#   CURATOR_HOST      bind host (default 127.0.0.1)
#   HU_CORPUS_ROOT    data root — REQUIRED (no default)
#   PYTHON            python interpreter (default: $(which python))
#   CURATOR_PARQUET   set directly to override the arg-based resolution

set -euo pipefail

PYTHON="${PYTHON:-$(which python)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -z "${HU_CORPUS_ROOT:-}" ]]; then
  echo "[error] HU_CORPUS_ROOT env var not set. Point it at your corpus data root." >&2
  exit 2
fi

MANIFESTS_DIR="${HU_CORPUS_ROOT}/processed/manifests"
PARQUETS_DIR="${HU_CORPUS_ROOT}/processed/parquets"
ARG="${1:-}"

# The new generic curator (2026-05-26 rewrite) discovers all parquets in
# both MANIFESTS_DIR and PARQUETS_DIR via the file picker UI. CURATOR_PARQUET
# just selects the INITIAL active parquet on startup.

# Named aliases for common parquets. Extend as new variants land.
case "${ARG}" in
  "")
    # Default: smoke.parquet (small, fast to load, good first impression).
    CURATOR_PARQUET="${PARQUETS_DIR}/smoke.parquet"
    ;;
  smoke)
    CURATOR_PARQUET="${PARQUETS_DIR}/smoke.parquet"
    ;;
  dev)
    CURATOR_PARQUET="${PARQUETS_DIR}/dev.parquet"
    ;;
  poc|poc_100h)
    CURATOR_PARQUET="${MANIFESTS_DIR}/manifest_poc_100h.parquet"
    ;;
  multi|multisource|poc_multi)
    CURATOR_PARQUET="${MANIFESTS_DIR}/manifest_poc_multisource.parquet"
    ;;
  v5)
    CURATOR_PARQUET="${MANIFESTS_DIR}/manifest_v5.parquet"
    ;;
  /*)
    # Absolute path: use as-is
    CURATOR_PARQUET="${ARG}"
    ;;
  *)
    # Treat as a filename — try parquets/ first, then manifests/.
    if [[ -f "${PARQUETS_DIR}/${ARG}" ]]; then
      CURATOR_PARQUET="${PARQUETS_DIR}/${ARG}"
    else
      CURATOR_PARQUET="${MANIFESTS_DIR}/${ARG}"
    fi
    ;;
esac

if [[ ! -f "${CURATOR_PARQUET}" ]]; then
  echo "[error] parquet not found: ${CURATOR_PARQUET}" >&2
  echo "        Available parquets:" >&2
  ls "${PARQUETS_DIR}"/*.parquet 2>/dev/null | sed 's|^|          |' >&2 || true
  ls "${MANIFESTS_DIR}"/*.parquet 2>/dev/null | sed 's|^|          |' >&2 || true
  echo "        Switch between them via the file-picker dropdown once running." >&2
  exit 2
fi

if ! "${PYTHON}" -c "import flask, duckdb, pyarrow" 2>/dev/null; then
  echo "[error] missing dependencies. Install with:" >&2
  echo "        pip install flask duckdb pyarrow" >&2
  exit 2
fi

export CURATOR_PARQUET
echo "[curator] loading: ${CURATOR_PARQUET}" >&2
exec "${PYTHON}" "${SCRIPT_DIR}/app.py"
