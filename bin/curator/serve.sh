#!/usr/bin/env bash
# Launch the corpus curator (Flask + DuckDB over a parquet file).
#
# Usage:
#   bash bin/curator/serve.sh                          # default: smoke.parquet
#   bash bin/curator/serve.sh smoke|dev|test|train|v5  # named alias
#   bash bin/curator/serve.sh some_set.parquet         # any parquet in processed/parquets/
#   bash bin/curator/serve.sh /abs/path/file.parquet   # absolute path
#
# Env vars:
#   CURATOR_PORT       listen port (default 8002)
#   CURATOR_HOST       bind host (default 127.0.0.1)
#   HU_CORPUS_ROOT     data root — REQUIRED (no default)
#   PYTHON             python interpreter (default: $(which python))
#   CURATOR_PARQUET    set directly to override the arg-based resolution
#   CURATOR_PARQUETS   comma-separated allow-list of parquet basenames for
#                      the file picker. Empty = show every parquet in
#                      processed/parquets/ + processed/manifests/.
#                      Default below: the 5 canonical sets + full v5.

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

# Named aliases for the canonical sets.
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
  test)
    CURATOR_PARQUET="${PARQUETS_DIR}/test.parquet"
    ;;
  train)
    CURATOR_PARQUET="${PARQUETS_DIR}/train.parquet"
    ;;
  v5|manifest_v5)
    CURATOR_PARQUET="${PARQUETS_DIR}/manifest_v5.parquet"
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

# Default file-picker allow-list. The 5 canonical sets + full v5. Comment
# out / unset to show every parquet under processed/{parquets,manifests}/.
: "${CURATOR_PARQUETS:=smoke.parquet,test.parquet,train.parquet,dev.parquet,manifest_v5.parquet}"

export CURATOR_PARQUET CURATOR_PARQUETS
echo "[curator] loading: ${CURATOR_PARQUET}" >&2
if [[ -n "${CURATOR_PARQUETS:-}" ]]; then
  echo "[curator] picker filter: ${CURATOR_PARQUETS}" >&2
fi
exec "${PYTHON}" "${SCRIPT_DIR}/app.py"
