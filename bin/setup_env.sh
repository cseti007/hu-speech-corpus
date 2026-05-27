#!/usr/bin/env bash
# Set up the dedicated conda env + clone facebookresearch/voxpopuli for the
# VoxPopuli session segmentation step.
#
# Idempotent: safe to re-run. Skips env creation and clone if already present.

set -euo pipefail

ENV_NAME="hu-speech-corpus"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOXPOPULI_DIR="${PROJECT_ROOT}/external/voxpopuli"
CONDA_ROOT="/media/cseti/datassd/conda/miniconda3"
CONDA_BIN="${CONDA_ROOT}/bin/conda"
ENV_PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
ENV_PIP="${CONDA_ROOT}/envs/${ENV_NAME}/bin/pip"

echo "[setup_env] project root: ${PROJECT_ROOT}"
echo "[setup_env] target env:   ${ENV_NAME}"

# --- 1. Create conda env if missing
if [[ -x "${ENV_PY}" ]]; then
    echo "[setup_env] env '${ENV_NAME}' already exists -> skip create"
else
    echo "[setup_env] creating conda env '${ENV_NAME}' (python 3.11)..."
    "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.11 -y
fi

# --- 2. Install Python dependencies (CPU-only torch/torchaudio)
echo "[setup_env] installing torch + torchaudio (CPU)..."
"${ENV_PIP}" install --upgrade pip
"${ENV_PIP}" install --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio

# torchaudio>=2.10 delegates ogg/vorbis decoding to torchcodec
"${ENV_PIP}" install torchcodec

echo "[setup_env] installing voxpopuli requirements + manifest-build + curator deps..."
"${ENV_PIP}" install \
    tqdm num2words edlib editdistance \
    pyyaml pyarrow pandas huggingface_hub \
    soundfile \
    flask duckdb

# --- 3. Clone facebookresearch/voxpopuli (shallow)
mkdir -p "$(dirname "${VOXPOPULI_DIR}")"
if [[ -d "${VOXPOPULI_DIR}/.git" ]]; then
    echo "[setup_env] voxpopuli repo already cloned at ${VOXPOPULI_DIR} -> skip"
else
    echo "[setup_env] cloning facebookresearch/voxpopuli -> ${VOXPOPULI_DIR}..."
    git clone --depth 1 https://github.com/facebookresearch/voxpopuli \
        "${VOXPOPULI_DIR}"
fi

# --- 3b. Patch torchaudio.datasets.utils.download_url removal (>=2.10)
TARGET_FILE="${VOXPOPULI_DIR}/voxpopuli/get_unlabelled_data.py"
if grep -q '^from torchaudio.datasets.utils import download_url' "${TARGET_FILE}"; then
    echo "[setup_env] patching get_unlabelled_data.py for torchaudio>=2.10..."
    python3 - <<PY
from pathlib import Path
p = Path("${TARGET_FILE}")
src = p.read_text()
old = "from torchaudio.datasets.utils import download_url"
new = (
    "# Shim for torchaudio.datasets.utils.download_url, removed in torchaudio>=2.10.\n"
    "# We only need to fetch one annotation file; urllib is sufficient.\n"
    "import urllib.request\n"
    "def download_url(url, root, filename):\n"
    "    out = Path(root) / filename\n"
    "    if not out.exists():\n"
    "        urllib.request.urlretrieve(url, out)"
)
p.write_text(src.replace(old, new))
PY
else
    echo "[setup_env] get_unlabelled_data.py already patched -> skip"
fi

# --- 4. Verify install
echo ""
echo "[setup_env] verifying install..."
cd "${VOXPOPULI_DIR}"
"${ENV_PY}" -c "import torch, torchaudio, tqdm; print(f'  torch        {torch.__version__}'); print(f'  torchaudio   {torchaudio.__version__}'); print(f'  tqdm         {tqdm.__version__}')"
"${ENV_PY}" -m voxpopuli.get_unlabelled_data --help > /dev/null && \
    echo "  voxpopuli.get_unlabelled_data is callable [OK]"

echo ""
echo "[setup_env] DONE. To use this env:"
echo "  ${ENV_PY}"
echo "or run scripts with:"
echo "  cd ${VOXPOPULI_DIR} && ${ENV_PY} -m voxpopuli.get_unlabelled_data ..."
