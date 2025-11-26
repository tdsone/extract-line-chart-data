#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup_local_env.sh

Prepares third-party dependencies required for running plextract locally:
  - Installs mmcv-full via OpenMIM (respecting torch/cu versions)
  - Clones ChartDete and installs its custom mmdetection fork

Assumes:
  - Script is run from the project root
  - Python 3.10 virtual environment with project dependencies is active
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${PROJECT_ROOT}/third_party"

if [[ ! -d "${THIRD_PARTY_DIR}" ]]; then
  mkdir -p "${THIRD_PARTY_DIR}"
fi

# Ensure openmim is available
if ! command -v mim >/dev/null 2>&1; then
  echo "Installing openmim..."
  uv pip install openmim
fi

echo "Installing mmcv-full via mim..."
uv run mim install mmcv-full

CHARTDETE_DIR="${THIRD_PARTY_DIR}/ChartDete"

if [[ -d "${CHARTDETE_DIR}" ]]; then
  echo "ChartDete already cloned. Pulling latest..."
  git -C "${CHARTDETE_DIR}" pull --ff-only
else
  echo "Cloning ChartDete..."
  git clone https://github.com/pengyu965/ChartDete.git "${CHARTDETE_DIR}"
fi

echo "Installing ChartDete mmdetection fork..."
uv pip install --no-build-isolation -e "${CHARTDETE_DIR}"

echo "Third-party setup complete."

