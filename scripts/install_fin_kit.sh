#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON:-python3}"

log() {
  printf "[fin-kit] %s\n" "$1"
}

log "Creating virtual environment at ${VENV_DIR}"
${PYTHON_BIN} -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

log "Upgrading pip"
pip install --upgrade pip wheel setuptools

if [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
  log "Installing project requirements"
  pip install -r "${REPO_ROOT}/requirements.txt"
fi

log "Installing fin-kit in editable mode"
pip install -e "${REPO_ROOT}"

log "Installation complete!"
log "Activate your environment with: source ${VENV_DIR}/bin/activate"
