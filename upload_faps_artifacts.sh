#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${HF_REPO_ID:-Yaozhong/FAPS}"
REPO_TYPE="${HF_REPO_TYPE:-model}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export HF_REPO_ID="${REPO_ID}"
export HF_REPO_TYPE="${REPO_TYPE}"
python "${SCRIPT_DIR}/upload_faps_artifacts.py"
