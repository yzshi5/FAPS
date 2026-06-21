#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PDE_ROOT}/pde_train/train_PDE_surrogate_2D.py" \
  --pde "helmholtz" \
  --data-path "/oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse_npy/helmholtz_flow.npy" \
  --save-dir "${PDE_ROOT}/checkpoints/PDE_surrogate" \
  --checkpoint-name "helmholtz_forward.pt" \
  --history-name "helmholtz_history.npy" \
  --device "cuda:0" \
  --seed 42 \
  --epochs 50 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --modes 48 \
  --hidden-channels 64 \
  --projection-channels 128 \
  --n-layers 4
