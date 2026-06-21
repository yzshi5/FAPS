#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REGRESSION_ROOT}/reg_train/OFM_prior_2D.py" \
  --dataset "bh" \
  --data-path "/net/wintermute/scratch/agao3/OpFlow/Unoflow/m_1_1_40_all.npy" \
  --save-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --checkpoint-name "bh_epoch_300.pt" \
  --figure-dir "${REGRESSION_ROOT}/outputs/OFM/BH" \
  --device "cuda:0" \
  --seed 22 \
  --rotation-augment \
  --n-x 64 \
  --n-y 64 \
  --modes 24 \
  --width 128 \
  --mlp-width 128 \
  --kernel-length 0.01 \
  --kernel-variance 1.0 \
  --kernel-nu 0.5 \
  --epochs 300 \
  --sigma-min 1e-4 \
  --batch-size 96 \
  --learning-rate 5e-4 \
  --eta-min 1e-6 \
  --n-eval 20 \
  --super-n-x 96 \
  --super-n-y 96
