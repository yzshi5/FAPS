#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REGRESSION_ROOT}/reg_train/OFM_prior_GP_matern.py" \
  --save-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --checkpoint-name "GP_matern_epoch_500.pt" \
  --figure-dir "${REGRESSION_ROOT}/outputs/OFM/GP_matern" \
  --device "cuda:0" \
  --seed 22 \
  --train-samples 20000 \
  --n-x 128 \
  --modes 32 \
  --width 256 \
  --mlp-width 128 \
  --kernel-length 0.01 \
  --kernel-variance 1.0 \
  --kernel-nu 0.5 \
  --gp-length-scale 0.3 \
  --gp-variance 1.0 \
  --gp-nu 1.5 \
  --epochs 500 \
  --sigma-min 1e-4 \
  --batch-size 1024 \
  --learning-rate 5e-4 \
  --eta-min 1e-6 \
  --super-n-x 512 \
  --super-train-samples 3000
