#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REGRESSION_ROOT}/reg_eval/2D_reg.py" \
  --dataset "bh" \
  --model-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --flat-checkpoint-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --checkpoint-name "bh_epoch_300.pt" \
  --results-dir "${REGRESSION_ROOT}/outputs/Regression_results/BH_reg" \
  --device "cuda:0" \
  --seed 22 \
  --n-x 64 \
  --n-y 64 \
  --modes 24 \
  --width 128 \
  --mlp-width 128 \
  --epochs 300 \
  --sigma-min 1e-4 \
  --kernel-length 0.01 \
  --kernel-variance 1.0 \
  --kernel-nu 0.5 \
  --test-data-path "/net/wintermute/scratch/agao3/OpFlow/Unoflow/m_1_1_40_all.npy" \
  --test-sample-idx 100 \
  --n-observations 256 \
  --noise-level 1e-4\
  --n-samples 32 \
  --annealing-steps 40 \
  --ode-steps 20 \
  --langevin-steps 50 \
  --langevin-lr 1e-4 \
  --tau 1.0 \
  --low-rank-cov-rank 32 \
  --low-rank-cov-samples 256 \
  --low-rank-cov-ode-steps 20 \
  --save-prefix "bh_reg"
