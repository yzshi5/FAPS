#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REGRESSION_ROOT}/reg_eval/GP_gibbs_reg.py" \
  --model-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --flat-checkpoint-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --results-dir "${REGRESSION_ROOT}/outputs/Regression_results/GP_gibbs_reg" \
  --device "cuda:0" \
  --seed 100 \
  --modes 32 \
  --width 256 \
  --mlp-width 128 \
  --epochs 500 \
  --sigma-min 1e-4 \
  --kernel-length 0.01 \
  --kernel-variance 1.0 \
  --kernel-nu 0.5 \
  --n-x 512 \
  --gibbs-l0 0.05 \
  --gibbs-l1 0.25 \
  --gibbs-sigma 1.0 \
  --test-sample-idx 100 \
  --n-test 1000 \
  --n-obs 7 \
  --noise-level 0.01 \
  --n-samples 256 \
  --annealing-steps 40 \
  --ode-steps 20 \
  --langevin-steps 50 \
  --langevin-lr 1e-3 \
  --tau 1.0 \
  --anchor-std-base 0.05 \
  --anchor-std-scale 1.0 \
  --low-rank-cov-rank 32 \
  --low-rank-cov-samples 256 \
  --low-rank-cov-ode-steps 20
