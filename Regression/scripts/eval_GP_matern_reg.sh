#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REGRESSION_ROOT}/reg_eval/GP_matern_reg.py" \
  --model-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --flat-checkpoint-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --results-dir "${REGRESSION_ROOT}/outputs/Regression_results/GP_matern_reg" \
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
  --gp-length-scale 0.3 \
  --gp-variance 1.0 \
  --gp-nu 1.5 \
  --base-case-nx 128 \
  --case-seed 100 \
  --test-sample-idx 100 \
  --n-test 1000 \
  --n-obs 7 \
  --noise-level 0.01 \
  --n-samples 128 \
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
