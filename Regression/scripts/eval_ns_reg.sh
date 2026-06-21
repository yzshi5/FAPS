#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REGRESSION_ROOT}/reg_eval/2D_reg.py" \
  --dataset "ns" \
  --model-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --flat-checkpoint-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --checkpoint-name "ns_epoch_300.pt" \
  --results-dir "${REGRESSION_ROOT}/outputs/Regression_results/NS_reg" \
  --device "cuda:0" \
  --seed 100 \
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
  --test-data-path "/net/ghisallo/scratch1/yshi5/OFM/dataset/N_S/ns_test_10000.npy" \
  --test-sample-idx 100 \
  --n-observations 64 \
  --noise-level 1e-2 \
  --n-samples 32 \
  --annealing-steps 40 \
  --ode-steps 20 \
  --langevin-steps 50 \
  --langevin-lr 1e-4 \
  --tau 1.0 \
  --low-rank-cov-rank 32 \
  --low-rank-cov-samples 256 \
  --low-rank-cov-ode-steps 20 \
  --save-prefix "ns_reg"
