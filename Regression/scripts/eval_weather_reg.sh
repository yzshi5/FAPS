#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REGRESSION_ROOT}/reg_eval/MINO_weather_reg.py" \
  --model-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --flat-checkpoint-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --save-dir "${REGRESSION_ROOT}/outputs/Regression_results/weather_reg" \
  --device "cuda:0" \
  --seed 22 \
  --n-longs 90 \
  --n-lats 46 \
  --epochs 200 \
  --sigma-min 1e-4 \
  --kernel-length 0.05 \
  --kernel-variance 1.0 \
  --kernel-nu 0.5 \
  --mino-x-dim 3 \
  --mino-query-longs 32 \
  --mino-query-lats 16 \
  --mino-co-domain 1 \
  --mino-radius 0.2 \
  --mino-dim 256 \
  --mino-num-heads 4 \
  --mino-enc-depth 5 \
  --mino-dec-depth 2 \
  --test-data-path "/net/ghisallo/scratch1/yshi5/OFM/dataset/weather/test_climate.npy" \
  --test-sample-idx 100 \
  --n-observations 125 \
  --noise-level 1e-3 \
  --n-samples 32 \
  --annealing-steps 40 \
  --ode-steps 20 \
  --langevin-steps 50 \
  --langevin-lr 1e-4 \
  --tau 1.0 \
  --low-rank-cov-rank 32 \
  --low-rank-cov-samples 256 \
  --low-rank-cov-batch-size 64 \
  --low-rank-cov-ode-steps 20 \
  --save-prefix "weather_reg"
