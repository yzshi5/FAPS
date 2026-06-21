#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRESSION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${REGRESSION_ROOT}/reg_train/OFM_MINO_prior_weather.py" \
  --data-path "/net/ghisallo/scratch1/yshi5/OFM/dataset/weather/train_climate.npy" \
  --save-dir "${REGRESSION_ROOT}/checkpoints/FAPS_prior" \
  --checkpoint-name "weather_epoch_200.pt" \
  --figure-dir "${REGRESSION_ROOT}/outputs/OFM/weather" \
  --device "cuda:0" \
  --seed 22 \
  --n-longs 90 \
  --n-lats 46 \
  --kernel-length 0.05 \
  --kernel-variance 1.0 \
  --kernel-nu 0.5 \
  --epochs 200 \
  --sigma-min 1e-4 \
  --batch-size 96 \
  --learning-rate 1e-4 \
  --eta-min 1e-6 \
  --eval-samples 64 \
  --sr-eval-samples 32 \
  --eval-batch-size 64 \
  --sr-eval-batch-size 16 \
  --n-eval 20 \
  --mino-x-dim 3 \
  --mino-query-longs 32 \
  --mino-query-lats 16 \
  --mino-co-domain 1 \
  --mino-radius 0.2 \
  --mino-dim 256 \
  --mino-num-heads 4 \
  --mino-enc-depth 5 \
  --mino-dec-depth 1 \
  --mino-backbone-path "${REGRESSION_ROOT}/../checkpoints/MINO_T_Climate/epoch_200.pt"
