#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PDE_ROOT}/pde_train/unet_prior_2D.py" \
  --pde "poisson" \
  --data-path "${PDE_ROOT}/datasets/poisson/poisson_flow.npy" \
  --save-dir "${PDE_ROOT}/checkpoints/FAPS_prior/UNet" \
  --checkpoint-name "poisson_unet_prior_100.pt" \
  --figure-dir "${PDE_ROOT}/outputs/OFM_UNet/poisson_input_prior" \
  --device "cuda:0" \
  --seed 22 \
  --n-x 128 \
  --vis-channels 1 \
  --n-dummy-conds 1 \
  --unet-hidden-channels 64 \
  --unet-res-blocks 1 \
  --unet-heads 4 \
  --unet-attention-res "16" \
  --unet-channel-mult "none" \
  --epochs 100 \
  --sigma-min 1e-4 \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --eta-min 1e-6 \
  --eval-samples 128 \
  --eval-batch-size 64 \
  --n-eval 40 \
  --sample-method "euler" \
  --autovar-samples 1000 \
  --hist-samples 1000
