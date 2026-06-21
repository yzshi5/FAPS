#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PDE_ROOT}/pde_eval/PDE_inverse_unet.py" \
  --pde "poisson" \
  --prior-path "${PDE_ROOT}/checkpoints/FAPS_prior/UNet/poisson_unet_prior_100.pt" \
  --forward-ckpt "${PDE_ROOT}/checkpoints/PDE_surrogate/poisson_forward.pt" \
  --test-data-path "${PDE_ROOT}/datasets/poisson/poisson_pde_test_200.npy" \
  --save-dir "${PDE_ROOT}/outputs/PDE_inverse_unet/poisson" \
  --save-prefix "poisson_reg_unet" \
  --device "cuda:0" \
  --seed 22 \
  --test-sample-idx 100 \
  --n-observations 125 \
  --noise-level 1e-3 \
  --n-samples 32 \
  --annealing-steps 20 \
  --ode-steps 10 \
  --langevin-steps 40 \
  --langevin-lr 4e-5 \
  --tau 1.0 \
  --low-rank-cov-rank 32 \
  --low-rank-cov-samples 256 \
  --low-rank-cov-ode-steps 20 \
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
  --forward-modes 48 \
  --forward-hidden-channels 64 \
  --forward-projection-channels 128 \
  --forward-n-layers 4
