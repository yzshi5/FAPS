#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PDE_ROOT}/pde_eval/PDE_inverse_metrics_unet.py" \
  --pde poisson \
  --prior-path "${PDE_ROOT}/checkpoints/FAPS_prior/UNet/poisson_unet_prior_100.pt" \
  --forward-ckpt "${PDE_ROOT}/checkpoints/PDE_surrogate/poisson_forward.pt" \
  --test-data-path "${PDE_ROOT}/datasets/poisson/poisson_pde_test_200.npy" \
  --save-dir "${PDE_ROOT}/outputs/PDE_inverse_metrics_unet/poisson" \
  --device cuda:0 \
  --seed 300 \
  --start-idx 0 \
  --num-test 100 \
  --n-observations 128 \
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
  --data-range 5.0 \
  --save-prefix poisson_unet_test_metrics
