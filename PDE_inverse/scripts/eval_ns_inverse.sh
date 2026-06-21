#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PDE_ROOT}/pde_eval/PDE_inverse.py" \
  --pde "ns" \
  --prior-path "${PDE_ROOT}/checkpoints/FAPS_prior/FNO/ns_fno_prior_100.pt" \
  --forward-ckpt "${PDE_ROOT}/checkpoints/PDE_surrogate/ns_forward.pt" \
  --test-data-path "${PDE_ROOT}/datasets/ns/ns_pde_test_200.npy" \
  --save-dir "${PDE_ROOT}/outputs/PDE_inverse/ns" \
  --save-prefix "ns_reg" \
  --device "cuda:0" \
  --seed 100 \
  --test-sample-idx 100 \
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
  --n-x 128 \
  --modes 48 \
  --width 64 \
  --mlp-width 128 \
  --epochs 100 \
  --sigma-min 1e-4 \
  --kernel-length 0.01 \
  --kernel-variance 1.0 \
  --kernel-nu 0.5 \
  --forward-modes 48 \
  --forward-hidden-channels 64 \
  --forward-projection-channels 128 \
  --forward-n-layers 4
