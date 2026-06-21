#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PDE_ROOT}/pde_train/OFM_prior_2D.py" \
  --pde "ns" \
  --data-path "${PDE_ROOT}/datasets/ns/ns_nonbounded_flow.npy" \
  --save-dir "${PDE_ROOT}/checkpoints/FAPS_prior/FNO" \
  --checkpoint-name "ns_fno_prior_100.pt" \
  --figure-dir "${PDE_ROOT}/outputs/OFM/ns_input_prior_full" \
  --device "cuda:0" \
  --seed 22 \
  --n-x 128 \
  --modes 48 \
  --width 64 \
  --mlp-width 128 \
  --kernel-length 0.01 \
  --kernel-variance 1.0 \
  --kernel-nu 0.5 \
  --epochs 100 \
  --sigma-min 1e-4 \
  --batch-size 128 \
  --learning-rate 5e-4 \
  --eta-min 1e-6 \
  --eval-samples 256 \
  --sr-eval-samples 128 \
  --eval-batch-size 64 \
  --sr-eval-batch-size 16 \
  --n-eval 20 \
  --super-n-x 160 \
  --autovar-samples 1000 \
  --hist-samples 1000
