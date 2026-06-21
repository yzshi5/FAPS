#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PDE_ROOT}/pde_eval/eval_PDE_surrogate_2D.py" \
  --pde "ns" \
  --test-data-path "/oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse_npy/ns_nonbounded_flow_test.npy" \
  --model-path "${PDE_ROOT}/checkpoints/PDE_surrogate/ns_forward.pt" \
  --fig-dir "${PDE_ROOT}/outputs/PDE_surrogate" \
  --metrics-name "ns_metrics.npy" \
  --save-prefix "ns" \
  --device "cuda:0" \
  --batch-size 256 \
  --num-workers 4 \
  --n-test-samples 1000 \
  --modes 48 \
  --hidden-channels 64 \
  --projection-channels 128 \
  --n-layers 4 \
  --n-examples 2
