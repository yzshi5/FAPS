#!/usr/bin/env python3
"""Prepare 100-sample PDE inverse test datasets for FAPS examples.

Creates:
  darcy/darcy_pde_test_100.npy
  poisson/poisson_pde_test_100.npy
  ns/ns_pde_test_100.npy
  helmholtz/helmholtz_test_100.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent

DATASETS = [
    {
        "source": Path("/oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse_npy/darcy_flow_test.npy"),
        "target": ROOT / "darcy" / "darcy_pde_test_200.npy",
    },
    {
        "source": Path("/oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse_npy/poisson_flow_test.npy"),
        "target": ROOT / "poisson" / "poisson_pde_test_200.npy",
    },
    {
        "source": Path("/oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse_npy/ns_nonbounded_flow_test.npy"),
        "target": ROOT / "ns" / "ns_pde_test_200.npy",
    },
    {
        "source": Path("/oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse_npy/helmholtz_flow_test.npy"),
        "target": ROOT / "helmholtz" / "helmholtz_test_200.npy",
    },
]


def prepare_dataset(source: Path, target: Path, n_samples: int, overwrite: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if target.exists() and not overwrite:
        print(f"Skip existing: {target}")
        return

    data = np.load(source, mmap_mode="r")
    if data.shape[0] < n_samples:
        raise ValueError(f"Need at least {n_samples} samples from {source}, got shape {data.shape}.")

    target.parent.mkdir(parents=True, exist_ok=True)
    subset = np.asarray(data[:n_samples])
    np.save(target, subset)
    print(f"Saved first {n_samples} samples: {source} -> {target} | shape={subset.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 200-sample PDE inverse test datasets.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target .npy files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for item in DATASETS:
        prepare_dataset(
            source=item["source"],
            target=item["target"],
            n_samples=args.n_samples,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
