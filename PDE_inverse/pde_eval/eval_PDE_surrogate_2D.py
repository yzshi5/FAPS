#!/usr/bin/env python3
"""Evaluate a trained 2D FNO surrogate for paired PDE inverse datasets."""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PDE_INVERSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, PDE_INVERSE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from faps_utils.fno_solver import FNOSolver


@dataclass(frozen=True)
class EvalConfig:
    pde: str
    test_data_path: Path
    model_path: Path
    fig_dir: Path
    metrics_name: str
    save_prefix: str
    device: str
    batch_size: int
    num_workers: int
    n_test_samples: int
    modes: int
    hidden_channels: int
    projection_channels: int
    n_layers: int
    n_examples: int

    @property
    def metrics_path(self) -> Path:
        return self.fig_dir / self.metrics_name


class PairedPDEDataset(Dataset):
    """Map .npy data [N, 2, H, W] to supervised pairs (x, y)."""

    def __init__(self, npy_path: Path):
        self.npy_path = npy_path
        self.data = np.load(npy_path, mmap_mode="r")
        if self.data.ndim != 4:
            raise ValueError(f"Expected 4D data [N, 2, H, W], got {self.data.shape}")
        if self.data.shape[1] != 2:
            raise ValueError(f"Expected channel size 2, got {self.data.shape}")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = np.asarray(self.data[idx], dtype=np.float32)
        x = torch.from_numpy(sample[0:1])
        y = torch.from_numpy(sample[1:2])
        return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a 2D FNO PDE surrogate.")
    parser.add_argument("--pde", type=str, required=True, help="PDE label used in logs.")
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=PDE_INVERSE_ROOT / "outputs" / "PDE_surrogate",
    )
    parser.add_argument("--metrics-name", type=str, default=None)
    parser.add_argument("--save-prefix", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-test-samples", type=int, default=1000)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--projection-channels", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-examples", type=int, default=2)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    pde = args.pde.strip()
    return EvalConfig(
        pde=pde,
        test_data_path=args.test_data_path.expanduser(),
        model_path=args.model_path.expanduser(),
        fig_dir=args.fig_dir.expanduser(),
        metrics_name=args.metrics_name or f"{pde}_metrics.npy",
        save_prefix=args.save_prefix or pde,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_test_samples=args.n_test_samples,
        modes=args.modes,
        hidden_channels=args.hidden_channels,
        projection_channels=args.projection_channels,
        n_layers=args.n_layers,
        n_examples=args.n_examples,
    )


def relative_l2_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff_flat = (pred - target).reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    num = torch.linalg.vector_norm(diff_flat, ord=2, dim=1)
    den = torch.linalg.vector_norm(target_flat, ord=2, dim=1) + eps
    return num / den


def count_parameters(model: torch.nn.Module) -> tuple[int, int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen


def build_model(cfg: EvalConfig, device: torch.device) -> FNOSolver:
    return FNOSolver(
        in_channels=1,
        out_channels=1,
        n_modes=(cfg.modes, cfg.modes),
        hidden_channels=cfg.hidden_channels,
        projection_channels=cfg.projection_channels,
        n_layers=cfg.n_layers,
    ).to(device)


def load_state_dict(model_path: Path, device: torch.device):
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    except pickle.UnpicklingError:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != "_metadata"}
    return state_dict


def save_example_figure(
    x: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: Path,
) -> None:
    error = pred - gt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    im0 = axes[0].imshow(x, cmap="viridis")
    axes[0].set_title("Input")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred, cmap="RdBu_r")
    axes[1].set_title("Output (Pred)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(gt, cmap="RdBu_r")
    axes[2].set_title("Ground Truth")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    vmax = np.max(np.abs(error))
    im3 = axes[3].imshow(error, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[3].set_title("Error (Pred - GT)")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate(cfg: EvalConfig) -> None:
    if not cfg.test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {cfg.test_data_path}")
    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {cfg.model_path}")

    device = torch.device(cfg.device)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = PairedPDEDataset(cfg.test_data_path)
    n_eval = min(cfg.n_test_samples, len(base_dataset))
    dataset = Subset(base_dataset, list(range(n_eval)))

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(cfg, device)
    model.load_state_dict(load_state_dict(cfg.model_path, device))
    model.eval()
    total_params, trainable_params, frozen_params = count_parameters(model)

    mse_total = 0.0
    rel_total = 0.0
    sample_count = 0
    example_triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)

            mse_ps = F.mse_loss(pred, y, reduction="none").mean(dim=(1, 2, 3))
            rel_ps = relative_l2_per_sample(pred, y)

            bs = x.shape[0]
            mse_total += float(mse_ps.sum().item())
            rel_total += float(rel_ps.sum().item())
            sample_count += bs

            if len(example_triplets) < cfg.n_examples:
                x_np = x[:, 0].detach().cpu().numpy()
                pred_np = pred[:, 0].detach().cpu().numpy()
                y_np = y[:, 0].detach().cpu().numpy()
                for i in range(bs):
                    if len(example_triplets) >= cfg.n_examples:
                        break
                    example_triplets.append((x_np[i], pred_np[i], y_np[i]))

    mse_mean = mse_total / sample_count
    rel_mean = rel_total / sample_count

    metrics = {
        "pde": cfg.pde,
        "evaluated_samples": sample_count,
        "mse": mse_mean,
        "relative_l2": rel_mean,
        "model_path": str(cfg.model_path),
        "test_data_path": str(cfg.test_data_path),
    }
    np.save(cfg.metrics_path, np.array(metrics, dtype=object))

    print(f"PDE: {cfg.pde}")
    print(f"Device: {device}")
    print(f"Model: {cfg.model_path}")
    print(
        "Model parameters: "
        f"total={total_params:,} ({total_params / 1e6:.3f}M), "
        f"trainable={trainable_params:,}, frozen={frozen_params:,}"
    )
    print(f"Test data: {cfg.test_data_path}")
    print(f"Evaluated samples: {sample_count}")
    print(f"Prediction mean MSE over first {sample_count}: {mse_mean:.6e}")
    print(f"Prediction mean relative L2 over first {sample_count}: {rel_mean:.6e}")
    print(f"Saved metrics: {cfg.metrics_path}")

    for i, (x_i, pred_i, gt_i) in enumerate(example_triplets, start=1):
        fig_path = cfg.fig_dir / f"{cfg.save_prefix}_test_example_{i}.png"
        save_example_figure(x_i, pred_i, gt_i, fig_path)
        print(f"Saved figure: {fig_path}")


def main() -> None:
    evaluate(build_config(parse_args()))


if __name__ == "__main__":
    main()
