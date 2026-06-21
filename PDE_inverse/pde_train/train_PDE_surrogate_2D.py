#!/usr/bin/env python3
"""Train a 2D FNO surrogate for paired PDE inverse datasets.

Dataset format:
    data.shape = [N, 2, n_x, n_y]
    - channel 0: input coefficient/field
    - channel 1: output solution/response
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PDE_INVERSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, PDE_INVERSE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from faps_utils.fno_solver import FNOSolver


@dataclass(frozen=True)
class TrainConfig:
    pde: str
    data_path: Path
    save_dir: Path
    checkpoint_name: str
    history_name: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    modes: int
    hidden_channels: int
    projection_channels: int
    n_layers: int

    @property
    def checkpoint_path(self) -> Path:
        return self.save_dir / self.checkpoint_name

    @property
    def history_path(self) -> Path:
        return self.save_dir / self.history_name


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
    parser = argparse.ArgumentParser(description="Train a 2D FNO PDE surrogate.")
    parser.add_argument("--pde", type=str, required=True, help="PDE label used in logs.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=PDE_INVERSE_ROOT / "checkpoints" / "PDE_surrogate",
    )
    parser.add_argument("--checkpoint-name", type=str, required=True)
    parser.add_argument("--history-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--projection-channels", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    pde = args.pde.strip()
    history_name = args.history_name or f"{pde}_history.npy"
    return TrainConfig(
        pde=pde,
        data_path=args.data_path.expanduser(),
        save_dir=args.save_dir.expanduser(),
        checkpoint_name=args.checkpoint_name,
        history_name=history_name,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        modes=args.modes,
        hidden_channels=args.hidden_channels,
        projection_channels=args.projection_channels,
        n_layers=args.n_layers,
    )


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff_flat = (pred - target).reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    num = torch.linalg.vector_norm(diff_flat, ord=2, dim=1)
    den = torch.linalg.vector_norm(target_flat, ord=2, dim=1) + eps
    return (num / den).mean()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    rel_sum = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = F.mse_loss(pred, y)
        rel = relative_l2(pred.detach(), y.detach())

        if is_train:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.item())
        rel_sum += float(rel.item())
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0
    return loss_sum / n_batches, rel_sum / n_batches


def build_model(cfg: TrainConfig, device: torch.device) -> FNOSolver:
    return FNOSolver(
        in_channels=1,
        out_channels=1,
        n_modes=(cfg.modes, cfg.modes),
        hidden_channels=cfg.hidden_channels,
        projection_channels=cfg.projection_channels,
        n_layers=cfg.n_layers,
    ).to(device)


def train(cfg: TrainConfig) -> None:
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"Training data not found: {cfg.data_path}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairedPDEDataset(cfg.data_path)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    print(f"PDE: {cfg.pde}")
    print(f"Device: {device}")
    print(f"Data: {cfg.data_path}")
    print(f"Save checkpoint: {cfg.checkpoint_path}")
    print(f"Total samples (all used for training): {len(dataset)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.perf_counter()
        train_loss, train_rel = run_epoch(model, train_loader, device, optimizer=optimizer)
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_t0

        row = {
            "epoch": float(epoch),
            "train_mse": train_loss,
            "train_rel_l2": train_rel,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        history.append(row)

        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train_mse={train_loss:.3e} "
            f"train_rel={train_rel:.3e} "
            f"time={epoch_time:.2f}s"
        )

    torch.save(model.state_dict(), cfg.checkpoint_path)
    np.save(cfg.history_path, np.array(history, dtype=object))
    print(f"Training finished. Final model.state_dict saved to: {cfg.checkpoint_path}")
    print(f"History saved to: {cfg.history_path}")


def main() -> None:
    cfg = build_config(parse_args())
    train(cfg)


if __name__ == "__main__":
    main()
