"""Train and evaluate OFM priors on generic 2D field datasets."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REGRESSION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import faps_utils as _faps_utils

sys.modules.setdefault("ofm_utils", _faps_utils)

from faps_utils.fno import FNO
from faps_utils.ofm_ind_likelihood import OFMModel

NS_DATA_PATH = "/net/ghisallo/scratch1/yshi5/OFM/dataset/N_S/ns_30000.npy"
BH_DATA_PATH = "/net/wintermute/scratch/agao3/OpFlow/Unoflow/m_1_1_40_all.npy"

DATASET_DEFAULTS = {
    "ns": {
        "label": "NS",
        "data_path": NS_DATA_PATH,
        "save_dir": REGRESSION_ROOT / "checkpoints" / "FAPS_prior",
        "checkpoint_name": "ns_epoch_300.pt",
        "figure_dir": REGRESSION_ROOT / "outputs" / "OFM" / "NS",
        "train_samples": 30_000,
        "cmap": "RdBu_r",
        "n_eval": 20,
        "train_split_idx": None,
        "rotation_augment": False,
    },
    "bh": {
        "label": "BH",
        "data_path": BH_DATA_PATH,
        "save_dir": REGRESSION_ROOT / "checkpoints" / "FAPS_prior",
        "checkpoint_name": "bh_epoch_300.pt",
        "figure_dir": REGRESSION_ROOT / "outputs" / "OFM" / "BH",
        "train_samples": None,
        "cmap": "viridis",
        "n_eval": 20,
        "train_split_idx": None,
        "rotation_augment": True,
    },
    "custom": {
        "label": "2D",
        "data_path": None,
        "save_dir": REGRESSION_ROOT / "checkpoints" / "FAPS_prior",
        "checkpoint_name": None,
        "figure_dir": REGRESSION_ROOT / "outputs" / "OFM" / "2D",
        "train_samples": None,
        "cmap": "RdBu_r",
        "n_eval": 20,
        "train_split_idx": None,
        "rotation_augment": False,
    },
}


@dataclass(frozen=True)
class TrainConfig:
    dataset: str
    label: str
    data_path: Path
    save_path: Path
    checkpoint_name: str | None
    figure_save_path: Path
    device: str
    seed: int
    train_samples: int | None
    train_split_idx: int | None
    rotation_augment: bool
    n_x: int
    n_y: int
    modes: int
    width: int
    mlp_width: int
    kernel_length: float
    kernel_variance: float
    kernel_nu: float
    epochs: int
    sigma_min: float
    batch_size: int
    learning_rate: float
    eta_min: float
    eval_samples: int
    sr_eval_samples: int
    eval_batch_size: int
    sr_eval_batch_size: int
    n_eval: int
    sample_method: str
    super_n_x: int
    super_n_y: int
    cmap: str

    @property
    def dims(self) -> list[int]:
        return [self.n_x, self.n_y]

    @property
    def label_lower(self) -> str:
        return self.label.lower()


def none_or_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate OFM on generic 2D field data."
    )
    parser.add_argument("--dataset", choices=DATASET_DEFAULTS.keys(), default="ns")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Optional checkpoint filename under --save-dir, e.g. ns_epoch_300.pt.",
    )
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--train-samples", type=none_or_int, default=None)
    parser.add_argument("--train-split-idx", type=none_or_int, default=None)
    parser.add_argument("--rotation-augment", action="store_true", default=None)
    parser.add_argument("--no-rotation-augment", action="store_false", dest="rotation_augment")
    parser.add_argument("--n-x", type=int, default=None)
    parser.add_argument("--n-y", type=int, default=None)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--kernel-length", type=float, default=0.01)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-nu", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--sr-eval-samples", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--sr-eval-batch-size", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--sample-method", type=str, default="euler")
    parser.add_argument("--super-n-x", type=int, default=96)
    parser.add_argument("--super-n-y", type=int, default=96)
    parser.add_argument("--cmap", type=str, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate from the existing checkpoint.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only train/save the prior checkpoint.",
    )
    return parser.parse_args()


def resolve_output_path(save_dir: Path) -> Path:
    """Create and validate the requested model output directory."""
    save_dir = save_dir.expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    probe = save_dir / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return save_dir


def build_config(args: argparse.Namespace, data_shape: tuple[int, int]) -> TrainConfig:
    defaults = DATASET_DEFAULTS[args.dataset]
    data_path = args.data_path or defaults["data_path"]
    if data_path is None:
        raise ValueError("--data-path is required when --dataset custom is used.")

    train_samples = args.train_samples
    if train_samples is None:
        train_samples = defaults["train_samples"]
    train_split_idx = args.train_split_idx
    if train_split_idx is None:
        train_split_idx = defaults["train_split_idx"]
    rotation_augment = args.rotation_augment
    if rotation_augment is None:
        rotation_augment = defaults["rotation_augment"]

    return TrainConfig(
        dataset=args.dataset,
        label=defaults["label"],
        data_path=Path(data_path).expanduser(),
        save_path=resolve_output_path(args.save_dir or defaults["save_dir"]),
        checkpoint_name=args.checkpoint_name or defaults["checkpoint_name"],
        figure_save_path=(args.figure_dir or defaults["figure_dir"]).expanduser(),
        device=args.device,
        seed=args.seed,
        train_samples=train_samples,
        train_split_idx=train_split_idx,
        rotation_augment=rotation_augment,
        n_x=args.n_x or data_shape[0],
        n_y=args.n_y or data_shape[1],
        modes=args.modes,
        width=args.width,
        mlp_width=args.mlp_width,
        kernel_length=args.kernel_length,
        kernel_variance=args.kernel_variance,
        kernel_nu=args.kernel_nu,
        epochs=args.epochs,
        sigma_min=args.sigma_min,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eta_min=args.eta_min,
        eval_samples=args.eval_samples,
        sr_eval_samples=args.sr_eval_samples,
        eval_batch_size=args.eval_batch_size,
        sr_eval_batch_size=args.sr_eval_batch_size,
        n_eval=args.n_eval if args.n_eval is not None else defaults["n_eval"],
        sample_method=args.sample_method,
        super_n_x=args.super_n_x,
        super_n_y=args.super_n_y,
        cmap=args.cmap or defaults["cmap"],
    )


def figure_path(cfg: TrainConfig, filename: str) -> Path:
    cfg.figure_save_path.mkdir(parents=True, exist_ok=True)
    return cfg.figure_save_path / filename


def default_checkpoint_path(cfg: TrainConfig) -> Path:
    return cfg.save_path / f"epoch_{cfg.epochs}.pt"


def checkpoint_path(cfg: TrainConfig) -> Path:
    if cfg.checkpoint_name:
        return cfg.save_path / cfg.checkpoint_name
    return default_checkpoint_path(cfg)


def finalize_saved_checkpoint(cfg: TrainConfig) -> None:
    """Rename the OFM default checkpoint to the configured public filename."""
    source = default_checkpoint_path(cfg)
    target = checkpoint_path(cfg)
    if source == target:
        return
    if not source.exists():
        raise FileNotFoundError(f"Expected training checkpoint not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    print(f"Saved checkpoint: {target}")


def sample_many(
    ofm: OFMModel,
    dims: list[int],
    total_samples: int,
    n_eval: int,
    sample_batch_size: int,
    method: str,
) -> torch.Tensor:
    chunks = []
    remaining = int(total_samples)
    while remaining > 0:
        cur_bs = min(sample_batch_size, remaining)
        chunks.append(ofm.sample(dims, n_samples=cur_bs, n_eval=n_eval, method=method).cpu())
        remaining -= cur_bs
    return torch.cat(chunks, dim=0)


def build_model(cfg: TrainConfig) -> FNO:
    return FNO(
        cfg.modes,
        vis_channels=1,
        hidden_channels=cfg.width,
        proj_channels=cfg.mlp_width,
        x_dim=2,
        t_scaling=1,
    ).to(cfg.device)


def build_ofm(model: FNO, cfg: TrainConfig) -> OFMModel:
    return OFMModel(
        model,
        kernel_length=cfg.kernel_length,
        kernel_variance=cfg.kernel_variance,
        nu=cfg.kernel_nu,
        sigma_min=cfg.sigma_min,
        device=cfg.device,
        dims=cfg.dims,
    )


def train_ofm(loader_tr: DataLoader, cfg: TrainConfig) -> None:
    model = build_model(cfg)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.eta_min,
    )
    ofm = build_ofm(model, cfg)
    ofm.train(
        loader_tr,
        optimizer,
        epochs=cfg.epochs,
        scheduler=scheduler,
        eval_int=0,
        save_int=cfg.epochs,
        generate=False,
        save_path=cfg.save_path,
        saved_model=True,
        loss_filename=None,
        loss_log_filename=f"{cfg.label_lower}_loss.csv",
    )


def load_trained_ofm(cfg: TrainConfig) -> OFMModel:
    model = build_model(cfg)
    for param in model.parameters():
        param.requires_grad = False

    model_path = checkpoint_path(cfg)
    if not model_path.exists() and cfg.checkpoint_name:
        model_path = default_checkpoint_path(cfg)
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if hasattr(checkpoint, "state_dict"):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict):
        checkpoint = {k: v for k, v in checkpoint.items() if k != "_metadata"}

    model.load_state_dict(checkpoint)
    return build_ofm(model, cfg)


def plot_training_dataset(
    x_train: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plot training samples and return mean/histogram statistics."""
    x_mean_true = x_train.mean(dim=0).squeeze(0)
    x_hist_true, bin_edges = x_train.histogram(range=[-4, 4], density=True)

    n_show = min(10, x_train.shape[0])
    fig, ax = plt.subplots(2, 5, figsize=(14, 6))
    ax = ax.reshape(-1)
    vmin = x_train[: min(1000, x_train.shape[0])].min().item()
    vmax = x_train[: min(1000, x_train.shape[0])].max().item()
    im = None
    for i in range(n_show):
        im = ax[i].imshow(x_train[i, 0].cpu().numpy(), cmap=cfg.cmap, vmin=vmin, vmax=vmax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    for i in range(n_show, len(ax)):
        ax[i].axis("off")
    if im is not None:
        fig.colorbar(im, ax=ax.tolist(), shrink=0.7, location="right")
    fig.suptitle(f"Training samples (2D {cfg.label})", y=0.98)
    plt.tight_layout()
    plt.savefig(figure_path(cfg, "training_samples.png"), dpi=200)
    plt.close(fig)

    return x_mean_true, x_hist_true, bin_edges


def evaluate_base_resolution(
    ofm: OFMModel,
    x_train: torch.Tensor,
    x_mean_true: torch.Tensor,
    x_hist_true: torch.Tensor,
    bin_edges: torch.Tensor,
    cfg: TrainConfig,
) -> None:
    with torch.no_grad():
        n_show = min(10, x_train.shape[0])
        x_hat = ofm.sample(cfg.dims, n_samples=n_show, n_eval=cfg.n_eval, method=cfg.sample_method).cpu()
        x_ground_truth = x_train[:n_show].cpu()
        vmin = min(x_ground_truth.min().item(), x_hat.min().item())
        vmax = max(x_ground_truth.max().item(), x_hat.max().item())

        fig, ax = plt.subplots(2, 5, figsize=(14, 6))
        ax = ax.reshape(-1)
        for i in range(n_show):
            ax[i].imshow(x_ground_truth[i, 0].numpy(), cmap=cfg.cmap, vmin=vmin, vmax=vmax)
            ax[i].set_title(f"GT #{i + 1}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        for i in range(n_show, len(ax)):
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label_lower}_samples_base_gt.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(2, 5, figsize=(14, 6))
        ax = ax.reshape(-1)
        for i in range(n_show):
            ax[i].imshow(x_hat[i, 0].numpy(), cmap=cfg.cmap, vmin=vmin, vmax=vmax)
            ax[i].set_title(f"OFM #{i + 1}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        for i in range(n_show, len(ax)):
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label_lower}_samples_base_ofm.png"), dpi=200)
        plt.close(fig)

        x_alt = sample_many(
            ofm,
            cfg.dims,
            total_samples=cfg.eval_samples,
            n_eval=cfg.n_eval,
            sample_batch_size=cfg.eval_batch_size,
            method=cfg.sample_method,
        )
        x_hist, bin_edges_alt = x_alt.histogram(range=[-4, 4], density=True)
        x_mean = x_alt.mean(dim=0).squeeze(0)

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
        im0 = ax[0].imshow(x_mean_true.cpu().numpy(), cmap=cfg.cmap)
        ax[0].set_title(f"GT mean field ({cfg.n_x}x{cfg.n_y})")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(x_mean.cpu().numpy(), cmap=cfg.cmap)
        ax[1].set_title(f"OFM mean field ({cfg.n_x}x{cfg.n_y})")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        ax[2].plot((bin_edges[1:] + bin_edges[:-1]) / 2, x_hist_true, c="k", lw=2, label="GT")
        ax[2].plot((bin_edges_alt[1:] + bin_edges_alt[:-1]) / 2, x_hist, c="r", ls="--", lw=2, label="OFM")
        ax[2].set_title("Value histogram")
        ax[2].set_xlabel("Value")
        ax[2].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label_lower}_stats_base.png"), dpi=200)
        plt.close(fig)


def evaluate_super_resolution(ofm: OFMModel, x_hist_true: torch.Tensor, bin_edges: torch.Tensor, cfg: TrainConfig) -> None:
    dims_sup = [cfg.super_n_x, cfg.super_n_y]
    with torch.no_grad():
        x_hat = ofm.sample(dims_sup, n_samples=10, n_eval=cfg.n_eval, method=cfg.sample_method).cpu()
        fig, ax = plt.subplots(2, 5, figsize=(14, 6))
        ax = ax.reshape(-1)
        vmin = x_hat.min().item()
        vmax = x_hat.max().item()
        for i in range(10):
            ax[i].imshow(x_hat[i, 0].numpy(), cmap=cfg.cmap, vmin=vmin, vmax=vmax)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"OFM SR #{i + 1}")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label_lower}_samples_sup_{cfg.super_n_x}x{cfg.super_n_y}.png"), dpi=200)
        plt.close(fig)

        x_alt = sample_many(
            ofm,
            dims_sup,
            total_samples=cfg.sr_eval_samples,
            n_eval=cfg.n_eval,
            sample_batch_size=cfg.sr_eval_batch_size,
            method=cfg.sample_method,
        )
        x_hist, bin_edges_alt = x_alt.histogram(range=[-4, 4], density=True)
        x_mean = x_alt.mean(dim=0).squeeze(0)

        fig, ax = plt.subplots(1, 2, figsize=(8, 3.8))
        im0 = ax[0].imshow(x_mean.cpu().numpy(), cmap=cfg.cmap)
        ax[0].set_title(f"OFM mean field ({cfg.super_n_x}x{cfg.super_n_y})")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        ax[1].plot((bin_edges[1:] + bin_edges[:-1]) / 2, x_hist_true, c="k", lw=2, label=f"Train {cfg.n_x}x{cfg.n_y}")
        ax[1].plot((bin_edges_alt[1:] + bin_edges_alt[:-1]) / 2, x_hist, c="r", ls="--", lw=2, label=f"OFM {cfg.super_n_x}x{cfg.super_n_y}")
        ax[1].set_title("Value histogram")
        ax[1].set_xlabel("Value")
        ax[1].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label_lower}_stats_sup_{cfg.super_n_x}x{cfg.super_n_y}.png"), dpi=200)
        plt.close(fig)


def load_2d_dataset(
    data_path: Path,
    *,
    dataset: str,
    train_samples: int | None,
    train_split_idx: int | None,
    rotation_augment: bool,
) -> torch.Tensor:
    array = np.load(data_path)
    x_train = torch.from_numpy(array).float()
    if x_train.ndim == 3:
        x_train = x_train.unsqueeze(1)
    if x_train.ndim != 4:
        raise ValueError(
            f"Expected 2D data shape [N, C, H, W] or [N, H, W], got {tuple(x_train.shape)}"
        )
    x_train = x_train[:, :1]

    if train_split_idx is not None:
        x_train = x_train[:train_split_idx]

    if rotation_augment:
        x_train = torch.cat(
            [torch.rot90(x_train, k, dims=(-2, -1)) for k in range(4)],
            dim=0,
        )
        if dataset == "bh":
            x_train = x_train + 1e-8

    if train_samples is not None:
        if rotation_augment and train_samples < x_train.shape[0]:
            perm = torch.randperm(x_train.shape[0])[:train_samples]
            x_train = x_train[perm]
        else:
            x_train = x_train[:train_samples]
    return x_train


def main() -> None:
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    raw_data_path = args.data_path or defaults["data_path"]
    if raw_data_path is None:
        raise ValueError("--data-path is required when --dataset custom is used.")
    data_path = Path(raw_data_path).expanduser()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_samples = args.train_samples if args.train_samples is not None else defaults["train_samples"]
    train_split_idx = args.train_split_idx if args.train_split_idx is not None else defaults["train_split_idx"]
    rotation_augment = args.rotation_augment
    if rotation_augment is None:
        rotation_augment = defaults["rotation_augment"]

    x_train = load_2d_dataset(
        data_path,
        dataset=args.dataset,
        train_samples=train_samples,
        train_split_idx=train_split_idx,
        rotation_augment=rotation_augment,
    )
    cfg = build_config(args, data_shape=tuple(x_train.shape[-2:]))

    print(f"Dataset: {cfg.label}")
    print(f"Data path: {cfg.data_path}")
    print(f"Training data shape: {tuple(x_train.shape)}")
    print(f"Model dims: {cfg.dims}")
    print(f"Using output directory: {cfg.save_path}")
    print(f"Using figure directory: {cfg.figure_save_path}")
    mode = "evaluation-only" if args.eval_only else "train + evaluation"
    if args.skip_eval:
        mode = "train-only"
    print(f"Run mode: {mode}")

    loader_tr = DataLoader(x_train, batch_size=cfg.batch_size, shuffle=True)
    x_mean_true, x_hist_true, bin_edges = plot_training_dataset(x_train, cfg)

    if not args.eval_only:
        train_ofm(loader_tr, cfg)
        finalize_saved_checkpoint(cfg)
    elif not checkpoint_path(cfg).exists() and not default_checkpoint_path(cfg).exists():
        raise FileNotFoundError(
            f"Checkpoint not found for --eval-only: {checkpoint_path(cfg)}"
        )

    if args.skip_eval:
        return

    ofm = load_trained_ofm(cfg)
    eval_t0 = time.perf_counter()
    evaluate_base_resolution(
        ofm,
        x_train,
        x_mean_true,
        x_hist_true,
        bin_edges,
        cfg,
    )
    evaluate_super_resolution(ofm, x_hist_true, bin_edges, cfg)
    eval_elapsed = time.perf_counter() - eval_t0
    print(f"Evaluation plotting + sampling time: {eval_elapsed:.2f} s")


if __name__ == "__main__":
    main()
