"""Train and evaluate an OFM FNO prior on 2D PDE input fields."""

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

PDE_INVERSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, PDE_INVERSE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from faps_utils.fno import FNO
from faps_utils.ofm_ind_likelihood import OFMModel


@dataclass(frozen=True)
class TrainConfig:
    pde: str
    data_path: Path
    save_path: Path
    checkpoint_name: str | None
    figure_save_path: Path
    device: str
    seed: int
    train_samples: int | None
    n_x: int
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
    autovar_samples: int
    hist_samples: int

    @property
    def dims(self) -> list[int]:
        return [self.n_x, self.n_x]

    @property
    def label(self) -> str:
        return self.pde.strip()


def none_or_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and/or evaluate an OFM prior on 2D PDE inputs.")
    parser.add_argument("--pde", type=str, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=PDE_INVERSE_ROOT / "checkpoints" / "FAPS_prior" / "FNO",
    )
    parser.add_argument("--checkpoint-name", type=str, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--train-samples", type=none_or_int, default=None)
    parser.add_argument("--n-x", type=int, default=128)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--kernel-length", type=float, default=0.01)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-nu", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--sr-eval-samples", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--sr-eval-batch-size", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--sample-method", type=str, default="euler")
    parser.add_argument("--super-n-x", type=int, default=160)
    parser.add_argument("--autovar-samples", type=int, default=1000)
    parser.add_argument("--hist-samples", type=int, default=1000)
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate an existing checkpoint.")
    parser.add_argument("--skip-eval", action="store_true", help="Only train/save the prior checkpoint.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        pde=args.pde,
        data_path=args.data_path.expanduser(),
        save_path=resolve_output_path(args.save_dir.expanduser()),
        checkpoint_name=args.checkpoint_name,
        figure_save_path=args.figure_dir.expanduser(),
        device=args.device,
        seed=args.seed,
        train_samples=args.train_samples,
        n_x=args.n_x,
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
        n_eval=args.n_eval,
        sample_method=args.sample_method,
        super_n_x=args.super_n_x,
        autovar_samples=args.autovar_samples,
        hist_samples=args.hist_samples,
    )


def resolve_output_path(save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    probe = save_dir / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return save_dir


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


def averaged_sample_histogram(x: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected shape [N, C, H, W], got {tuple(x.shape)}")
    n_samples = int(x.shape[0])
    if n_samples == 0:
        raise ValueError("Cannot compute histogram for empty tensor.")

    n_values_per_sample = int(np.prod(x.shape[1:]))
    counts = torch.histogram(x, bins=bin_edges, density=False)[0]
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    return counts / (n_samples * n_values_per_sample * bin_widths)


def random_subset(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    n_total = int(x.shape[0])
    if n_total == 0:
        raise ValueError("Cannot sample from an empty tensor.")
    n_take = min(int(n_samples), n_total)
    idx = torch.randperm(n_total)[:n_take]
    return x[idx]


def averaged_radial_autovariance(
    x: torch.Tensor,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 4:
        raise ValueError(f"Expected shape [N, C, H, W], got {tuple(x.shape)}")
    if x.shape[1] != 1:
        raise ValueError(f"Expected single-channel fields [N, 1, H, W], got {tuple(x.shape)}")

    subset = random_subset(x, n_samples)
    fields = subset[:, 0].float()
    _, h, w = fields.shape

    fields = fields - fields.mean(dim=(-2, -1), keepdim=True)
    f_fields = torch.fft.rfft2(fields, dim=(-2, -1))
    power = (f_fields.conj() * f_fields).real
    acov = torch.fft.irfft2(power, s=(h, w), dim=(-2, -1)) / (h * w)
    mean_acov = torch.fft.fftshift(acov.mean(dim=0), dim=(-2, -1))

    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cy, cx = h // 2, w // 2
    rr = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
    rr_int = torch.round(rr).long()
    max_r = int(rr_int.max().item())

    sums = torch.bincount(rr_int.reshape(-1), weights=mean_acov.reshape(-1), minlength=max_r + 1)
    counts = torch.bincount(rr_int.reshape(-1), minlength=max_r + 1)
    profile = sums / counts.clamp_min(1)
    radii = torch.arange(profile.shape[0], dtype=profile.dtype)
    return radii, profile


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
        loss_log_filename=f"{cfg.label}_loss.csv",
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


def plot_training_dataset(x_train: torch.Tensor, cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    bin_edges = torch.linspace(-4.0, 4.0, 101, dtype=x_train.dtype, device=x_train.device)
    x_hist_true = averaged_sample_histogram(random_subset(x_train, cfg.hist_samples), bin_edges)

    n_show = min(10, x_train.shape[0])
    fig, ax = plt.subplots(2, 5, figsize=(14, 6))
    ax = ax.reshape(-1)
    vmin = x_train[: min(1000, x_train.shape[0])].min().item()
    vmax = x_train[: min(1000, x_train.shape[0])].max().item()
    im = None
    for i in range(n_show):
        im = ax[i].imshow(x_train[i, 0].cpu().numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    for i in range(n_show, len(ax)):
        ax[i].axis("off")
    if im is not None:
        fig.colorbar(im, ax=ax.tolist(), shrink=0.7, location="right")
    fig.suptitle(f"Training samples ({cfg.label} input prior)", y=0.98)
    plt.savefig(figure_path(cfg, "training_samples.png"), dpi=200)
    plt.close(fig)

    return x_hist_true, bin_edges


def evaluate_base_resolution(
    ofm: OFMModel,
    x_train: torch.Tensor,
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
            ax[i].imshow(x_ground_truth[i, 0].numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax[i].set_title(f"GT #{i + 1}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        for i in range(n_show, len(ax)):
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label}_samples_base_gt.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(2, 5, figsize=(14, 6))
        ax = ax.reshape(-1)
        for i in range(n_show):
            ax[i].imshow(x_hat[i, 0].numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax[i].set_title(f"OFM #{i + 1}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        for i in range(n_show, len(ax)):
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label}_samples_base_ofm.png"), dpi=200)
        plt.close(fig)

        x_alt = sample_many(
            ofm,
            cfg.dims,
            total_samples=max(cfg.eval_samples, cfg.autovar_samples),
            n_eval=cfg.n_eval,
            sample_batch_size=cfg.eval_batch_size,
            method=cfg.sample_method,
        )
        x_hist = averaged_sample_histogram(x_alt, bin_edges)
        radii_gt, acov_gt = averaged_radial_autovariance(x_train, cfg.autovar_samples)
        radii_ofm, acov_ofm = averaged_radial_autovariance(x_alt, cfg.autovar_samples)

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        ax[0].plot(centers, x_hist_true, c="k", lw=2, label="GT")
        ax[0].plot(centers, x_hist, c="r", ls="--", lw=2, label="OFM")
        ax[0].set_title(f"Average sample-wise histogram ({cfg.hist_samples} random GT)")
        ax[0].set_xlabel("Value")
        ax[0].set_ylabel("Density")
        ax[0].legend(loc="upper right")

        ax[1].plot(radii_gt.cpu().numpy(), acov_gt.cpu().numpy(), c="k", lw=2, label="GT")
        ax[1].plot(radii_ofm.cpu().numpy(), acov_ofm.cpu().numpy(), c="r", ls="--", lw=2, label="OFM")
        ax[1].set_title(f"Radial autovariance ({cfg.autovar_samples} random samples)")
        ax[1].set_xlabel("Pixel lag")
        ax[1].set_ylabel("Autovariance")
        ax[1].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label}_stats_base.png"), dpi=200)
        plt.close(fig)


def evaluate_super_resolution(ofm: OFMModel, x_hist_true: torch.Tensor, bin_edges: torch.Tensor, cfg: TrainConfig) -> None:
    dims_sup = [cfg.super_n_x, cfg.super_n_x]
    with torch.no_grad():
        x_hat = ofm.sample(dims_sup, n_samples=10, n_eval=cfg.n_eval, method=cfg.sample_method).cpu()
        fig, ax = plt.subplots(2, 5, figsize=(14, 6))
        ax = ax.reshape(-1)
        vmin = x_hat.min().item()
        vmax = x_hat.max().item()
        for i in range(10):
            ax[i].imshow(x_hat[i, 0].numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"OFM SR #{i + 1}")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label}_samples_sup_{cfg.super_n_x}.png"), dpi=200)
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
        im0 = ax[0].imshow(x_mean.cpu().numpy(), cmap="RdBu_r")
        ax[0].set_title(f"OFM mean field ({cfg.super_n_x}x{cfg.super_n_x})")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        ax[1].plot((bin_edges[1:] + bin_edges[:-1]) / 2, x_hist_true, c="k", lw=2, label=f"Train {cfg.n_x}x{cfg.n_x}")
        ax[1].plot(
            (bin_edges_alt[1:] + bin_edges_alt[:-1]) / 2,
            x_hist,
            c="r",
            ls="--",
            lw=2,
            label=f"OFM {cfg.super_n_x}x{cfg.super_n_x}",
        )
        ax[1].set_title("Value histogram")
        ax[1].set_xlabel("Value")
        ax[1].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"{cfg.label}_stats_sup_{cfg.super_n_x}.png"), dpi=200)
        plt.close(fig)


def load_pde_input_dataset(data_path: str | Path, n_samples: int | None = None) -> torch.Tensor:
    array = np.load(data_path)
    x_train = torch.from_numpy(array).float()
    if x_train.ndim != 4:
        raise ValueError(f"Expected PDE data shape [N, 2, H, W], got {tuple(x_train.shape)}")
    if x_train.shape[1] < 1:
        raise ValueError(f"Expected at least one channel, got shape {tuple(x_train.shape)}")
    x_train = x_train[:, 0:1]
    if n_samples is not None:
        x_train = x_train[:n_samples]
    return x_train


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    print(f"PDE: {cfg.label}")
    print(f"Using output directory: {cfg.save_path}")
    print(f"Using checkpoint name: {checkpoint_path(cfg).name}")
    print(f"Using figure directory: {cfg.figure_save_path}")
    mode = "evaluation-only" if args.eval_only else "train + evaluation"
    if args.skip_eval:
        mode = "train-only"
    print(f"Run mode: {mode}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    x_train = load_pde_input_dataset(cfg.data_path, n_samples=cfg.train_samples)
    print(f"Training data shape: {tuple(x_train.shape)}")

    loader_tr = DataLoader(x_train, batch_size=cfg.batch_size, shuffle=True)
    x_hist_true, bin_edges = plot_training_dataset(x_train, cfg)

    if not args.eval_only:
        train_ofm(loader_tr, cfg)
        finalize_saved_checkpoint(cfg)
    elif not checkpoint_path(cfg).exists() and not default_checkpoint_path(cfg).exists():
        raise FileNotFoundError(f"Checkpoint not found for --eval-only: {checkpoint_path(cfg)}")

    if args.skip_eval:
        print("Skipping evaluation plots (--skip-eval).")
        return

    ofm = load_trained_ofm(cfg)
    eval_t0 = time.perf_counter()
    evaluate_base_resolution(ofm, x_train, x_hist_true, bin_edges, cfg)
    evaluate_super_resolution(ofm, x_hist_true, bin_edges, cfg)
    eval_elapsed = time.perf_counter() - eval_t0
    print(f"Evaluation plotting + sampling time: {eval_elapsed:.2f} s")


if __name__ == "__main__":
    main()
