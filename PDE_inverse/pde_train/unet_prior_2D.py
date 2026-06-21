"""Train and evaluate an OFM UNet prior on 2D PDE input fields."""

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

from faps_utils.ofm_white_velocity_pred import OFMModel
from faps_utils.unet_ofm import UNet_cond


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
    vis_channels: int
    n_dummy_conds: int
    unet_hidden_channels: int
    unet_res_blocks: int
    unet_heads: int
    unet_attention_res: str
    unet_channel_mult: str | None
    epochs: int
    sigma_min: float
    batch_size: int
    learning_rate: float
    eta_min: float
    eval_samples: int
    eval_batch_size: int
    n_eval: int
    sample_method: str
    autovar_samples: int
    hist_samples: int

    @property
    def dims(self) -> list[int]:
        return [self.n_x, self.n_x]

    @property
    def label(self) -> str:
        return self.pde.strip()

    @property
    def loss_log_name(self) -> str:
        return f"{self.label}_unet_loss.csv"


def none_or_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "-1"}:
        return None
    return int(value)


def none_or_str(value: str) -> str | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and/or evaluate an OFM UNet prior on 2D PDE inputs.")
    parser.add_argument("--pde", type=str, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=PDE_INVERSE_ROOT / "checkpoints" / "FAPS_prior" / "UNet",
    )
    parser.add_argument("--checkpoint-name", type=str, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--train-samples", type=none_or_int, default=None)
    parser.add_argument("--n-x", type=int, default=128)
    parser.add_argument("--vis-channels", type=int, default=1)
    parser.add_argument("--n-dummy-conds", type=int, default=1)
    parser.add_argument("--unet-hidden-channels", type=int, default=64)
    parser.add_argument("--unet-res-blocks", type=int, default=1)
    parser.add_argument("--unet-heads", type=int, default=4)
    parser.add_argument("--unet-attention-res", type=str, default="16")
    parser.add_argument("--unet-channel-mult", type=none_or_str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--n-eval", type=int, default=40)
    parser.add_argument("--sample-method", type=str, default="euler")
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
        vis_channels=args.vis_channels,
        n_dummy_conds=args.n_dummy_conds,
        unet_hidden_channels=args.unet_hidden_channels,
        unet_res_blocks=args.unet_res_blocks,
        unet_heads=args.unet_heads,
        unet_attention_res=args.unet_attention_res,
        unet_channel_mult=args.unet_channel_mult,
        epochs=args.epochs,
        sigma_min=args.sigma_min,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eta_min=args.eta_min,
        eval_samples=args.eval_samples,
        eval_batch_size=args.eval_batch_size,
        n_eval=args.n_eval,
        sample_method=args.sample_method,
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


class PDEInputOFMDataset(torch.utils.data.Dataset):
    """PDE input prior data with dummy conditioning."""

    def __init__(self, x_data: torch.Tensor, n_dummy_conds: int) -> None:
        self.x_data = x_data
        self.n_dummy_conds = int(n_dummy_conds)

    def __len__(self) -> int:
        return int(self.x_data.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        conds = torch.zeros(self.n_dummy_conds, dtype=torch.float32)
        return self.x_data[idx], conds


class VelocityUNetAdapter(torch.nn.Module):
    """Use UNet_cond as an unconditional velocity model with dummy conditions."""

    def __init__(self, model: UNet_cond) -> None:
        super().__init__()
        self.model = model
        hidden_channels = int(self.model.unet_dims[0])
        in_channels = int(self.model.in_channels)
        self.input_proj_2d = (
            torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
            if in_channels != hidden_channels
            else torch.nn.Identity()
        )
        self.output_proj_2d = (
            torch.nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
            if hidden_channels != in_channels
            else torch.nn.Identity()
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        conds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 2D input tensor [B, C, H, W], got shape {tuple(x.shape)}")
        t = t / self.model.t_scaling
        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(x.shape[0], device=t.device, dtype=t.dtype) * t
        if t.dim() != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"Expected time shape [B], got {tuple(t.shape)} for batch {x.shape[0]}")
        x = self.input_proj_2d(x)
        out = self.model.unet_backbone(t, x, None)
        return self.output_proj_2d(out)


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    base_model = UNet_cond(
        dims=[cfg.vis_channels, *cfg.dims],
        hidden_channels=cfg.unet_hidden_channels,
        conds_channels=cfg.n_dummy_conds,
        num_res_blocks=cfg.unet_res_blocks,
        num_heads=cfg.unet_heads,
        attention_res=cfg.unet_attention_res,
        channel_mult=cfg.unet_channel_mult,
        in_channels=cfg.vis_channels,
    )
    return VelocityUNetAdapter(base_model).to(cfg.device)


def build_ofm(cfg: TrainConfig, model: torch.nn.Module) -> OFMModel:
    return OFMModel(
        model,
        sigma_min=cfg.sigma_min,
        device=cfg.device,
        dims=cfg.dims,
    )


def train_ofm(cfg: TrainConfig, loader_tr: DataLoader) -> None:
    model = build_model(cfg)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.eta_min,
    )
    ofm = build_ofm(cfg, model)
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
        loss_log_filename=cfg.loss_log_name,
    )
    finalize_saved_checkpoint(cfg)


def load_checkpoint_state(model: torch.nn.Module, checkpoint: dict) -> None:
    if hasattr(checkpoint, "state_dict"):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        checkpoint = {k: v for k, v in checkpoint.items() if k != "_metadata"}
    model.load_state_dict(checkpoint)


def load_trained_ofm(cfg: TrainConfig) -> OFMModel:
    model = build_model(cfg)
    for param in model.parameters():
        param.requires_grad = False
    model_path = checkpoint_path(cfg)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    load_checkpoint_state(model, checkpoint)
    model.eval()
    return build_ofm(cfg, model)


def sample_many(
    cfg: TrainConfig,
    ofm: OFMModel,
    total_samples: int,
    n_eval: int,
    sample_batch_size: int,
) -> torch.Tensor:
    chunks = []
    remaining = int(total_samples)
    while remaining > 0:
        cur_bs = min(sample_batch_size, remaining)
        conds = torch.zeros(cur_bs, cfg.n_dummy_conds, device=cfg.device, dtype=torch.float32)
        chunks.append(
            ofm.sample(
                cfg.dims,
                conds=conds,
                n_channels=cfg.vis_channels,
                n_samples=cur_bs,
                n_eval=n_eval,
                method=cfg.sample_method,
            ).cpu()
        )
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


def plot_training_dataset(cfg: TrainConfig, x_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bin_edges = torch.linspace(-4.0, 4.0, 101, dtype=x_train.dtype, device=x_train.device)
    x_hist_true = averaged_sample_histogram(random_subset(x_train, cfg.hist_samples), bin_edges)
    fig, ax = plt.subplots(2, 5, figsize=(14, 6))
    ax = ax.reshape(-1)
    vmin = x_train[:1000].min().item()
    vmax = x_train[:1000].max().item()
    for i in range(10):
        im = ax[i].imshow(x_train[i, 0].cpu().numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    fig.colorbar(im, ax=ax.tolist(), shrink=0.7, location="right")
    fig.suptitle(f"Training samples ({cfg.label} input OFM-UNet prior)", y=0.98)
    fig.savefig(figure_path(cfg, "training_samples.png"), dpi=200)
    plt.close(fig)
    return x_hist_true, bin_edges


def evaluate_base_resolution(
    cfg: TrainConfig,
    ofm: OFMModel,
    x_train: torch.Tensor,
    x_hist_true: torch.Tensor,
    bin_edges: torch.Tensor,
) -> None:
    with torch.no_grad():
        conds = torch.zeros(10, cfg.n_dummy_conds, device=cfg.device, dtype=torch.float32)
        x_hat = ofm.sample(
            cfg.dims,
            conds=conds,
            n_channels=cfg.vis_channels,
            n_samples=10,
            n_eval=cfg.n_eval,
            method=cfg.sample_method,
        ).cpu()
        x_ground_truth = x_train[:10].cpu()
        vmin = min(x_ground_truth.min().item(), x_hat.min().item())
        vmax = max(x_ground_truth.max().item(), x_hat.max().item())

        fig, ax = plt.subplots(2, 5, figsize=(14, 6))
        ax = ax.reshape(-1)
        for i in range(10):
            ax[i].imshow(x_ground_truth[i, 0].numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax[i].set_title(f"GT #{i + 1}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.tight_layout()
        fig.savefig(figure_path(cfg, f"{cfg.label}_samples_base_gt.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(2, 5, figsize=(14, 6))
        ax = ax.reshape(-1)
        for i in range(10):
            ax[i].imshow(x_hat[i, 0].numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax[i].set_title(f"OFM #{i + 1}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.tight_layout()
        fig.savefig(figure_path(cfg, f"{cfg.label}_samples_base_ofm.png"), dpi=200)
        plt.close(fig)

        x_alt = sample_many(
            cfg,
            ofm,
            total_samples=max(cfg.eval_samples, cfg.autovar_samples),
            n_eval=cfg.n_eval,
            sample_batch_size=cfg.eval_batch_size,
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
        fig.savefig(figure_path(cfg, f"{cfg.label}_stats_base.png"), dpi=200)
        plt.close(fig)


def load_pde_input_dataset(data_path: str | Path, n_samples: int | None = None) -> torch.Tensor:
    array = np.load(data_path)
    x_train = torch.from_numpy(array).float()
    if x_train.ndim != 4:
        raise ValueError(f"Expected PDE data shape [N, C, H, W], got {tuple(x_train.shape)}")
    if x_train.shape[1] < 1:
        raise ValueError(f"Expected at least one channel, got shape {tuple(x_train.shape)}")
    x_train = x_train[:, 0:1]
    if n_samples is not None:
        x_train = x_train[:n_samples]
    return x_train


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    print(f"Cuda Device: {cfg.device}")
    print(f"Using output directory: {cfg.save_path}")
    print(f"Using figure directory: {cfg.figure_save_path}")
    mode = "evaluation-only" if args.eval_only else "train + evaluation"
    if args.skip_eval:
        mode = "train only" if not args.eval_only else "checkpoint validation only"
    print(f"Run mode: {mode}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    x_train = load_pde_input_dataset(cfg.data_path, n_samples=cfg.train_samples)
    print(f"Training data shape: {tuple(x_train.shape)}")

    loader_tr = DataLoader(
        PDEInputOFMDataset(x_train, cfg.n_dummy_conds),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    if not args.skip_eval:
        x_hist_true, bin_edges = plot_training_dataset(cfg, x_train)

    if not args.eval_only:
        train_ofm(cfg, loader_tr)
    elif not checkpoint_path(cfg).exists():
        raise FileNotFoundError(f"Checkpoint not found for --eval-only: {checkpoint_path(cfg)}")

    if args.skip_eval:
        return

    ofm = load_trained_ofm(cfg)
    eval_t0 = time.perf_counter()
    evaluate_base_resolution(
        cfg,
        ofm,
        x_train,
        x_hist_true,
        bin_edges,
    )
    eval_elapsed = time.perf_counter() - eval_t0
    print(f"Evaluation plotting + sampling time: {eval_elapsed:.2f} s")


if __name__ == "__main__":
    main()
