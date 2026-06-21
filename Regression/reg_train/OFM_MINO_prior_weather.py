"""Train and evaluate an OFM prior with a MINO backbone on weather data."""

from __future__ import annotations

import argparse
import math
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

from faps_utils.models.mino_modules.conditioner_timestep import ConditionerTimestep
from faps_utils.models.mino_modules.decoder_perceiver import DecoderPerceiver
from faps_utils.models.mino_modules.encoder_supernodes_gno_cross_attention import (
    EncoderSupernodes,
)
from faps_utils.models.mino_transformer import MINO
from faps_utils.ofm_ind_seq_likelihood import OFMModel


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path
    save_path: Path
    checkpoint_name: str | None
    figure_save_path: Path
    device: str
    seed: int
    train_samples: int | None
    n_longs: int
    n_lats: int
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
    mino_x_dim: int
    mino_query_longs: int
    mino_query_lats: int
    mino_co_domain: int
    mino_radius: float
    mino_dim: int
    mino_num_heads: int
    mino_enc_depth: int
    mino_dec_depth: int
    load_pretrained_backbone: bool
    mino_backbone_path: Path

    @property
    def dims(self) -> list[int]:
        return [self.n_longs, self.n_lats]


def sphere_positions(num_longs: int, num_lats: int) -> np.ndarray:
    """Create xyz positions on a sphere from longitude/latitude grid."""
    longs, lats = np.mgrid[0 : 2 * np.pi : (num_longs + 1) * 1j, 0 : np.pi : num_lats * 1j]
    longs, lats = longs[:-1, :], lats[:-1, :]
    x = np.sin(lats) * np.cos(longs)
    y = np.sin(lats) * np.sin(longs)
    z = np.cos(lats)
    return np.c_[x.ravel(), y.ravel(), z.ravel()]


def latent_query_sphere(num_longs: int, num_lats: int) -> np.ndarray:
    """MINO query grid used in Climate_MINO_T (remove near-pole nodes)."""
    longs, lats = np.mgrid[0 : 2 * np.pi : (num_longs + 1) * 1j, 0 : np.pi : num_lats * 1j]
    longs, lats = longs[:-1, 2:-2], lats[:-1, 2:-2]
    x = np.sin(lats) * np.cos(longs)
    y = np.sin(lats) * np.sin(longs)
    z = np.cos(lats)
    return np.c_[x.ravel(), y.ravel(), z.ravel()]


class MINOBackboneWrapper(torch.nn.Module):
    """Adapt MINO(input_feat,input_pos,query_pos,timestep) to model(t, x)."""

    def __init__(self, mino: MINO, query_dims: tuple[int, int]):
        super().__init__()
        self.mino = mino
        query_pos = torch.tensor(
            latent_query_sphere(query_dims[0], query_dims[1] + 4), dtype=torch.float32
        ).T  # [3, Q]
        self.register_buffer("query_pos", query_pos, persistent=False)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] where H/W correspond to [num_longs, num_lats]
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, -1)
        input_pos = torch.tensor(
            sphere_positions(h, w), dtype=x.dtype, device=x.device
        ).T  # [3, H*W]
        pos = input_pos.unsqueeze(0).repeat(b, 1, 1)
        qpos = self.query_pos.unsqueeze(0).repeat(b, 1, 1)
        out_flat = self.mino(
            input_feat=x_flat,
            input_pos=pos,
            query_pos=qpos,
            timestep=t,
        )
        return out_flat.reshape(b, c, h, w)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate OFM on global weather climate field."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=REGRESSION_ROOT / "checkpoints" / "FAPS_prior")
    parser.add_argument("--checkpoint-name", type=str, default="weather_epoch_200.pt")
    parser.add_argument("--figure-dir", type=Path, default=REGRESSION_ROOT / "outputs" / "OFM" / "weather")
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--n-longs", type=int, default=90)
    parser.add_argument("--n-lats", type=int, default=46)
    parser.add_argument("--kernel-length", type=float, default=0.05)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-nu", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--sr-eval-samples", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--sr-eval-batch-size", type=int, default=16)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--sample-method", type=str, default="euler")
    parser.add_argument("--mino-x-dim", type=int, default=3)
    parser.add_argument("--mino-query-longs", type=int, default=32)
    parser.add_argument("--mino-query-lats", type=int, default=16)
    parser.add_argument("--mino-co-domain", type=int, default=1)
    parser.add_argument("--mino-radius", type=float, default=0.2)
    parser.add_argument("--mino-dim", type=int, default=256)
    parser.add_argument("--mino-num-heads", type=int, default=4)
    parser.add_argument("--mino-enc-depth", type=int, default=5)
    parser.add_argument("--mino-dec-depth", type=int, default=2)
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
    parser.add_argument(
        "--load-pretrained-backbone",
        action="store_true",
        help="Initialize MINO backbone from a pretrained checkpoint before OFM training.",
    )
    parser.add_argument(
        "--mino-backbone-path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "MINO_T_Climate" / "epoch_200.pt",
        help="Path to pretrained MINO checkpoint (used only with --load-pretrained-backbone).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_path=args.data_path.expanduser(),
        save_path=resolve_output_path(args.save_dir.expanduser()),
        checkpoint_name=args.checkpoint_name,
        figure_save_path=args.figure_dir.expanduser(),
        device=args.device,
        seed=args.seed,
        train_samples=args.train_samples,
        n_longs=args.n_longs,
        n_lats=args.n_lats,
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
        mino_x_dim=args.mino_x_dim,
        mino_query_longs=args.mino_query_longs,
        mino_query_lats=args.mino_query_lats,
        mino_co_domain=args.mino_co_domain,
        mino_radius=args.mino_radius,
        mino_dim=args.mino_dim,
        mino_num_heads=args.mino_num_heads,
        mino_enc_depth=args.mino_enc_depth,
        mino_dec_depth=args.mino_dec_depth,
        load_pretrained_backbone=args.load_pretrained_backbone,
        mino_backbone_path=args.mino_backbone_path.expanduser(),
    )


def resolve_output_path(save_dir: Path) -> Path:
    """Create and validate the requested model output directory."""
    save_dir.mkdir(parents=True, exist_ok=True)
    probe = save_dir / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return save_dir


def figure_path(cfg: TrainConfig, filename: str) -> Path:
    """Build figure output path under local quick-check folder."""
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
    """Generate samples in chunks to reduce eval-time memory pressure."""
    chunks = []
    remaining = int(total_samples)
    while remaining > 0:
        cur_bs = min(sample_batch_size, remaining)
        chunks.append(ofm.sample(dims, n_samples=cur_bs, n_eval=n_eval, method=method).cpu())
        remaining -= cur_bs
    return torch.cat(chunks, dim=0)


def flip_xy(field_2d: np.ndarray) -> np.ndarray:
    """Swap x/y axes for visualization."""
    return np.swapaxes(field_2d, -2, -1)


def build_model(
    cfg: TrainConfig,
    show_init_message: bool = True,
) -> torch.nn.Module:
    conditioner = ConditionerTimestep(dim=cfg.mino_dim)
    mino = MINO(
        conditioner=conditioner,
        encoder=EncoderSupernodes(
            input_dim=cfg.mino_co_domain,
            ndim=cfg.mino_x_dim,
            radius=cfg.mino_radius,
            enc_dim=cfg.mino_dim,
            enc_num_heads=cfg.mino_num_heads,
            enc_depth=cfg.mino_enc_depth,
            cond_dim=conditioner.cond_dim,
        ),
        decoder=DecoderPerceiver(
            input_dim=cfg.mino_dim,
            output_dim=cfg.mino_co_domain,
            ndim=cfg.mino_x_dim,
            dim=cfg.mino_dim,
            num_heads=cfg.mino_num_heads,
            depth=cfg.mino_dec_depth,
            unbatch_mode="dense_to_sparse_unpadded",
            cond_dim=conditioner.cond_dim,
        ),
    )

    if cfg.load_pretrained_backbone:
        ckpt_path = cfg.mino_backbone_path
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"MINO backbone checkpoint not found: {ckpt_path}. "
                "Set --mino-backbone-path to your checkpoint path."
            )
        if show_init_message:
            print(f"Loading MINO backbone checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if hasattr(ckpt, "state_dict"):
            ckpt = ckpt.state_dict()
        if isinstance(ckpt, dict):
            ckpt = {k: v for k, v in ckpt.items() if k != "_metadata"}
        mino.load_state_dict(ckpt, strict=False)
    else:
        if show_init_message:
            print("Using randomly initialized MINO backbone (training from scratch).")

    query_dims = (cfg.mino_query_longs, cfg.mino_query_lats)
    return MINOBackboneWrapper(mino=mino, query_dims=query_dims).to(cfg.device)


def build_ofm(model: torch.nn.Module, cfg: TrainConfig) -> OFMModel:
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
        loss_log_filename="weather_loss.csv",
    )


def load_trained_ofm(cfg: TrainConfig) -> OFMModel:
    model = build_model(cfg, show_init_message=False)
    for param in model.parameters():
        param.requires_grad = False

    model_path = checkpoint_path(cfg)
    if not model_path.exists() and cfg.checkpoint_name:
        model_path = default_checkpoint_path(cfg)
    print(f"Loading trained OFM checkpoint for evaluation: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Backward compatibility for checkpoints saved before weights_only format.
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if hasattr(checkpoint, "state_dict"):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict):
        # Some legacy checkpoints include serialization metadata as a dict key.
        checkpoint = {k: v for k, v in checkpoint.items() if k != "_metadata"}

    model.load_state_dict(checkpoint)
    return build_ofm(model, cfg)


def plot_training_dataset(
    x_train: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plot climate-field training samples and return mean/hist statistics."""
    x_mean_true = x_train.mean(dim=0).squeeze(0)
    x_hist_true, bin_edges = x_train.histogram(range=[-4, 4], density=True)

    n_plot = 6
    n_cols = 3
    n_rows = math.ceil(n_plot / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows))
    ax = ax.reshape(-1)
    vmin = x_train[:1000].min().item()
    vmax = x_train[:1000].max().item()
    for i in range(n_plot):
        im = ax[i].imshow(flip_xy(x_train[i, 0].cpu().numpy()), cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    for i in range(n_plot, n_rows * n_cols):
        ax[i].axis("off")
    fig.colorbar(im, ax=ax.tolist(), shrink=0.7, location="right")
    fig.suptitle("Training samples (Global weather climate field)", y=0.98)
    #plt.tight_layout()
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
        n_plot = 6
        x_hat = ofm.sample(cfg.dims, n_samples=n_plot, n_eval=cfg.n_eval, method=cfg.sample_method).cpu()
        x_ground_truth = x_train[:n_plot].cpu()
        vmin = min(x_ground_truth.min().item(), x_hat.min().item())
        vmax = max(x_ground_truth.max().item(), x_hat.max().item())

        n_cols = 3
        n_rows = math.ceil(n_plot / n_cols)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows))
        ax = ax.reshape(-1)
        for i in range(n_plot):
            ax[i].imshow(flip_xy(x_ground_truth[i, 0].numpy()), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax[i].set_title(f"GT #{i+1}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        for i in range(n_plot, n_rows * n_cols):
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, "weather_samples_base_gt.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows))
        ax = ax.reshape(-1)
        for i in range(n_plot):
            ax[i].imshow(flip_xy(x_hat[i, 0].numpy()), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax[i].set_title(f"OFM #{i+1}")
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        for i in range(n_plot, n_rows * n_cols):
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, "weather_samples_base_ofm.png"), dpi=200)
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
        im0 = ax[0].imshow(flip_xy(x_mean_true.cpu().numpy()), cmap="RdBu_r")
        ax[0].set_title(f"GT mean field ({cfg.n_longs}x{cfg.n_lats})")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(flip_xy(x_mean.cpu().numpy()), cmap="RdBu_r")
        ax[1].set_title(f"OFM mean field ({cfg.n_longs}x{cfg.n_lats})")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        ax[2].plot((bin_edges[1:] + bin_edges[:-1]) / 2, x_hist_true, c="k", lw=2, label="GT")
        ax[2].plot((bin_edges_alt[1:] + bin_edges_alt[:-1]) / 2, x_hist, c="r", ls="--", lw=2, label="OFM")
        ax[2].set_title("Value histogram")
        ax[2].set_xlabel("Value")
        ax[2].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, "weather_stats_base.png"), dpi=200)
        plt.close(fig)


def evaluate_super_resolution(
    ofm: OFMModel,
    x_hist_true: torch.Tensor,
    bin_edges: torch.Tensor,
    cfg: TrainConfig,
) -> None:
    # Follow Climate_MINO_T super-resolution convention (gen_sup_pos):
    # Base [n_longs, n_lats] -> super [2*n_longs, 2*n_lats].
    n_longs_sup = cfg.n_longs * 2
    n_lats_sup = cfg.n_lats * 2
    dims_sup = [n_longs_sup, n_lats_sup]
    with torch.no_grad():
        n_plot = 6
        x_hat = ofm.sample(dims_sup, n_samples=n_plot, n_eval=cfg.n_eval, method=cfg.sample_method).cpu()
        n_cols = 3
        n_rows = math.ceil(n_plot / n_cols)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows))
        ax = ax.reshape(-1)
        vmin = x_hat.min().item()
        vmax = x_hat.max().item()
        for i in range(n_plot):
            ax[i].imshow(flip_xy(x_hat[i, 0].numpy()), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(f"OFM SR #{i+1}")
        for i in range(n_plot, n_rows * n_cols):
            ax[i].axis("off")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"weather_samples_sup_{n_longs_sup}x{n_lats_sup}.png"), dpi=200)
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
        im0 = ax[0].imshow(flip_xy(x_mean.cpu().numpy()), cmap="RdBu_r")
        ax[0].set_title(f"OFM mean field ({n_longs_sup}x{n_lats_sup})")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        ax[1].plot(
            (bin_edges[1:] + bin_edges[:-1]) / 2,
            x_hist_true,
            c="k",
            lw=2,
            label=f"Train {cfg.n_longs}x{cfg.n_lats}",
        )
        ax[1].plot(
            (bin_edges_alt[1:] + bin_edges_alt[:-1]) / 2,
            x_hist,
            c="r",
            ls="--",
            lw=2,
            label=f"OFM {n_longs_sup}x{n_lats_sup}",
        )
        ax[1].set_title("Value histogram")
        ax[1].set_xlabel("Value")
        ax[1].legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"weather_stats_sup_{n_longs_sup}x{n_lats_sup}.png"), dpi=200)
        plt.close(fig)


def load_weather_dataset(data_path: str | Path, n_samples: int | None = None) -> torch.Tensor:
    array = np.load(data_path)
    x_train = torch.from_numpy(array).float()  # [N, 3, 46, 90]
    if x_train.ndim != 4 or x_train.shape[1] < 3:
        raise ValueError(
            f"Expected weather data shape [N, 3, 46, 90], got {tuple(x_train.shape)}"
        )
    # Channel 2 is the climate scalar field; channels 0/1 are lon/lat coordinates.
    x_train = x_train[:, 2:3]  # [N, 1, 46, 90]
    # Match Climate_MINO_T convention: [N, 1, 90, 46] before flattening.
    x_train = x_train.permute(0, 1, 3, 2)
    if n_samples is not None:
        x_train = x_train[:n_samples]
    return x_train


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    print(f"Using output directory: {cfg.save_path}")
    print(f"Using checkpoint name: {checkpoint_path(cfg).name}")
    print(f"Using figure directory: {cfg.figure_save_path}")
    mode = "evaluation-only" if args.eval_only else "train + evaluation"
    print(f"Run mode: {mode}")
    if not args.eval_only:
        backbone_mode = (
            "pretrained"
            if cfg.load_pretrained_backbone
            else "scratch (random init)"
        )
        print(f"Backbone init mode: {backbone_mode}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    x_train = load_weather_dataset(cfg.data_path, n_samples=cfg.train_samples)
    print(f"Training data shape: {tuple(x_train.shape)}")

    loader_tr = DataLoader(x_train, batch_size=cfg.batch_size, shuffle=True)

    if not args.eval_only:
        train_ofm(loader_tr, cfg)
        finalize_saved_checkpoint(cfg)
    elif not checkpoint_path(cfg).exists() and not default_checkpoint_path(cfg).exists():
        raise FileNotFoundError(
            f"Checkpoint not found for --eval-only: {checkpoint_path(cfg)}"
        )

    if args.skip_eval:
        print("Skipping evaluation plots (--skip-eval).")
        return

    x_mean_true, x_hist_true, bin_edges = plot_training_dataset(x_train, cfg)
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
