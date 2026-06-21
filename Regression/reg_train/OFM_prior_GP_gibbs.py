"""Train and evaluate OFM on 1D Gibbs GP prior data."""

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REGRESSION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from faps_utils.fno import FNO
from faps_utils.ofm_ind_likelihood import OFMModel
from faps_utils.true_gaussian_process_gibbs import true_GPPrior_Gibbs as true_GPPriorGibbs


@dataclass(frozen=True)
class TrainConfig:
    save_path: Path
    checkpoint_name: str | None
    figure_save_path: Path
    device: str
    seed: int
    train_samples: int
    n_x: int
    modes: int
    width: int
    mlp_width: int
    kernel_length: float
    kernel_variance: float
    kernel_nu: float
    gibbs_l0: float
    gibbs_l1: float
    gibbs_sigma: float
    epochs: int
    sigma_min: float
    batch_size: int
    learning_rate: float
    eta_min: float
    super_n_x: int
    super_train_samples: int

    @property
    def dims(self) -> list[int]:
        return [self.n_x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate OFM on 1D Gibbs GP prior data."
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REGRESSION_ROOT / "outputs" / "OFM" / "GP_gibbs",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Optional checkpoint filename under --save-dir, e.g. GP_gibbs_epoch_500.pt.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=REGRESSION_ROOT / "outputs" / "OFM" / "GP_gibbs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--train-samples", type=int, default=40_000)
    parser.add_argument("--n-x", type=int, default=256)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--kernel-length", type=float, default=0.01)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-nu", type=float, default=0.5)
    parser.add_argument("--gibbs-l0", type=float, default=0.05)
    parser.add_argument("--gibbs-l1", type=float, default=0.25)
    parser.add_argument("--gibbs-sigma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--super-n-x", type=int, default=512)
    parser.add_argument("--super-train-samples", type=int, default=3000)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate from the existing checkpoint.",
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


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        save_path=resolve_output_path(args.save_dir),
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
        gibbs_l0=args.gibbs_l0,
        gibbs_l1=args.gibbs_l1,
        gibbs_sigma=args.gibbs_sigma,
        epochs=args.epochs,
        sigma_min=args.sigma_min,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eta_min=args.eta_min,
        super_n_x=args.super_n_x,
        super_train_samples=args.super_train_samples,
    )


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


def mean_autocovariance(samples: torch.Tensor, nlag: int) -> torch.Tensor:
    """Average autocovariance across samples of shape [N, L]."""
    acovf = [
        torch.tensor(sm.tsa.acovf(sample.cpu().numpy(), nlag=nlag), dtype=torch.float32)
        for sample in samples
    ]
    return torch.stack(acovf).mean(dim=0)


def build_model(cfg: TrainConfig) -> FNO:
    return FNO(
        cfg.modes,
        vis_channels=1,
        hidden_channels=cfg.width,
        proj_channels=cfg.mlp_width,
        x_dim=1,
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
        loss_log_filename="GP_gibbs_loss.csv",
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plot generated training samples and return mean/hist/acovf statistics."""
    x_mean_true = x_train.mean(dim=0).squeeze()
    x_hist_true, bin_edges = x_train.histogram(range=[-4, 4], density=True)
    x_acovf_true = mean_autocovariance(x_train[:, 0, :], nlag=50)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(10):
        x = x_train[i, 0]
        ax[0].plot(x)
        ax[1].plot(sm.tsa.acovf(x.cpu().numpy(), nlag=50))
    ax[0].plot(x_mean_true, c="k", lw=3)
    ax[0].set_ylim([-3, 3])
    ax[0].set_title("X samples")
    ax[1].plot(x_acovf_true, c="k", lw=3)
    ax[1].set_ylim([0.0, 1.0])
    ax[1].set_title("Autocovariance")
    plt.tight_layout()
    plt.savefig(figure_path(cfg, "training_samples.png"), dpi=200)
    plt.close(fig)

    return x_mean_true, x_hist_true, bin_edges, x_acovf_true


def evaluate_base_resolution(
    ofm: OFMModel,
    x_train: torch.Tensor,
    x_mean_true: torch.Tensor,
    x_hist_true: torch.Tensor,
    bin_edges: torch.Tensor,
    x_acovf_true: torch.Tensor,
    cfg: TrainConfig,
) -> None:
    x_pos = np.linspace(0, 1, cfg.n_x)
    with torch.no_grad():
        x_hat = ofm.sample([cfg.n_x], n_samples=10, n_eval=4).cpu().squeeze()
        x_ground_truth = x_train[:10].squeeze()

        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        for i in range(10):
            ax[0].plot(x_pos, x_ground_truth[i, :])
            ax[1].plot(x_pos, x_hat[i, :])
        ax[0].set_title("Ground Truth")
        ax[1].set_title("Operator Flow Matching (OFM)")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, "gp_samples_base.png"), dpi=200)
        plt.close(fig)

        x_alt = torch.vstack(
            [ofm.sample([cfg.n_x], n_samples=1000, n_eval=4).cpu().squeeze() for _ in range(3)]
        )
        x_hist, bin_edges_alt = x_alt.histogram(range=[-4, 4], density=True)
        x_mean = x_alt.mean(dim=0)
        x_acovf = mean_autocovariance(x_alt, nlag=50)

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].plot(x_pos, x_mean_true, c="k", lw=3, label="Ground Truth")
        ax[0].plot(x_pos, x_mean, c="r", ls="--", lw=3, label="OFM")
        ax[0].set_xlabel("Domain")
        ax[0].set_ylim([-4, 4])
        ax[0].set_title("Mean")
        ax[0].legend(loc="upper right")

        ax[1].plot(x_acovf_true, c="k", lw=3)
        ax[1].plot(x_acovf, c="r", ls="--", lw=3)
        ax[1].set_xlabel("Number of lags")
        ax[1].set_title("Autocovariance")

        ax[2].plot((bin_edges[1:] + bin_edges[:-1]) / 2, x_hist_true, c="k", lw=3)
        ax[2].plot((bin_edges_alt[1:] + bin_edges_alt[:-1]) / 2, x_hist, c="r", ls="--", lw=3)
        ax[2].set_title("Histogram")
        ax[2].set_xlabel("Value")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, "gp_stats_base.png"), dpi=200)
        plt.close(fig)


def evaluate_super_resolution(ofm: OFMModel, cfg: TrainConfig) -> None:
    n_x_sup = cfg.super_n_x
    nlag_sup = 200
    train_samples_sup = cfg.super_train_samples

    gp_sup = true_GPPriorGibbs(
        l0=cfg.gibbs_l0,
        l1=cfg.gibbs_l1,
        sigma=cfg.gibbs_sigma,
        device=cfg.device,
        dims=[n_x_sup],
    )
    x_train_sup = gp_sup.sample_train_data(dims=[n_x_sup], n_samples=train_samples_sup)
    x_hist_true_sup, bin_edges_sup = x_train_sup.histogram(range=[-4, 4], density=True)
    x_acovf_true_sup = mean_autocovariance(x_train_sup[:, 0, :], nlag=nlag_sup)

    with torch.no_grad():
        x_hat = ofm.sample([n_x_sup], n_samples=10, n_eval=10).cpu().squeeze()
        x_ground_truth = x_train_sup[:10].squeeze()

        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        for i in range(10):
            ax[0].plot(x_ground_truth[i, :])
            ax[1].plot(x_hat[i, :])
        ax[0].set_title(f"Ground Truth (resolution={n_x_sup})")
        ax[1].set_title("Operator Flow Matching (OFM)")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"gp_samples_sup_{n_x_sup}.png"), dpi=200)
        plt.close(fig)

        x_alt = torch.vstack(
            [
                ofm.sample([n_x_sup], n_samples=1000, n_eval=10).cpu().squeeze()
                for _ in range(3)
            ]
        )
        x_hist, bin_edges_alt = x_alt.histogram(range=[-4, 4], density=True)
        x_acovf = mean_autocovariance(x_alt, nlag=nlag_sup)

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].plot(x_acovf_true_sup, c="k", lw=3, label="Ground Truth")
        ax[0].plot(x_acovf, c="r", ls="--", lw=3, label="OFM")
        ax[0].set_xlabel("Number of lags")
        ax[0].set_title("Autocovariance")
        ax[0].legend(loc="upper right")

        ax[1].plot((bin_edges_sup[1:] + bin_edges_sup[:-1]) / 2, x_hist_true_sup, c="k", lw=3)
        ax[1].plot((bin_edges_alt[1:] + bin_edges_alt[:-1]) / 2, x_hist, c="r", ls="--", lw=3)
        ax[1].set_title("Histogram")
        ax[1].set_xlabel("Value")
        plt.tight_layout()
        plt.savefig(figure_path(cfg, f"gp_stats_sup_{n_x_sup}.png"), dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    print(f"Using output directory: {cfg.save_path}")
    print(f"Using figure directory: {cfg.figure_save_path}")
    mode = "evaluation-only" if args.eval_only else "train + evaluation"
    print(f"Run mode: {mode}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    gp = true_GPPriorGibbs(
        l0=cfg.gibbs_l0,
        l1=cfg.gibbs_l1,
        sigma=cfg.gibbs_sigma,
        device=cfg.device,
        dims=cfg.dims,
    )
    x_train = gp.sample_train_data(dims=cfg.dims, n_samples=cfg.train_samples)
    print(f"Training data shape: {tuple(x_train.shape)}")

    loader_tr = DataLoader(x_train, batch_size=cfg.batch_size, shuffle=True)
    x_mean_true, x_hist_true, bin_edges, x_acovf_true = plot_training_dataset(x_train, cfg)

    if not args.eval_only:
        train_ofm(loader_tr, cfg)
        finalize_saved_checkpoint(cfg)
    elif not checkpoint_path(cfg).exists() and not default_checkpoint_path(cfg).exists():
        raise FileNotFoundError(
            f"Checkpoint not found for --eval-only: {checkpoint_path(cfg)}"
        )

    ofm = load_trained_ofm(cfg)

    evaluate_base_resolution(
        ofm,
        x_train,
        x_mean_true,
        x_hist_true,
        bin_edges,
        x_acovf_true,
        cfg,
    )
    evaluate_super_resolution(ofm, cfg)


if __name__ == "__main__":
    main()
