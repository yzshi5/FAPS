"""Super-resolution PDE inverse regression with an OFM prior and FNO surrogate."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PDE_EVAL_ROOT = Path(__file__).resolve().parent
if str(PDE_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PDE_EVAL_ROOT))

from faps_utils.fno import FNO
from faps_utils.ofm_ind_likelihood import OFMModel

import PDE_inverse as inverse

Tensor = torch.Tensor
PDE_INVERSE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SuperEvalConfig:
    pde: str
    prior_path: Path
    forward_ckpt: Path
    base_test_data_path: Path
    test_data_path: Path | None
    save_dir: Path
    save_prefix: str
    device: str
    seed: int
    test_sample_idx: int
    n_observations: int
    noise_level: float
    n_samples: int
    annealing_steps: int
    ode_steps: int
    langevin_steps: int
    langevin_lr: float
    tau: float
    low_rank_cov_rank: int
    low_rank_cov_samples: int
    low_rank_cov_ode_steps: int
    grad_clip_norm: Optional[float]
    state_clamp: Optional[float]
    base_n_x: int
    super_n_x: int
    modes: int
    width: int
    mlp_width: int
    epochs: int
    sigma_min: float
    kernel_length: float
    kernel_variance: float
    kernel_nu: float
    forward_modes: int
    forward_hidden_channels: int
    forward_projection_channels: int
    forward_n_layers: int

    @property
    def dims(self) -> list[int]:
        return [self.super_n_x, self.super_n_x]

    @property
    def label(self) -> str:
        return f"{self.pde.strip()} super"

    def inverse_config(self) -> inverse.EvalConfig:
        return inverse.EvalConfig(
            pde=self.label,
            prior_path=self.prior_path,
            forward_ckpt=self.forward_ckpt,
            test_data_path=self.test_data_path or self.base_test_data_path,
            save_dir=self.save_dir,
            save_prefix=self.save_prefix,
            device=self.device,
            seed=self.seed,
            test_sample_idx=self.test_sample_idx,
            n_observations=self.n_observations,
            noise_level=self.noise_level,
            n_samples=self.n_samples,
            annealing_steps=self.annealing_steps,
            ode_steps=self.ode_steps,
            langevin_steps=self.langevin_steps,
            langevin_lr=self.langevin_lr,
            tau=self.tau,
            low_rank_cov_rank=self.low_rank_cov_rank,
            low_rank_cov_samples=self.low_rank_cov_samples,
            low_rank_cov_ode_steps=self.low_rank_cov_ode_steps,
            grad_clip_norm=self.grad_clip_norm,
            state_clamp=self.state_clamp,
            n_x=self.super_n_x,
            modes=self.modes,
            width=self.width,
            mlp_width=self.mlp_width,
            epochs=self.epochs,
            sigma_min=self.sigma_min,
            kernel_length=self.kernel_length,
            kernel_variance=self.kernel_variance,
            kernel_nu=self.kernel_nu,
            forward_modes=self.forward_modes,
            forward_hidden_channels=self.forward_hidden_channels,
            forward_projection_channels=self.forward_projection_channels,
            forward_n_layers=self.forward_n_layers,
        )


def optional_path(value: str) -> Path | None:
    if value.strip() == "":
        return None
    return Path(value).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Super-resolution PDE inverse regression with an OFM prior and FNO surrogate."
    )
    parser.add_argument("--pde", type=str, required=True)
    parser.add_argument("--prior-path", type=Path, required=True)
    parser.add_argument("--forward-ckpt", type=Path, required=True)
    parser.add_argument("--base-test-data-path", type=Path, required=True)
    parser.add_argument(
        "--test-data-path",
        type=optional_path,
        default=None,
        help="Optional [N, 2, super_n_x, super_n_x] test data. If omitted, upsample base data.",
    )
    parser.add_argument("--save-dir", type=Path, default=PDE_INVERSE_ROOT / "outputs" / "PDE_inverse_sup")
    parser.add_argument("--save-prefix", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--test-sample-idx", type=int, default=100)
    parser.add_argument("--n-observations", type=int, default=125)
    parser.add_argument("--noise-level", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--annealing-steps", type=int, default=20)
    parser.add_argument("--ode-steps", type=int, default=10)
    parser.add_argument("--langevin-steps", type=int, default=40)
    parser.add_argument("--langevin-lr", type=float, default=4e-5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--low-rank-cov-rank", type=int, default=32)
    parser.add_argument("--low-rank-cov-samples", type=int, default=256)
    parser.add_argument("--low-rank-cov-ode-steps", type=int, default=20)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--state-clamp", type=float, default=None)
    parser.add_argument("--base-n-x", type=int, default=128)
    parser.add_argument("--super-n-x", type=int, default=160)
    parser.add_argument("--modes", type=int, default=48)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--kernel-length", type=float, default=0.01)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-nu", type=float, default=0.5)
    parser.add_argument("--forward-modes", type=int, default=48)
    parser.add_argument("--forward-hidden-channels", type=int, default=64)
    parser.add_argument("--forward-projection-channels", type=int, default=128)
    parser.add_argument("--forward-n-layers", type=int, default=4)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SuperEvalConfig:
    return SuperEvalConfig(
        pde=args.pde,
        prior_path=args.prior_path.expanduser(),
        forward_ckpt=args.forward_ckpt.expanduser(),
        base_test_data_path=args.base_test_data_path.expanduser(),
        test_data_path=args.test_data_path,
        save_dir=args.save_dir.expanduser(),
        save_prefix=args.save_prefix,
        device=args.device,
        seed=args.seed,
        test_sample_idx=args.test_sample_idx,
        n_observations=args.n_observations,
        noise_level=args.noise_level,
        n_samples=args.n_samples,
        annealing_steps=args.annealing_steps,
        ode_steps=args.ode_steps,
        langevin_steps=args.langevin_steps,
        langevin_lr=args.langevin_lr,
        tau=args.tau,
        low_rank_cov_rank=args.low_rank_cov_rank,
        low_rank_cov_samples=args.low_rank_cov_samples,
        low_rank_cov_ode_steps=args.low_rank_cov_ode_steps,
        grad_clip_norm=args.grad_clip_norm,
        state_clamp=args.state_clamp,
        base_n_x=args.base_n_x,
        super_n_x=args.super_n_x,
        modes=args.modes,
        width=args.width,
        mlp_width=args.mlp_width,
        epochs=args.epochs,
        sigma_min=args.sigma_min,
        kernel_length=args.kernel_length,
        kernel_variance=args.kernel_variance,
        kernel_nu=args.kernel_nu,
        forward_modes=args.forward_modes,
        forward_hidden_channels=args.forward_hidden_channels,
        forward_projection_channels=args.forward_projection_channels,
        forward_n_layers=args.forward_n_layers,
    )


def build_prior_model(cfg: SuperEvalConfig) -> OFMModel:
    if not cfg.prior_path.exists():
        raise FileNotFoundError(f"OFM prior checkpoint not found: {cfg.prior_path}")
    print(f"Loading OFM prior checkpoint from: {cfg.prior_path}")
    print(f"Building FNO OFM prior at super-resolution dims={cfg.dims}")

    model = FNO(
        cfg.modes,
        vis_channels=1,
        hidden_channels=cfg.width,
        proj_channels=cfg.mlp_width,
        x_dim=2,
        t_scaling=1,
    ).to(cfg.device)
    for param in model.parameters():
        param.requires_grad = False

    try:
        checkpoint = torch.load(cfg.prior_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(cfg.prior_path, map_location="cpu", weights_only=False)

    if hasattr(checkpoint, "state_dict"):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        checkpoint = {k: v for k, v in checkpoint.items() if k != "_metadata"}

    model.load_state_dict(checkpoint)
    return OFMModel(
        model,
        kernel_length=cfg.kernel_length,
        kernel_variance=cfg.kernel_variance,
        nu=cfg.kernel_nu,
        sigma_min=cfg.sigma_min,
        device=cfg.device,
        dims=cfg.dims,
    )


def load_real_test_case(cfg: SuperEvalConfig) -> tuple[Tensor, Tensor, str]:
    if cfg.test_data_path is None:
        raise ValueError("test_data_path is required for loading a real super-resolution test case.")
    array = np.load(cfg.test_data_path)
    x_test = torch.from_numpy(array).float()
    if x_test.ndim != 4 or x_test.shape[1] < 2:
        raise ValueError(f"Expected test data shape [N, 2, H, W], got {tuple(x_test.shape)}")
    if tuple(x_test.shape[-2:]) != tuple(cfg.dims):
        raise ValueError(f"Expected super data spatial shape {tuple(cfg.dims)}, got {tuple(x_test.shape[-2:])}.")
    if cfg.test_sample_idx < 0 or cfg.test_sample_idx >= x_test.shape[0]:
        raise IndexError(f"test_sample_idx={cfg.test_sample_idx} out of range [0, {x_test.shape[0] - 1}]")
    sample = x_test[cfg.test_sample_idx : cfg.test_sample_idx + 1]
    return sample[:, 0:1], sample[:, 1:2], f"data:{cfg.test_data_path}[{cfg.test_sample_idx}]"


@torch.no_grad()
def load_bicubic_test_case(
    cfg: SuperEvalConfig,
    forward_model: torch.nn.Module,
) -> tuple[Tensor, Tensor, str]:
    array = np.load(cfg.base_test_data_path)
    x_test = torch.from_numpy(array).float()
    if x_test.ndim != 4 or x_test.shape[1] < 1:
        raise ValueError(f"Expected base data shape [N, C, H, W], got {tuple(x_test.shape)}")
    if tuple(x_test.shape[-2:]) != (cfg.base_n_x, cfg.base_n_x):
        raise ValueError(
            f"Expected {cfg.base_n_x}x{cfg.base_n_x} base data, got {tuple(x_test.shape[-2:])}."
        )
    if cfg.test_sample_idx < 0 or cfg.test_sample_idx >= x_test.shape[0]:
        raise IndexError(f"test_sample_idx={cfg.test_sample_idx} out of range [0, {x_test.shape[0] - 1}]")

    x_base = x_test[cfg.test_sample_idx : cfg.test_sample_idx + 1, 0:1].to(
        device=cfg.device,
        dtype=torch.float32,
    )
    x_true = F.interpolate(x_base, size=tuple(cfg.dims), mode="bicubic", align_corners=False)
    y_true = forward_model(x_true)
    return x_true.cpu(), y_true.cpu(), f"bicubic:{cfg.base_test_data_path}[{cfg.test_sample_idx}]"


def load_or_generate_test_case(
    cfg: SuperEvalConfig,
    forward_model: torch.nn.Module,
) -> tuple[Tensor, Tensor, str]:
    if cfg.test_data_path is not None:
        return load_real_test_case(cfg)
    return load_bicubic_test_case(cfg, forward_model)


def main() -> None:
    cfg = build_config(parse_args())
    base_cfg = cfg.inverse_config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    t0 = time.perf_counter()
    print(
        "Super-resolution regression config: "
        f"pde={cfg.pde}, dims={cfg.dims}, sample_idx={cfg.test_sample_idx}, "
        f"n_obs={cfg.n_observations}, noise_var={cfg.noise_level}, n_samples={cfg.n_samples}, "
        f"anneal={cfg.annealing_steps}, ode_steps={cfg.ode_steps}, "
        f"langevin_steps={cfg.langevin_steps}, langevin_lr={cfg.langevin_lr}, "
        f"low_rank_cov_rank={cfg.low_rank_cov_rank}, grad_clip={cfg.grad_clip_norm}, "
        f"state_clamp={cfg.state_clamp}"
    )

    prior = build_prior_model(cfg)
    forward_model = inverse.build_forward_model(base_cfg)
    x_true_full, y_true_full, truth_source = load_or_generate_test_case(cfg, forward_model)
    print(f"Truth source: {truth_source}")
    print(f"Truth shapes: x={tuple(x_true_full.shape)}, y={tuple(y_true_full.shape)}")

    pos_mask, y_obs_part = inverse.create_random_observation(
        y_true_full=y_true_full,
        n_observations=cfg.n_observations,
        noise_level=cfg.noise_level,
        device=cfg.device,
    )

    posterior = inverse.run_surrogate_regression_FAPS(
        G=prior,
        forward_model=forward_model,
        dims=cfg.dims,
        pos_mask=pos_mask,
        y_obs_part=y_obs_part,
        noise_level=cfg.noise_level,
        n_samples=cfg.n_samples,
        annealing_steps=cfg.annealing_steps,
        ode_steps=cfg.ode_steps,
        langevin_steps=cfg.langevin_steps,
        langevin_lr=cfg.langevin_lr,
        tau=cfg.tau,
        low_rank_cov_rank=cfg.low_rank_cov_rank,
        low_rank_cov_samples=cfg.low_rank_cov_samples,
        low_rank_cov_ode_steps=cfg.low_rank_cov_ode_steps,
        grad_clip_norm=cfg.grad_clip_norm,
        state_clamp=cfg.state_clamp,
        device=cfg.device,
    )

    inverse.visualize_results(
        posterior=posterior,
        x_true_full=x_true_full,
        y_true_full=y_true_full,
        y_obs_part=y_obs_part,
        pos_mask=pos_mask,
        forward_model=forward_model,
        cfg=base_cfg,
    )
    elapsed = time.perf_counter() - t0
    print(f"Total super-resolution regression runtime: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
