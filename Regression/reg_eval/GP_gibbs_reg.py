from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import torch
from torchdiffeq import odeint

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REGRESSION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from faps_utils.fno import FNO
from faps_utils.ofm_ind_likelihood import OFMModel
from faps_utils.true_gaussian_process_gibbs import (
    gibbs_kernel_cov,
    true_GPPrior_Gibbs,
)

Tensor = torch.Tensor

@dataclass(frozen=True)
class EvalConfig:
    checkpoint_path: Optional[Path]
    model_dir: Path
    flat_checkpoint_dir: Path
    results_dir: Path
    device: torch.device
    seed: int
    modes: int
    width: int
    mlp_width: int
    epochs: int
    sigma_min: float
    kernel_length: float
    kernel_variance: float
    kernel_nu: float
    n_x: int
    gibbs_l0: float
    gibbs_l1: float
    gibbs_sigma: float
    test_sample_idx: int
    n_test: int
    n_obs: int
    noise_level: float
    n_samples: int
    annealing_steps: int
    ode_steps: int
    langevin_steps: int
    langevin_lr: float
    tau: float
    anchor_std_base: float
    anchor_std_scale: float
    low_rank_cov_rank: int
    low_rank_cov_samples: int
    low_rank_cov_ode_steps: int

    @property
    def dims(self) -> list[int]:
        return [self.n_x]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OFM-FAPS masked regression with a Gibbs GP prior.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional direct path to a checkpoint.")
    parser.add_argument("--model-dir", type=Path, default=REGRESSION_ROOT / "checkpoints" / "FAPS_prior")
    parser.add_argument("--flat-checkpoint-dir", type=Path, default=REGRESSION_ROOT / "checkpoints" / "FAPS_prior")
    parser.add_argument("--results-dir", type=Path, default=REGRESSION_ROOT / "outputs" / "Regression_results" / "GP_gibbs_reg")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--kernel-length", type=float, default=0.01)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-nu", type=float, default=0.5)
    parser.add_argument("--n-x", type=int, default=512)
    parser.add_argument("--gibbs-l0", type=float, default=0.05)
    parser.add_argument("--gibbs-l1", type=float, default=0.25)
    parser.add_argument("--gibbs-sigma", type=float, default=1.0)
    parser.add_argument("--test-sample-idx", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-obs", type=int, default=7)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--annealing-steps", type=int, default=40)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--langevin-steps", type=int, default=50)
    parser.add_argument("--langevin-lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--anchor-std-base", type=float, default=0.05)
    parser.add_argument("--anchor-std-scale", type=float, default=1.0)
    parser.add_argument("--low-rank-cov-rank", type=int, default=32)
    parser.add_argument("--low-rank-cov-samples", type=int, default=512)
    parser.add_argument("--low-rank-cov-ode-steps", type=int, default=24)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        checkpoint_path=args.checkpoint.expanduser() if args.checkpoint else None,
        model_dir=args.model_dir.expanduser(),
        flat_checkpoint_dir=args.flat_checkpoint_dir.expanduser(),
        results_dir=args.results_dir.expanduser(),
        device=torch.device(args.device),
        seed=args.seed,
        modes=args.modes,
        width=args.width,
        mlp_width=args.mlp_width,
        epochs=args.epochs,
        sigma_min=args.sigma_min,
        kernel_length=args.kernel_length,
        kernel_variance=args.kernel_variance,
        kernel_nu=args.kernel_nu,
        n_x=args.n_x,
        gibbs_l0=args.gibbs_l0,
        gibbs_l1=args.gibbs_l1,
        gibbs_sigma=args.gibbs_sigma,
        test_sample_idx=args.test_sample_idx,
        n_test=args.n_test,
        n_obs=args.n_obs,
        noise_level=args.noise_level,
        n_samples=args.n_samples,
        annealing_steps=args.annealing_steps,
        ode_steps=args.ode_steps,
        langevin_steps=args.langevin_steps,
        langevin_lr=args.langevin_lr,
        tau=args.tau,
        anchor_std_base=args.anchor_std_base,
        anchor_std_scale=args.anchor_std_scale,
        low_rank_cov_rank=args.low_rank_cov_rank,
        low_rank_cov_samples=args.low_rank_cov_samples,
        low_rank_cov_ode_steps=args.low_rank_cov_ode_steps,
    )


def resolve_checkpoint_path(cfg: EvalConfig) -> Path:
    if cfg.checkpoint_path is not None:
        if cfg.checkpoint_path.exists():
            return cfg.checkpoint_path
        raise FileNotFoundError(f"Could not find checkpoint: {cfg.checkpoint_path}")
    candidates = [
        cfg.model_dir / f"epoch_{cfg.epochs}.pt",
        cfg.model_dir / f"GP_gibbs_epoch_{cfg.epochs}.pt",
        cfg.flat_checkpoint_dir / f"GP_gibbs_epoch_{cfg.epochs}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Gibbs prior checkpoint. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def build_prior_model(cfg: EvalConfig) -> OFMModel:
    model_checkpoint_path = resolve_checkpoint_path(cfg)
    print(f"Loading OFM prior checkpoint from: {model_checkpoint_path}")

    model = FNO(
        cfg.modes,
        vis_channels=1,
        hidden_channels=cfg.width,
        proj_channels=cfg.mlp_width,
        x_dim=1,
        t_scaling=1,
    ).to(cfg.device)
    for param in model.parameters():
        param.requires_grad = False

    try:
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)

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


def build_regression_case(cfg: EvalConfig) -> tuple[Tensor, Tensor, Tensor, np.ndarray]:
    gp = true_GPPrior_Gibbs(
        l0=cfg.gibbs_l0,
        l1=cfg.gibbs_l1,
        sigma=cfg.gibbs_sigma,
        device="cpu",
        dims=cfg.dims,
    )
    x_test = gp.sample_from_prior(dims=cfg.dims, n_samples=cfg.n_test)

    pos_mask = torch.zeros(cfg.n_x, dtype=torch.bool)
    pos_idx = np.random.choice(cfg.n_x, cfg.n_obs, replace=False)
    pos_idx.sort()
    pos_mask[torch.from_numpy(pos_idx)] = True

    u_obs_full = x_test[cfg.test_sample_idx : cfg.test_sample_idx + 1, 0:1]
    u_obs_part = u_obs_full[:, :, pos_mask].to(cfg.device)

    noise_pattern = torch.randn_like(u_obs_part) * math.sqrt(float(cfg.noise_level))
    u_obs_part = u_obs_part + noise_pattern
    return u_obs_full.squeeze(), u_obs_part, pos_mask, pos_idx


@dataclass
class FAPSSamples:
    samples: Tensor
    mean: Tensor
    std: Tensor
    trajectory: Optional[List[Tuple[float, Tensor]]] = None


class OFMMaskedFAPSSampler:
    """
    FAPS-style posterior sampler for functional regression with a masking operator.

    This class is designed to work with the `OFMModel` API used in SPL_OFM:
      - `G.model(t, x)` is the learned velocity field
      - `G.gp.sample(dims, n_samples, n_channels)` draws GP samples
      - `G.gp.base_dist.scale_tril` (or `G.gp.new_dist(dims).scale_tril`) encodes the GP covariance
      - `G.sigma_min` is the small path noise used during OFM training

    Observation model:
        y = A u + eta,
        eta ~ N(0, sigma_y^2 I),
    where A is a masking operator selecting a subset of function values.

    Covariance handling follows the v-pred sampler:
      - Bridge/base noise always uses Sigma_0 (the RF base covariance).
      - Posterior correction covariance uses gp_ref if provided, else a low-rank
        surrogate estimated from clean samples (or identity fallback).
    """

    def __init__(
        self,
        G,
        dims: Sequence[int],
        pos_mask: Tensor,
        u_obs_part: Tensor,
        noise_level: float,
        *,
        gp_ref: Optional[true_GPPrior_Gibbs] = None,
        tau: float = 1.0,
        exact_linear_update: bool = False,
        anchor_std_base: float = 0.05,
        anchor_std_scale: float = 1.0,
        low_rank_cov_rank: int = 32,
        low_rank_cov_samples: int = 256,
        low_rank_cov_ode_steps: int = 24,
        low_rank_cov_jitter: float = 1e-5,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.G = G
        self.dims = list(dims)
        self.n_points = int(np.prod(self.dims))
        self.device = device or getattr(G, "device", ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype

        self.pos_mask = self._normalize_mask(pos_mask).to(self.device)
        self.obs_idx = torch.where(self.pos_mask)[0]
        self.u_obs_part = self._normalize_obs(u_obs_part).to(self.device, dtype=self.dtype)
        self.obs_noise_var = float(noise_level)
        self.tau = float(tau)
        self.exact_linear_update = bool(exact_linear_update)
        self.anchor_std_base = float(anchor_std_base)
        self.anchor_std_scale = float(anchor_std_scale)
        self.low_rank_cov_rank = int(max(0, low_rank_cov_rank))
        self.low_rank_cov_samples = int(max(0, low_rank_cov_samples))
        self.low_rank_cov_ode_steps = int(max(1, low_rank_cov_ode_steps))
        self.low_rank_cov_jitter = float(max(0.0, low_rank_cov_jitter))

        # Match the GP distribution to the requested discretization.
        if list(getattr(self.G.gp, "dims", [])) == self.dims:
            self.gp_dist = self.G.gp.base_dist
        else:
            self.gp_dist = self.G.gp.new_dist(self.dims)

        # Base bridge covariance (Sigma_0): RF maps N(0, Sigma_0) -> data.
        self.base_scale_tril = self.gp_dist.scale_tril.to(self.device, dtype=self.dtype)
        self.n_channels = int(self.u_obs_part.shape[1])
        self.use_gp_cov = gp_ref is not None
        if self.use_gp_cov:
            if list(getattr(gp_ref, "dims", [])) == self.dims:
                gp_ref_dist = gp_ref.base_dist
            else:
                gp_ref_dist = gp_ref.new_dist(self.dims)
            self.cov_scale_tril = gp_ref_dist.scale_tril.to(self.device, dtype=self.dtype)
            self.gp_cov = (self.cov_scale_tril @ self.cov_scale_tril.T).to(self.device, dtype=self.dtype)
            eye = torch.eye(self.n_points, device=self.device, dtype=self.dtype)
            self.gp_cov = 0.5 * (self.gp_cov + self.gp_cov.T) + self.low_rank_cov_jitter * eye
            self.cov_scale_tril = torch.linalg.cholesky(self.gp_cov)
            self.sigma_uo = self.gp_cov[:, self.obs_idx]
            self.sigma_oo = self.gp_cov[self.obs_idx][:, self.obs_idx]
            print("Using covariance source: gp_ref")
        else:
            self.gp_cov = self._estimate_low_rank_covariance() if self.low_rank_cov_rank > 0 else None
            if self.gp_cov is not None:
                eye = torch.eye(self.n_points, device=self.device, dtype=self.dtype)
                self.gp_cov = 0.5 * (self.gp_cov + self.gp_cov.T) + self.low_rank_cov_jitter * eye
                self.cov_scale_tril = torch.linalg.cholesky(self.gp_cov)
                self.sigma_uo = self.gp_cov[:, self.obs_idx]
                self.sigma_oo = self.gp_cov[self.obs_idx][:, self.obs_idx]
                print(
                    f"Using covariance source: low-rank surrogate "
                    f"(rank={self.low_rank_cov_rank}, samples={self.low_rank_cov_samples})"
                )
            else:
                self.cov_scale_tril = None
                self.sigma_uo = None
                self.sigma_oo = None
                print("Using covariance source: identity (no gp_ref / no low-rank surrogate)")

    @torch.no_grad()
    def _estimate_low_rank_covariance(self) -> Optional[Tensor]:
        if self.low_rank_cov_samples < 2:
            return None
        n = self.low_rank_cov_samples
        u0 = self.bridge_std(0.0) * self._sample_base_gp(n)
        u1 = self.endpoint_anchor(
            u0,
            t_start=0.0,
            ode_steps=self.low_rank_cov_ode_steps,
            ode_method="euler",
            rtol=1e-5,
            atol=1e-5,
        )
        u_flat = self._flatten(u1).reshape(n * self.n_channels, self.n_points)
        if u_flat.shape[0] < 2:
            return None
        u_centered = u_flat - u_flat.mean(dim=0, keepdim=True)
        _, s, vh = torch.linalg.svd(u_centered, full_matrices=False)
        eigvals = (s * s) / max(u_centered.shape[0] - 1, 1)
        rank = min(self.low_rank_cov_rank, eigvals.shape[0], vh.shape[0])
        if rank <= 0:
            return None
        basis = vh[:rank].T
        top_eigs = eigvals[:rank]
        cov_lr = (basis * top_eigs.unsqueeze(0)) @ basis.T
        if rank < eigvals.shape[0]:
            res_var = torch.mean(eigvals[rank:])
            cov_lr = cov_lr + res_var * torch.eye(self.n_points, device=self.device, dtype=self.dtype)
        return cov_lr

    def _normalize_mask(self, mask: Tensor) -> Tensor:
        mask = torch.as_tensor(mask)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.numel() != self.n_points:
            raise ValueError(f"pos_mask has {mask.numel()} entries, expected {self.n_points}.")
        return mask.reshape(-1)

    def _normalize_obs(self, u_obs_part: Tensor) -> Tensor:
        u_obs_part = torch.as_tensor(u_obs_part)
        if u_obs_part.ndim == 2:
            u_obs_part = u_obs_part.unsqueeze(0)
        if u_obs_part.ndim != 3:
            raise ValueError(
                f"u_obs_part must have shape [1, C, n_obs] or [C, n_obs], got {tuple(u_obs_part.shape)}."
            )
        if u_obs_part.shape[-1] != int(self.pos_mask.sum()):
            raise ValueError(
                f"u_obs_part last dimension is {u_obs_part.shape[-1]}, expected {int(self.pos_mask.sum())}."
            )
        return u_obs_part

    def _flatten(self, u: Tensor) -> Tensor:
        return u.reshape(u.shape[0], u.shape[1], -1)

    def _unflatten(self, u_flat: Tensor) -> Tensor:
        return u_flat.reshape(u_flat.shape[0], u_flat.shape[1], *self.dims)

    def _sample_base_gp(self, n_samples: int, n_channels: Optional[int] = None) -> Tensor:
        n_channels = self.n_channels if n_channels is None else n_channels
        return self.gp_dist.sample(sample_shape=torch.Size([n_samples * n_channels])).reshape(
            n_samples, n_channels, *self.dims
        ).to(self.device, dtype=self.dtype)

    def _sample_white(self, n_samples: int) -> Tensor:
        return torch.randn(n_samples, self.n_channels, *self.dims, device=self.device, dtype=self.dtype)

    def _sample_bridge_base(self, n_samples: int) -> Tensor:
        if self.cov_scale_tril is None:
            return self._sample_white(n_samples)
        eps = torch.randn(
            n_samples * self.n_channels,
            self.n_points,
            device=self.device,
            dtype=self.dtype,
        )
        gp_eps = eps @ self.cov_scale_tril.T
        return gp_eps.reshape(n_samples, self.n_channels, *self.dims)

    def _apply_mask(self, u: Tensor) -> Tensor:
        u_flat = self._flatten(u)
        return u_flat[..., self.pos_mask]

    def _loglik_grad(self, u: Tensor) -> Tensor:
        """Gradient of log p(y | A u) for a masking operator A."""
        u_flat = self._flatten(u)
        grad = torch.zeros_like(u_flat)

        y = self.u_obs_part.expand(u.shape[0], -1, -1)
        residual = self._apply_mask(u) - y
        grad[..., self.pos_mask] = -residual / self.obs_noise_var
        return self._unflatten(grad)

    def _cov_matvec(self, x: Tensor) -> Tensor:
        """Apply selected covariance to x without materializing Sigma explicitly."""
        if self.cov_scale_tril is None:
            return x
        x_flat = self._flatten(x)  # [B, C, N]
        B, C, N = x_flat.shape
        x_bc = x_flat.reshape(B * C, N)
        # row-wise x @ Sigma = ((x @ L) @ L^T)
        out = (x_bc @ self.cov_scale_tril) @ self.cov_scale_tril.T
        return out.reshape(B, C, *self.dims)

    def bridge_std(self, t: float) -> float:
        sigma_min = float(getattr(self.G, "sigma_min", 0.0))
        return math.sqrt((1.0 - float(t)) ** 2 + sigma_min**2)

    def bridge_var(self, t: float) -> float:
        std_t = self.bridge_std(t)
        return std_t * std_t

    def anchor_std(self, t: float) -> float:
        return max(self.anchor_std_base, self.anchor_std_scale * (1.0 - float(t)))

    @torch.no_grad()
    def endpoint_anchor(
        self,
        u_t: Tensor,
        t_start: float,
        ode_steps: int = 4,
        ode_method: str = "euler",
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> Tensor:
        if not (0.0 <= t_start <= 1.0):
            raise ValueError(f"t_start must be in [0, 1], got {t_start}.")
        if abs(t_start - 1.0) < 1e-12:
            return u_t.clone()
        t_span = torch.linspace(t_start, 1.0, ode_steps + 1, device=self.device, dtype=self.dtype)
        out = odeint(self.G.model, u_t, t_span, method=ode_method, rtol=rtol, atol=atol)
        return out[-1]

    @torch.no_grad()
    def rebridge(self, u_clean: Tensor, t_next: float) -> Tensor:
        if t_next >= 1.0:
            return u_clean
        return float(t_next) * u_clean + self.bridge_std(t_next) * self._sample_base_gp(u_clean.shape[0])

    def _exact_linear_gaussian_update(self, u_hat: Tensor, bridge_var_t: float) -> Tensor:
        y = self.u_obs_part.expand(u_hat.shape[0], -1, -1)
        u_hat_flat = self._flatten(u_hat)
        residual = y - u_hat_flat[..., self.pos_mask]
        bc = u_hat.shape[0] * self.n_channels
        residual_bc = residual.reshape(bc, -1)

        eye_obs = torch.eye(int(self.obs_idx.numel()), device=self.device, dtype=self.dtype)
        if self.use_gp_cov:
            assert self.sigma_uo is not None and self.sigma_oo is not None and self.gp_cov is not None
            K = bridge_var_t * self.sigma_oo + self.obs_noise_var * eye_obs
            alpha = torch.linalg.solve(K, residual_bc.T).T
            mean_shift = alpha @ (bridge_var_t * self.sigma_uo).T
            post_mean_flat = u_hat_flat.reshape(bc, -1) + mean_shift

            cov_post = bridge_var_t * self.gp_cov - (bridge_var_t**2) * self.sigma_uo @ torch.linalg.solve(
                K, self.sigma_uo.T
            )
            cov_post = 0.5 * (cov_post + cov_post.T) + 1e-6 * torch.eye(self.n_points, device=self.device, dtype=self.dtype)
            chol_post = torch.linalg.cholesky(cov_post)
            eps = torch.randn(bc, self.n_points, device=self.device, dtype=self.dtype)
            post_flat = post_mean_flat + eps @ chol_post.T
        else:
            gain = bridge_var_t / (bridge_var_t + self.obs_noise_var)
            post_flat = u_hat_flat.clone()
            post_flat[..., self.pos_mask] = u_hat_flat[..., self.pos_mask] + gain * residual
            post_flat = post_flat.reshape(bc, self.n_points)

            obs_var = bridge_var_t * self.obs_noise_var / (bridge_var_t + self.obs_noise_var)
            eps = torch.randn_like(post_flat)
            post_flat = post_flat + math.sqrt(bridge_var_t) * eps
            post_flat = post_flat.reshape(u_hat.shape[0], self.n_channels, self.n_points)
            post_flat[..., self.pos_mask] = post_flat[..., self.pos_mask] + (
                math.sqrt(max(obs_var, 0.0)) - math.sqrt(bridge_var_t)
            ) * eps.reshape(u_hat.shape[0], self.n_channels, self.n_points)[..., self.pos_mask]
            return self._unflatten(post_flat)

        return self._unflatten(post_flat.reshape(u_hat.shape[0], self.n_channels, self.n_points))

    def sample(
        self,
        n_samples: int = 256,
        annealing_steps: int = 32,
        ode_steps: int = 4,
        langevin_steps: int = 16,
        langevin_lr: float = 5e-2,
        ode_method: str = "euler",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        return_path: bool = False,
        record_every: int = 4,
        init: Optional[Tensor] = None,
    ) -> FAPSSamples:
        """
        Draw posterior samples for masked functional regression.

        Parameters
        ----------
        n_samples:
            Number of posterior samples to generate in parallel.
        annealing_steps:
            Number of outer FAPS levels in t-space from 0 -> 1.
        ode_steps:
            Number of solver evaluations used when computing the endpoint anchor.
        langevin_steps:
            Number of inner Langevin steps per outer level.
        langevin_lr:
            Step size for the preconditioned Langevin dynamics.
        init:
            Optional initial path state at t=0. If omitted, a GP sample is used.
        """
        t_schedule = torch.linspace(0.0, 1.0, annealing_steps + 1, device=self.device, dtype=self.dtype)

        if init is None:
            u_t = self.bridge_std(t=0.0) * self._sample_base_gp(n_samples)
        else:
            u_t = init.to(self.device, dtype=self.dtype)
            if u_t.shape != (n_samples, self.n_channels, *self.dims):
                raise ValueError(
                    f"init has shape {tuple(u_t.shape)}, expected {(n_samples, self.n_channels, *self.dims)}"
                )

        trajectory: Optional[List[Tuple[float, Tensor]]] = [] if return_path else None
        u_clean = None

        for k in range(annealing_steps):
            t_cur = float(t_schedule[k].item())
            t_next = float(t_schedule[k + 1].item())

            # Step 1: solve the learned flow from the current path state to t=1.
            u_hat = self.endpoint_anchor(
                u_t,
                t_start=t_cur,
                ode_steps=ode_steps,
                ode_method=ode_method,
                rtol=rtol,
                atol=atol,
            )
            # Step 2: FAPS correction in function space.
            u_clean = u_hat.clone()
            bridge_var_t = self.bridge_var(t_cur)
            lam2 = self.anchor_std(t_cur) ** 2
            if self.exact_linear_update:
                u_clean = self._exact_linear_gaussian_update(u_hat, bridge_var_t)
            else:
                for _ in range(langevin_steps):
                    ll_grad = self._loglik_grad(u_clean)
                    prior_grad = -(u_clean - u_hat) / lam2
                    grad = prior_grad + self.tau * self._cov_matvec(ll_grad)
                    noise = math.sqrt(2.0 * langevin_lr) * self._sample_bridge_base(u_clean.shape[0])
                    u_clean = u_clean + langevin_lr * grad + noise
            # Step 3: draw the next path state from the bridge kernel.
            if return_path and (k % record_every == 0 or k == annealing_steps - 1):
                trajectory.append((t_cur, u_clean.detach().cpu()))

            u_t = self.rebridge(u_clean, t_next=t_next)


        samples = u_clean.clone().detach().cpu()
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return FAPSSamples(samples=samples, mean=mean, std=std, trajectory=trajectory)


def run_masked_regression_FAPS(
    G,
    dims: Sequence[int],
    pos_mask: Tensor,
    u_obs_part: Tensor,
    noise_level: float,
    *,
    gp_ref: Optional[true_GPPrior_Gibbs] = None,
    n_samples: int = 512,
    annealing_steps: int = 32,
    ode_steps: int = 4,
    langevin_steps: int = 16,
    langevin_lr: float = 1e-3,
    tau: float = 1.0,
    exact_linear_update: bool = False,
    anchor_std_base: float = 0.05,
    anchor_std_scale: float = 1.0,
    low_rank_cov_rank: int = 32,
    low_rank_cov_samples: int = 256,
    low_rank_cov_ode_steps: int = 24,
    device: Optional[str] = None,
) -> FAPSSamples:
    """Convenience wrapper matching the regression notebook's variable names."""
    sampler = OFMMaskedFAPSSampler(
        G=G,
        dims=dims,
        pos_mask=pos_mask,
        u_obs_part=u_obs_part,
        noise_level=noise_level,
        gp_ref=gp_ref,
        tau=tau,
        exact_linear_update=exact_linear_update,
        anchor_std_base=anchor_std_base,
        anchor_std_scale=anchor_std_scale,
        low_rank_cov_rank=low_rank_cov_rank,
        low_rank_cov_samples=low_rank_cov_samples,
        low_rank_cov_ode_steps=low_rank_cov_ode_steps,
        device=device,
    )
    return sampler.sample(
        n_samples=n_samples,
        annealing_steps=annealing_steps,
        ode_steps=ode_steps,
        langevin_steps=langevin_steps,
        langevin_lr=langevin_lr,
        return_path=True,
    )

def _mean_acovf(samples_np: np.ndarray, nlag: int = 50) -> np.ndarray:
    """Average autocovariance over a batch of 1D trajectories."""
    nlag_eff = min(nlag, samples_np.shape[-1] - 1)
    acovf_stack = [sm.tsa.acovf(x, nlag=nlag_eff) for x in samples_np]
    return np.mean(np.stack(acovf_stack, axis=0), axis=0)


def ground_truth_gp_posterior_reference(
    pos_mask: Tensor,
    u_obs_part: Tensor,
    n_points: int,
    noise_level: float,
    n_draws: int,
    cfg: EvalConfig,
) -> Dict[str, np.ndarray]:
    """
    Compute the exact GP posterior reference under the data-generating Gibbs GP prior.
    """
    mask_np = pos_mask.detach().cpu().numpy().astype(bool).reshape(-1)
    obs_values = u_obs_part.detach().cpu().numpy().reshape(-1)
    obs_idx = np.where(mask_np)[0]

    x_grid = np.linspace(0.0, 1.0, n_points).reshape(-1, 1)
    x_train = x_grid[obs_idx]
    eps = 1e-8
    K_uu = gibbs_kernel_cov(x_grid, l0=cfg.gibbs_l0, l1=cfg.gibbs_l1, sigma=cfg.gibbs_sigma)
    K_oo = gibbs_kernel_cov(x_train, l0=cfg.gibbs_l0, l1=cfg.gibbs_l1, sigma=cfg.gibbs_sigma)
    K_uo = K_uu[:, obs_idx]
    K_ou = K_uo.T

    C_oo = K_oo + float(noise_level) * np.eye(K_oo.shape[0], dtype=np.float64)
    alpha = np.linalg.solve(C_oo, obs_values.astype(np.float64))
    gp_mean = K_uo @ alpha

    v = np.linalg.solve(C_oo, K_ou)
    gp_cov = K_uu - K_uo @ v
    gp_cov = 0.5 * (gp_cov + gp_cov.T) + eps * np.eye(gp_cov.shape[0], dtype=np.float64)
    gp_std = np.sqrt(np.maximum(np.diag(gp_cov), 0.0))

    rng = np.random.default_rng(22)
    gp_samples = rng.multivariate_normal(gp_mean, gp_cov, size=n_draws).astype(np.float32)
    return {"mean": gp_mean.astype(np.float32), "std": gp_std.astype(np.float32), "samples": gp_samples}


def visualize_masked_regression(
    posterior: FAPSSamples,
    u_obs_full: Tensor,
    u_obs_part: Tensor,
    pos_mask: Tensor,
    save_dir: Path,
    cfg: EvalConfig,
    max_plot_samples: int = 100,
    nlag: int = 50,
) -> None:
    """Create paper-style notebook plots for masked regression results."""
    save_dir.mkdir(parents=True, exist_ok=True)

    true_curve = u_obs_full.detach().cpu().numpy().reshape(-1)
    obs_values = u_obs_part.detach().cpu().numpy().reshape(-1)
    mask_np = pos_mask.detach().cpu().numpy().astype(bool).reshape(-1)
    obs_idx = np.where(mask_np)[0]

    samples_np = posterior.samples[:, 0, :].numpy()
    post_mean = posterior.mean[0].numpy()
    post_std = posterior.std[0].numpy()
    x_axis = np.arange(samples_np.shape[-1])
    gp_ref = ground_truth_gp_posterior_reference(
        pos_mask=pos_mask,
        u_obs_part=u_obs_part,
        n_points=samples_np.shape[-1],
        noise_level=cfg.noise_level,
        n_draws=samples_np.shape[0],
        cfg=cfg,
    )
    gp_mean = gp_ref["mean"]
    gp_std = gp_ref["std"]
    gp_samples = gp_ref["samples"]

    plot_style = {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }

    def _save_figure(fig: plt.Figure, name: str) -> None:
        fig.savefig(save_dir / f"{name}.png", dpi=240)
        #fig.savefig(save_dir / f"{name}.pdf")
        plt.close(fig)

    # Figure 1: Posterior draws, posterior mean +/- 2 std, and observed points.
    with plt.rc_context(plot_style):
        n_draws = min(max_plot_samples, samples_np.shape[0])
        draw_idx = np.random.choice(samples_np.shape[0], size=n_draws, replace=False)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        for idx in draw_idx:
            ax.plot(x_axis, samples_np[idx], color="#4C72B0", alpha=0.07, linewidth=0.9)
        ax.plot(x_axis, post_mean, color="#4C72B0", linewidth=2.6, label="Posterior mean")
        ax.fill_between(
            x_axis,
            post_mean - 2.0 * post_std,
            post_mean + 2.0 * post_std,
            color="#4C72B0",
            alpha=0.18,
            label="Mean +/- 2 std",
        )
        ax.plot(
            x_axis,
            gp_mean,
            color="#55A868",
            linewidth=2.2,
            linestyle="--",
            label="True GP posterior mean",
        )
        ax.fill_between(
            x_axis,
            gp_mean - 2.0 * gp_std,
            gp_mean + 2.0 * gp_std,
            color="#55A868",
            alpha=0.14,
            label="True GP mean +/- 2 std",
        )
        ax.plot(x_axis, true_curve, color="k", linewidth=2.3, label="Ground truth")
        ax.scatter(
            obs_idx,
            obs_values,
            color="#C44E52",
            edgecolor="white",
            linewidth=0.6,
            s=44,
            zorder=5,
            label="Observed points",
        )
        ax.set_title("Masked GP Regression Posterior")
        ax.set_xlabel("Spatial index")
        ax.set_ylabel("Function value")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.22)
        _save_figure(fig, "masked_regression_posterior")

        # Figure 2: compare OFM-FAPS and true GP posterior distributions.
        post_acovf = _mean_acovf(samples_np, nlag=nlag)
        gp_acovf = _mean_acovf(gp_samples, nlag=nlag)

        hist_post, edges_post = np.histogram(samples_np.reshape(-1), bins=50, range=(-4, 4), density=True)
        hist_gp, edges_gp = np.histogram(gp_samples.reshape(-1), bins=50, range=(-4, 4), density=True)
        center_post = 0.5 * (edges_post[1:] + edges_post[:-1])
        center_gp = 0.5 * (edges_gp[1:] + edges_gp[:-1])

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].plot(post_acovf, color="#4C72B0", linewidth=2.4, label="OFM-FAPS posterior")
        ax[0].plot(gp_acovf, color="#55A868", linestyle="--", linewidth=2.4, label="True GP posterior")
        ax[0].set_title("Autocovariance")
        ax[0].set_xlabel("Lag")
        ax[0].legend(loc="upper right", frameon=False)
        ax[0].grid(alpha=0.22)

        ax[1].plot(center_post, hist_post, color="#4C72B0", linewidth=2.4, label="OFM-FAPS posterior")
        ax[1].plot(center_gp, hist_gp, color="#55A868", linestyle="--", linewidth=2.4, label="True GP posterior")
        ax[1].set_title("Value Histogram")
        ax[1].set_xlabel("Value")
        ax[1].legend(loc="upper right", frameon=False)
        ax[1].grid(alpha=0.22)

        _save_figure(fig, "masked_regression_diagnostics")

        # Figure 3: posterior std and mask locations.
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.2))
        ax.plot(x_axis, post_std, color="#DD8452", linewidth=2.6, label="OFM-FAPS std")
        ax.plot(x_axis, gp_std, color="#55A868", linestyle="--", linewidth=2.2, label="True GP posterior std")
        ax.scatter(
            obs_idx,
            post_std[obs_idx],
            color="#C44E52",
            edgecolor="white",
            linewidth=0.6,
            s=44,
            zorder=5,
            label="Observed locations",
        )
        ax.set_title("Posterior Uncertainty vs Observation Mask")
        ax.set_xlabel("Spatial index")
        ax.set_ylabel("Std")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.22)
        _save_figure(fig, "masked_regression_std")


def main() -> None:
    cfg = build_config(parse_args())
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    t0 = time.perf_counter()
    fmot = build_prior_model(cfg)
    u_obs_full, u_obs_part, pos_mask, pos_idx = build_regression_case(cfg)
    print(
        f"Regression case config: query_nx={cfg.n_x}, seed={cfg.seed}, "
        f"test_idx={cfg.test_sample_idx}, n_obs={cfg.n_obs}"
    )
    print(f"Observed idx ({cfg.n_x}-grid): {pos_idx.tolist()}")

    ts = time.perf_counter()
    posterior = run_masked_regression_FAPS(
        G=fmot,
        dims=cfg.dims,
        pos_mask=pos_mask,
        u_obs_part=u_obs_part,
        noise_level=cfg.noise_level,
        n_samples=cfg.n_samples,
        annealing_steps=cfg.annealing_steps,
        ode_steps=cfg.ode_steps,
        langevin_steps=cfg.langevin_steps,
        langevin_lr=cfg.langevin_lr,
        tau=cfg.tau,
        anchor_std_base=cfg.anchor_std_base,
        anchor_std_scale=cfg.anchor_std_scale,
        gp_ref=None,
        exact_linear_update=False,
        low_rank_cov_rank=cfg.low_rank_cov_rank,
        low_rank_cov_samples=cfg.low_rank_cov_samples,
        low_rank_cov_ode_steps=cfg.low_rank_cov_ode_steps,
        device=str(cfg.device),
    )
    sample_elapsed = time.perf_counter() - ts
    print(f"Posterior sampling runtime: {sample_elapsed:.2f} s")

    tv = time.perf_counter()
    visualize_masked_regression(
        posterior=posterior,
        u_obs_full=u_obs_full,
        u_obs_part=u_obs_part,
        pos_mask=pos_mask,
        save_dir=cfg.results_dir,
        cfg=cfg,
    )
    viz_elapsed = time.perf_counter() - tv
    print(f"Visualization runtime: {viz_elapsed:.2f} s")
    print(f"Saved masked-regression visualizations to: {cfg.results_dir}")
    elapsed = time.perf_counter() - t0
    print(f"Total regression runtime: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
