"""Evaluate UNet-OFM inverse posterior metrics for PDE test sets.

This standalone evaluator shares one fixed-resolution UNet-OFM DAPS
implementation across Darcy, Poisson, Helmholtz, and non-bounded
Navier-Stokes.  PDE-specific differences live in a small config table: paths,
default device, observation count, and sampling hyperparameters.

CRPS and SSR compare all posterior samples to the true input field.  PSNR,
SSIM, and Relative L2 compare one randomly selected posterior sample to the
true input field for each test case.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torchdiffeq import odeint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faps_utils.fno_solver import FNOSolver
from faps_utils.ofm_white_velocity_pred import OFMModel
from faps_utils.unet_ofm import UNet_cond

Tensor = torch.Tensor

N_X = 128
DIMS = [N_X, N_X]
VIS_CHANNELS = 1
N_DUMMY_CONDS = 1
UNET_HIDDEN_CHANNELS = 64
UNET_CONDS_CHANNELS = N_DUMMY_CONDS
UNET_RES_BLOCKS = 1
UNET_HEADS = 4
UNET_ATTENTION_RES = "16"
UNET_CHANNEL_MULT = None
EPOCHS = 100
SIGMA_MIN = 1e-4

FORWARD_MODES = 48
FORWARD_HIDDEN_CHANNELS = 64
FORWARD_PROJECTION_CHANNELS = 128
FORWARD_N_LAYERS = 4

DEFAULT_SEED = 22
DEFAULT_NOISE_LEVEL = 1e-3
DEFAULT_POSTERIOR_SAMPLES = 32
DEFAULT_LOW_RANK_COV_RANK = 32
DEFAULT_LOW_RANK_COV_SAMPLES = 256
DEFAULT_LOW_RANK_COV_ODE_STEPS = 20
DEFAULT_ANCHOR_STD_BASE = 0.05
DEFAULT_ANCHOR_STD_SCALE = 1.0
DEFAULT_GRAD_CLIP_NORM = None
DEFAULT_STATE_CLAMP = None
DEFAULT_NUM_TEST = 100
DEFAULT_START_IDX = 0
DEFAULT_DATA_RANGE = 5.0

METRIC_NAMES = ["CRPS", "SSR", "PSNR", "SSIM", "Relative_L2"]
OAK_PRIOR_ROOT = Path("/oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse/prior_training_outputs")
OAK_FORWARD_ROOT = Path("/oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse/forward_op_training_outputs")
OAK_DATA_ROOT = Path("/oak-data/yshi5/yshi5/FAPS/OFM/PDE_inverse_npy")
PDE_INVERSE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PDEConfig:
    label: str
    prior_dir_name: str
    forward_dir_name: str
    test_data_name: str
    default_device: str
    default_n_observations: int
    default_annealing_steps: int
    default_ode_steps: int
    default_langevin_steps: int
    default_langevin_lr: float
    default_tau: float = 1.0

    @property
    def prior_dir(self) -> Path:
        return OAK_PRIOR_ROOT / self.prior_dir_name

    @property
    def prior_fallback_dir(self) -> Path:
        return PDE_INVERSE_ROOT / "checkpoints" / "FAPS_prior" / "UNet"

    @property
    def local_prior_path(self) -> Path:
        label = "ns" if self.label == "ns_nonbounded" else self.label
        return self.prior_fallback_dir / f"{label}_unet_prior_{EPOCHS}.pt"

    @property
    def local_forward_ckpt(self) -> Path:
        label = "ns" if self.label == "ns_nonbounded" else self.label
        return PDE_INVERSE_ROOT / "checkpoints" / "PDE_surrogate" / f"{label}_forward.pt"

    @property
    def forward_ckpt(self) -> Path:
        return OAK_FORWARD_ROOT / self.forward_dir_name / "last.pt"

    @property
    def test_data_path(self) -> Path:
        return OAK_DATA_ROOT / self.test_data_name

    @property
    def default_save_dir(self) -> Path:
        label = "ns" if self.label == "ns_nonbounded" else self.label
        return PDE_INVERSE_ROOT / "outputs" / "PDE_inverse_metrics_unet" / label


PDE_CONFIGS = {
    "darcy": PDEConfig(
        label="darcy",
        prior_dir_name="darcy_ofm_unet_prior",
        forward_dir_name="darcy_fno_full",
        test_data_name="darcy_flow_test.npy",
        default_device="cuda:6" if torch.cuda.is_available() else "cpu",
        default_n_observations=128,
        default_annealing_steps=20,
        default_ode_steps=10,
        default_langevin_steps=40,
        default_langevin_lr=4e-5,
    ),
    "poisson": PDEConfig(
        label="poisson",
        prior_dir_name="poisson_ofm_unet_prior",
        forward_dir_name="poisson_fno_full",
        test_data_name="poisson_flow_test.npy",
        default_device="cuda:7" if torch.cuda.is_available() else "cpu",
        default_n_observations=128,
        default_annealing_steps=20,
        default_ode_steps=10,
        default_langevin_steps=40,
        default_langevin_lr=4e-5,
    ),
    "helmholtz": PDEConfig(
        label="helmholtz",
        prior_dir_name="helmholtz_ofm_unet_prior",
        forward_dir_name="helmholtz_fno_full",
        test_data_name="helmholtz_flow_test.npy",
        default_device="cuda:5" if torch.cuda.is_available() else "cpu",
        default_n_observations=128,
        default_annealing_steps=20,
        default_ode_steps= 10,
        default_langevin_steps=40,
        default_langevin_lr=4e-5,
    ),
    "ns_nonbounded": PDEConfig(
        label="ns_nonbounded",
        prior_dir_name="ns_nonbounded_ofm_unet_prior",
        forward_dir_name="ns_nonbounded_fno_full",
        test_data_name="ns_nonbounded_flow_test.npy",
        default_device="cuda:5" if torch.cuda.is_available() else "cpu",
        default_n_observations=128,
        default_annealing_steps=20,
        default_ode_steps=10,
        default_langevin_steps=40,
        default_langevin_lr=4e-5,
    ),
}


@dataclass
class DAPSSamples:
    samples: Tensor
    mean: Tensor
    std: Tensor


def _load_metrics_module():
    metrics_path = PROJECT_ROOT / "faps_utils" / "2D_metrics.py"
    spec = importlib.util.spec_from_file_location("metrics_2d", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metrics module from {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics_2d = _load_metrics_module()


class VelocityUNetAdapter(torch.nn.Module):
    """Keep using UNet_cond while dropping dummy conditioning at the backbone."""

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
            raise ValueError(f"Expected time tensor [B], got shape {tuple(t.shape)} for batch {x.shape[0]}")
        x = self.input_proj_2d(x)
        out = self.model.unet_backbone(t, x, None)
        return self.output_proj_2d(out)


class FixedResolutionWhiteBaseDist:
    """Distribution adapter expected by the DAPS sampler."""

    def __init__(self, dims: list[int], device: str, dtype: torch.dtype = torch.float32) -> None:
        self.dims = list(dims)
        self.device = device
        self.dtype = dtype

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return torch.randn(*sample_shape, *self.dims, device=self.device, dtype=self.dtype)


class OFMSurrogateDAPSSampler:
    """DAPS sampler with nonlinear likelihood through a pretrained FNO surrogate."""

    def __init__(
        self,
        G: OFMModel,
        forward_model: torch.nn.Module,
        dims: Sequence[int],
        pos_mask: Tensor,
        y_obs_part: Tensor,
        noise_level: float,
        *,
        tau: float = 1.0,
        anchor_std_base: float = DEFAULT_ANCHOR_STD_BASE,
        anchor_std_scale: float = DEFAULT_ANCHOR_STD_SCALE,
        low_rank_cov_rank: int = DEFAULT_LOW_RANK_COV_RANK,
        low_rank_cov_samples: int = DEFAULT_LOW_RANK_COV_SAMPLES,
        low_rank_cov_ode_steps: int = DEFAULT_LOW_RANK_COV_ODE_STEPS,
        low_rank_cov_jitter: float = 1e-5,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.G = G
        self.forward_model = forward_model.eval()
        for p in self.forward_model.parameters():
            p.requires_grad = False

        self.dims = list(dims)
        self.n_points = int(np.prod(self.dims))
        self.device = device or getattr(G, "device", "cpu")
        self.dtype = dtype

        self.pos_mask = self._normalize_mask(pos_mask).to(self.device)
        self.y_obs_part = self._normalize_obs(y_obs_part).to(self.device, dtype=self.dtype)
        self.obs_noise_var = float(noise_level)
        self.tau = float(tau)
        self.anchor_std_base = float(anchor_std_base)
        self.anchor_std_scale = float(anchor_std_scale)
        self.low_rank_cov_rank = int(max(0, low_rank_cov_rank))
        self.low_rank_cov_samples = int(max(0, low_rank_cov_samples))
        self.low_rank_cov_ode_steps = int(max(1, low_rank_cov_ode_steps))
        self.low_rank_cov_jitter = float(max(0.0, low_rank_cov_jitter))

        if list(getattr(self.G.gp, "dims", [])) == self.dims:
            self.gp_dist = self.G.gp.base_dist
        else:
            raise ValueError(
                "UNet OFM prior is fixed-resolution; requested dims "
                f"{self.dims}, but prior was trained for {list(getattr(self.G.gp, 'dims', []))}."
            )

        self.n_channels = int(self.y_obs_part.shape[1])
        self.gp_cov = self._estimate_low_rank_covariance() if self.low_rank_cov_rank > 0 else None
        if self.gp_cov is not None:
            eye = torch.eye(self.n_points, device=self.device, dtype=self.dtype)
            self.gp_cov = 0.5 * (self.gp_cov + self.gp_cov.T) + self.low_rank_cov_jitter * eye
            self.cov_scale_tril = torch.linalg.cholesky(self.gp_cov)
            print(
                f"Using covariance source: low-rank surrogate "
                f"(rank={self.low_rank_cov_rank}, samples={self.low_rank_cov_samples})"
            )
        else:
            self.cov_scale_tril = None
            print("Using covariance source: identity (no low-rank surrogate)")

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
            del u0, u1, u_flat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        u_centered = u_flat - u_flat.mean(dim=0, keepdim=True)
        _, s, vh = torch.linalg.svd(u_centered, full_matrices=False)
        eigvals = (s * s) / max(u_centered.shape[0] - 1, 1)
        rank = min(self.low_rank_cov_rank, eigvals.shape[0], vh.shape[0])
        if rank <= 0:
            del u0, u1, u_flat, u_centered, s, vh, eigvals
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        basis = vh[:rank].T
        top_eigs = eigvals[:rank]
        cov_lr = (basis * top_eigs.unsqueeze(0)) @ basis.T
        if rank < eigvals.shape[0]:
            remaining_dims = max(self.n_points - rank, 1)
            res_var = torch.sum(eigvals[rank:]) / remaining_dims
            cov_lr = cov_lr + res_var * torch.eye(self.n_points, device=self.device, dtype=self.dtype)
        del u0, u1, u_flat, u_centered, s, vh, eigvals, basis, top_eigs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cov_lr

    def _normalize_mask(self, mask: Tensor) -> Tensor:
        mask = torch.as_tensor(mask)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.numel() != self.n_points:
            raise ValueError(f"pos_mask has {mask.numel()} entries, expected {self.n_points}.")
        return mask.reshape(-1)

    def _normalize_obs(self, y_obs_part: Tensor) -> Tensor:
        y_obs_part = torch.as_tensor(y_obs_part)
        if y_obs_part.ndim == 2:
            y_obs_part = y_obs_part.unsqueeze(0)
        if y_obs_part.ndim != 3:
            raise ValueError(
                f"y_obs_part must have shape [1, C, n_obs] or [C, n_obs], got {tuple(y_obs_part.shape)}."
            )
        if y_obs_part.shape[-1] != int(self.pos_mask.sum()):
            raise ValueError(
                f"y_obs_part last dimension is {y_obs_part.shape[-1]}, expected {int(self.pos_mask.sum())}."
            )
        return y_obs_part

    def _flatten(self, u: Tensor) -> Tensor:
        return u.reshape(u.shape[0], u.shape[1], -1)

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

    def _apply_mask(self, y: Tensor) -> Tensor:
        return self._flatten(y)[..., self.pos_mask]

    def _loglik_grad(self, u: Tensor) -> Tensor:
        u_req = u.detach().clone().requires_grad_(True)
        y_pred = self.forward_model(u_req)
        residual = self._apply_mask(y_pred) - self.y_obs_part.expand(u.shape[0], -1, -1)
        neg_loglik = 0.5 * torch.sum(residual * residual) / self.obs_noise_var
        grad_neg_loglik = torch.autograd.grad(neg_loglik, u_req)[0]
        return -grad_neg_loglik.detach()

    def _cov_matvec(self, x: Tensor) -> Tensor:
        if self.cov_scale_tril is None:
            return x
        x_flat = self._flatten(x)
        b, c, n = x_flat.shape
        x_bc = x_flat.reshape(b * c, n)
        out = (x_bc @ self.cov_scale_tril) @ self.cov_scale_tril.T
        return out.reshape(b, c, *self.dims)

    def bridge_std(self, t: float) -> float:
        sigma_min = float(getattr(self.G, "sigma_min", 0.0))
        return math.sqrt((1.0 - float(t)) ** 2 + sigma_min**2)

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

    def sample(
        self,
        n_samples: int,
        annealing_steps: int,
        ode_steps: int,
        langevin_steps: int,
        langevin_lr: float,
        grad_clip_norm: Optional[float],
        state_clamp: Optional[float],
        ode_method: str = "euler",
    ) -> DAPSSamples:
        t_schedule = torch.linspace(0.0, 1.0, annealing_steps + 1, device=self.device, dtype=self.dtype)
        u_t = self.bridge_std(t=0.0) * self._sample_base_gp(n_samples)
        u_clean = None

        for k in range(annealing_steps):
            t_cur = float(t_schedule[k].item())
            t_next = float(t_schedule[k + 1].item())
            u_hat = self.endpoint_anchor(u_t, t_start=t_cur, ode_steps=ode_steps, ode_method=ode_method)
            u_clean = u_hat.clone()
            lam2 = self.anchor_std(t_cur) ** 2

            for _ in range(langevin_steps):
                ll_grad = self._loglik_grad(u_clean)
                prior_grad = -(u_clean - u_hat) / lam2
                grad = prior_grad + self.tau * self._cov_matvec(ll_grad)
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    grad_flat = grad.reshape(grad.shape[0], -1)
                    grad_norm = grad_flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    scale = (float(grad_clip_norm) / grad_norm).clamp(max=1.0)
                    grad = (grad_flat * scale).reshape_as(grad)
                noise = math.sqrt(2.0 * langevin_lr) * self._sample_bridge_base(u_clean.shape[0])
                u_clean = u_clean + langevin_lr * grad + noise
                u_clean = torch.nan_to_num(u_clean, nan=0.0, posinf=1e6, neginf=-1e6)
                if state_clamp is not None and state_clamp > 0:
                    u_clean = u_clean.clamp(min=-float(state_clamp), max=float(state_clamp))

            u_t = self.rebridge(u_clean, t_next=t_next)

        assert u_clean is not None
        samples = u_clean.clone().detach().cpu()
        return DAPSSamples(samples=samples, mean=samples.mean(dim=0), std=samples.std(dim=0))


def parse_args() -> tuple[argparse.Namespace, PDEConfig]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--pde", choices=sorted(PDE_CONFIGS), default="darcy")
    known, remaining = pre.parse_known_args()
    cfg = PDE_CONFIGS[known.pde]

    parser = argparse.ArgumentParser(
        description="Evaluate UNet-OFM inverse posterior metrics over a PDE test set.",
        parents=[pre],
    )
    parser.add_argument("--prior-path", type=Path, default=cfg.local_prior_path)
    parser.add_argument("--prior-dir", type=Path, default=None)
    parser.add_argument("--test-data-path", type=Path, default=cfg.test_data_path)
    parser.add_argument("--forward-ckpt", type=Path, default=cfg.local_forward_ckpt)
    parser.add_argument("--start-idx", type=int, default=DEFAULT_START_IDX)
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST)
    parser.add_argument("--n-observations", type=int, default=cfg.default_n_observations)
    parser.add_argument("--noise-level", type=float, default=DEFAULT_NOISE_LEVEL)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_POSTERIOR_SAMPLES)
    parser.add_argument("--annealing-steps", type=int, default=cfg.default_annealing_steps)
    parser.add_argument("--ode-steps", type=int, default=cfg.default_ode_steps)
    parser.add_argument("--langevin-steps", type=int, default=cfg.default_langevin_steps)
    parser.add_argument("--langevin-lr", type=float, default=cfg.default_langevin_lr)
    parser.add_argument("--tau", type=float, default=cfg.default_tau)
    parser.add_argument("--low-rank-cov-rank", type=int, default=DEFAULT_LOW_RANK_COV_RANK)
    parser.add_argument("--low-rank-cov-samples", type=int, default=DEFAULT_LOW_RANK_COV_SAMPLES)
    parser.add_argument("--low-rank-cov-ode-steps", type=int, default=DEFAULT_LOW_RANK_COV_ODE_STEPS)
    parser.add_argument("--grad-clip-norm", type=float, default=DEFAULT_GRAD_CLIP_NORM)
    parser.add_argument("--state-clamp", type=float, default=DEFAULT_STATE_CLAMP)
    parser.add_argument("--data-range", type=float, default=DEFAULT_DATA_RANGE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=cfg.default_device)
    parser.add_argument("--save-dir", type=Path, default=cfg.default_save_dir)
    parser.add_argument("--save-prefix", type=str, default=f"{cfg.label}_test_metrics")
    parser.add_argument(
        "--reuse-low-rank-cov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate the low-rank bridge covariance once and reuse it for all test cases.",
    )
    args = parser.parse_args(remaining, namespace=known)
    return args, cfg


def load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if hasattr(checkpoint, "state_dict"):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    if isinstance(checkpoint, dict):
        checkpoint = {k: v for k, v in checkpoint.items() if k != "_metadata"}
    return checkpoint


def resolve_prior_path(cfg: PDEConfig, prior_path: Path | None, prior_dir: Path | None) -> Path:
    candidates = []
    if prior_path is not None:
        candidates.append(prior_path.expanduser())
    if prior_dir is not None:
        candidates.append(prior_dir.expanduser() / f"epoch_{EPOCHS}.pt")
    candidates.extend([cfg.local_prior_path, cfg.prior_dir / f"epoch_{EPOCHS}.pt"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find UNet OFM prior checkpoint in: {searched}")


def build_unet_model(device: str) -> torch.nn.Module:
    base_model = UNet_cond(
        dims=[VIS_CHANNELS, *DIMS],
        hidden_channels=UNET_HIDDEN_CHANNELS,
        conds_channels=UNET_CONDS_CHANNELS,
        num_res_blocks=UNET_RES_BLOCKS,
        num_heads=UNET_HEADS,
        attention_res=UNET_ATTENTION_RES,
        channel_mult=UNET_CHANNEL_MULT,
        in_channels=VIS_CHANNELS,
    )
    return VelocityUNetAdapter(base_model).to(device)


def build_prior_model(
    cfg: PDEConfig,
    device: str,
    prior_path: Path | None,
    prior_dir: Path | None,
) -> OFMModel:
    checkpoint_path = resolve_prior_path(cfg, prior_path=prior_path, prior_dir=prior_dir)
    print(f"Loading UNet OFM prior checkpoint from: {checkpoint_path}")
    model = build_unet_model(device)
    for param in model.parameters():
        param.requires_grad = False
    model.load_state_dict(load_checkpoint(checkpoint_path))
    model.eval()

    prior = OFMModel(
        model,
        sigma_min=SIGMA_MIN,
        device=device,
        dims=DIMS,
    )
    prior.gp.base_dist = FixedResolutionWhiteBaseDist(DIMS, device=device)
    return prior


def build_forward_model(device: str, ckpt_path: Path) -> torch.nn.Module:
    ckpt_path = ckpt_path.expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Forward surrogate checkpoint not found: {ckpt_path}")
    print(f"Loading forward surrogate checkpoint from: {ckpt_path}")
    model = FNOSolver(
        in_channels=1,
        out_channels=1,
        n_modes=(FORWARD_MODES, FORWARD_MODES),
        hidden_channels=FORWARD_HIDDEN_CHANNELS,
        projection_channels=FORWARD_PROJECTION_CHANNELS,
        n_layers=FORWARD_N_LAYERS,
    ).to(device)
    model.load_state_dict(load_checkpoint(ckpt_path))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_test_pair(test_data: torch.Tensor, idx: int) -> tuple[Tensor, Tensor]:
    if idx < 0 or idx >= test_data.shape[0]:
        raise IndexError(f"test index {idx} out of range [0, {test_data.shape[0] - 1}]")
    sample = test_data[idx : idx + 1]
    return sample[:, 0:1].contiguous(), sample[:, 1:2].contiguous()


def create_random_observation(
    y_true_full: Tensor,
    n_observations: int,
    noise_level: float,
    *,
    device: str,
) -> tuple[Tensor, Tensor]:
    n_points = int(np.prod(y_true_full.shape[-2:]))
    if n_observations <= 0 or n_observations > n_points:
        raise ValueError(f"n_observations must be in [1, {n_points}], got {n_observations}.")
    pos_mask = torch.zeros(n_points, dtype=torch.bool)
    pos_mask[torch.randperm(n_points)[:n_observations]] = True
    y_obs_part = y_true_full.reshape(1, 1, -1)[..., pos_mask].to(device)
    y_obs_part = y_obs_part + torch.randn_like(y_obs_part) * math.sqrt(float(noise_level))
    return pos_mask, y_obs_part


def make_sampler(
    prior: OFMModel,
    forward_model: torch.nn.Module,
    pos_mask: Tensor,
    y_obs_part: Tensor,
    args: argparse.Namespace,
    *,
    low_rank_cov_rank: int,
) -> OFMSurrogateDAPSSampler:
    return OFMSurrogateDAPSSampler(
        G=prior,
        forward_model=forward_model,
        dims=DIMS,
        pos_mask=pos_mask,
        y_obs_part=y_obs_part,
        noise_level=args.noise_level,
        tau=args.tau,
        anchor_std_base=DEFAULT_ANCHOR_STD_BASE,
        anchor_std_scale=DEFAULT_ANCHOR_STD_SCALE,
        low_rank_cov_rank=low_rank_cov_rank,
        low_rank_cov_samples=args.low_rank_cov_samples,
        low_rank_cov_ode_steps=args.low_rank_cov_ode_steps,
        device=args.device,
    )


def sample_case(
    prior: OFMModel,
    forward_model: torch.nn.Module,
    y_true: Tensor,
    args: argparse.Namespace,
    *,
    cached_cov_scale_tril: Tensor | None,
) -> tuple[DAPSSamples, Tensor | None]:
    pos_mask, y_obs_part = create_random_observation(
        y_true_full=y_true,
        n_observations=args.n_observations,
        noise_level=args.noise_level,
        device=args.device,
    )

    if cached_cov_scale_tril is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            sampler = make_sampler(
                prior=prior,
                forward_model=forward_model,
                pos_mask=pos_mask,
                y_obs_part=y_obs_part,
                args=args,
                low_rank_cov_rank=0,
            )
        sampler.cov_scale_tril = cached_cov_scale_tril
        sampler.gp_cov = None
        print("Using covariance source: cached low-rank surrogate")
    else:
        sampler = make_sampler(
            prior=prior,
            forward_model=forward_model,
            pos_mask=pos_mask,
            y_obs_part=y_obs_part,
            args=args,
            low_rank_cov_rank=args.low_rank_cov_rank,
        )
        if args.reuse_low_rank_cov and sampler.cov_scale_tril is not None:
            cached_cov_scale_tril = sampler.cov_scale_tril.detach()

    posterior = sampler.sample(
        n_samples=args.n_samples,
        annealing_steps=args.annealing_steps,
        ode_steps=args.ode_steps,
        langevin_steps=args.langevin_steps,
        langevin_lr=args.langevin_lr,
        grad_clip_norm=args.grad_clip_norm,
        state_clamp=args.state_clamp,
        ode_method="euler",
    )
    return posterior, cached_cov_scale_tril


def evaluate_case(
    posterior: DAPSSamples,
    x_true: Tensor,
    rng: np.random.Generator,
    data_range: float,
) -> dict[str, float | int]:
    posterior_samples = posterior.samples[:, 0].float()
    target = x_true[0].cpu().float()
    one_idx = int(rng.integers(0, posterior_samples.shape[0]))
    one_sample = posterior_samples[one_idx : one_idx + 1]
    values = metrics_2d.compute_all_metrics(
        posterior_samples=posterior_samples,
        one_sample=one_sample,
        target=target,
        data_range=data_range,
    )
    row: dict[str, float | int] = {name: float(values[name].detach().cpu().item()) for name in METRIC_NAMES}
    row["posterior_draw_idx"] = one_idx
    return row


def save_results(rows: list[dict[str, float | int]], save_dir: Path, save_prefix: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{save_prefix}_per_case.csv"
    fieldnames = ["sample_idx", "posterior_draw_idx", *METRIC_NAMES]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_array = np.array([[float(row[name]) for name in METRIC_NAMES] for row in rows], dtype=np.float64)
    mean = metric_array.mean(axis=0)
    std = metric_array.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros_like(mean)
    sem = std / math.sqrt(max(len(rows), 1))
    np.savez_compressed(
        save_dir / f"{save_prefix}_summary.npz",
        metric_names=np.array(METRIC_NAMES),
        per_case=metric_array,
        mean=mean,
        std=std,
        sem=sem,
        sample_idx=np.array([int(row["sample_idx"]) for row in rows], dtype=np.int64),
        posterior_draw_idx=np.array([int(row["posterior_draw_idx"]) for row in rows], dtype=np.int64),
    )

    print("\nAveraged metrics over test cases:")
    for name, m, s, e in zip(METRIC_NAMES, mean, std, sem):
        print(f"  {name:12s}: mean={m:.6e}, std={s:.6e}, sem={e:.6e}")
    print(f"Saved per-case metrics to: {csv_path}")
    print(f"Saved summary arrays to: {save_dir / f'{save_prefix}_summary.npz'}")


def main() -> None:
    args, cfg = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    test_data_path = args.test_data_path.expanduser()
    test_data = torch.from_numpy(np.load(test_data_path)).float()
    if test_data.ndim != 4 or test_data.shape[1] < 2:
        raise ValueError(f"Expected test data shape [N, 2, H, W], got {tuple(test_data.shape)}")
    if args.num_test <= 0:
        raise ValueError(f"--num-test must be positive, got {args.num_test}")
    end_idx = args.start_idx + args.num_test
    if args.start_idx < 0 or end_idx > test_data.shape[0]:
        raise IndexError(
            f"Requested test range [{args.start_idx}, {end_idx}) but dataset has {test_data.shape[0]} samples."
        )

    print(
        f"{cfg.label} UNet-OFM test metrics config: "
        f"indices=[{args.start_idx}, {end_idx}), n_obs={args.n_observations}, "
        f"noise_var={args.noise_level}, n_samples={args.n_samples}, "
        f"anneal={args.annealing_steps}, ode_steps={args.ode_steps}, "
        f"langevin_steps={args.langevin_steps}, langevin_lr={args.langevin_lr}, "
        f"tau={args.tau}, low_rank_cov_rank={args.low_rank_cov_rank}, "
        f"data_range={args.data_range}, device={args.device}"
    )

    t0 = time.perf_counter()
    prior = build_prior_model(
        cfg=cfg,
        device=args.device,
        prior_path=args.prior_path,
        prior_dir=args.prior_dir,
    )
    forward_model = build_forward_model(device=args.device, ckpt_path=args.forward_ckpt)

    rows: list[dict[str, float | int]] = []
    cached_cov_scale_tril: Tensor | None = None
    for case_num, idx in enumerate(range(args.start_idx, end_idx), start=1):
        case_t0 = time.perf_counter()
        torch.manual_seed(args.seed + idx)
        x_true, y_true = load_test_pair(test_data, idx)
        posterior, cached_cov_scale_tril = sample_case(
            prior=prior,
            forward_model=forward_model,
            y_true=y_true,
            args=args,
            cached_cov_scale_tril=cached_cov_scale_tril if args.reuse_low_rank_cov else None,
        )
        row = evaluate_case(posterior=posterior, x_true=x_true, rng=rng, data_range=args.data_range)
        row["sample_idx"] = idx
        rows.append(row)
        metric_text = ", ".join(f"{name}={float(row[name]):.4e}" for name in METRIC_NAMES)
        print(
            f"[{case_num:03d}/{args.num_test:03d}] idx={idx}, draw={int(row['posterior_draw_idx'])}, "
            f"{metric_text}, runtime={time.perf_counter() - case_t0:.2f}s"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_results(rows, args.save_dir.expanduser(), args.save_prefix)
    print(f"Total runtime: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
