"""PDE inverse regression with an OFM prior and pretrained FNO surrogate."""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faps_utils.fno import FNO
from faps_utils.fno_solver import FNOSolver
from faps_utils.ofm_ind_likelihood import OFMModel

Tensor = torch.Tensor

PDE_INVERSE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvalConfig:
    pde: str
    prior_path: Path
    forward_ckpt: Path
    test_data_path: Path
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
    n_x: int
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
        return [self.n_x, self.n_x]

    @property
    def label(self) -> str:
        return self.pde.strip()


@dataclass
class FAPSSamples:
    samples: Tensor
    mean: Tensor
    std: Tensor
    trajectory: Optional[List[Tuple[float, Tensor]]] = None


class OFMSurrogateFAPSSampler:
    """FAPS sampler with nonlinear likelihood through pretrained surrogate."""

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
        anchor_std_base: float = 0.05,
        anchor_std_scale: float = 1.0,
        low_rank_cov_rank: int = 32,
        low_rank_cov_samples: int = 256,
        low_rank_cov_ode_steps: int = 20,
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
            self.gp_dist = self.G.gp.new_dist(self.dims)

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
        has_res_var = False
        if rank < eigvals.shape[0]:
            # The discarded eigenvalues are total variance over feature-space
            # directions. Spread that trace over the unmodeled spatial degrees
            # of freedom; using their mean directly over-inflates the diagonal
            # by roughly n_points / n_samples at full resolution.
            remaining_dims = max(self.n_points - rank, 1)
            res_var = torch.sum(eigvals[rank:]) / remaining_dims
            cov_lr = cov_lr + res_var * torch.eye(self.n_points, device=self.device, dtype=self.dtype)
            has_res_var = True
        # Release temporary GPU tensors aggressively; keep only cov_lr alive.
        del u0, u1, u_flat, u_centered, s, vh, eigvals, basis, top_eigs
        if has_res_var:
            del res_var
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
            n_samples * self.n_channels, self.n_points, device=self.device, dtype=self.dtype
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
        return float(t_next) * u_clean + self.bridge_std(t_next) * self._sample_base_gp(
            u_clean.shape[0]
        )

    def sample(
        self,
        n_samples: int = 256,
        annealing_steps: int = 32,
        ode_steps: int = 4,
        langevin_steps: int = 16,
        langevin_lr: float = 5e-2,
        grad_clip_norm: Optional[float] = None,
        state_clamp: Optional[float] = None,
        ode_method: str = "euler",
        return_path: bool = False,
        record_every: int = 4,
    ) -> FAPSSamples:
        t_schedule = torch.linspace(0.0, 1.0, annealing_steps + 1, device=self.device, dtype=self.dtype)
        u_t = self.bridge_std(t=0.0) * self._sample_base_gp(n_samples)
        trajectory: Optional[List[Tuple[float, Tensor]]] = [] if return_path else None
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

            if return_path and (k % record_every == 0 or k == annealing_steps - 1):
                trajectory.append((t_cur, u_clean.detach().cpu()))
            u_t = self.rebridge(u_clean, t_next=t_next)

        assert u_clean is not None
        samples = u_clean.clone().detach().cpu()
        return FAPSSamples(
            samples=samples,
            mean=samples.mean(dim=0),
            std=samples.std(dim=0),
            trajectory=trajectory,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDE inverse regression with an OFM prior and pretrained FNO surrogate."
    )
    parser.add_argument("--pde", type=str, required=True)
    parser.add_argument("--prior-path", type=Path, required=True)
    parser.add_argument("--forward-ckpt", type=Path, required=True)
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=PDE_INVERSE_ROOT / "outputs" / "PDE_inverse")
    parser.add_argument("--save-prefix", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--test-sample-idx", type=int, default=100)
    parser.add_argument("--n-observations", type=int, default=128)
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
    parser.add_argument("--n-x", type=int, default=128)
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


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        pde=args.pde,
        prior_path=args.prior_path.expanduser(),
        forward_ckpt=args.forward_ckpt.expanduser(),
        test_data_path=args.test_data_path.expanduser(),
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
        n_x=args.n_x,
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


def build_prior_model(cfg: EvalConfig) -> OFMModel:
    if not cfg.prior_path.exists():
        raise FileNotFoundError(f"OFM prior checkpoint not found: {cfg.prior_path}")
    print(f"Loading OFM prior checkpoint from: {cfg.prior_path}")
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


def build_forward_model(cfg: EvalConfig) -> torch.nn.Module:
    if not cfg.forward_ckpt.exists():
        raise FileNotFoundError(f"Forward surrogate checkpoint not found: {cfg.forward_ckpt}")
    print(f"Loading forward surrogate checkpoint from: {cfg.forward_ckpt}")
    model = FNOSolver(
        in_channels=1,
        out_channels=1,
        n_modes=(cfg.forward_modes, cfg.forward_modes),
        hidden_channels=cfg.forward_hidden_channels,
        projection_channels=cfg.forward_projection_channels,
        n_layers=cfg.forward_n_layers,
    ).to(cfg.device)

    try:
        checkpoint = torch.load(cfg.forward_ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(cfg.forward_ckpt, map_location="cpu")
    except pickle.UnpicklingError:
        checkpoint = torch.load(cfg.forward_ckpt, map_location="cpu", weights_only=False)

    if hasattr(checkpoint, "state_dict"):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        checkpoint = {k: v for k, v in checkpoint.items() if k != "_metadata"}

    model.load_state_dict(checkpoint)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_test_case(test_data_path: str, test_sample_idx: int) -> tuple[Tensor, Tensor]:
    array = np.load(test_data_path)
    x_test = torch.from_numpy(array).float()
    if x_test.ndim != 4 or x_test.shape[1] < 2:
        raise ValueError(f"Expected test data shape [N, 2, H, W], got {tuple(x_test.shape)}")
    if test_sample_idx < 0 or test_sample_idx >= x_test.shape[0]:
        raise IndexError(f"test_sample_idx={test_sample_idx} out of range [0, {x_test.shape[0] - 1}]")
    sample = x_test[test_sample_idx : test_sample_idx + 1]
    return sample[:, 0:1], sample[:, 1:2]


def create_random_observation(
    y_true_full: Tensor, n_observations: int, noise_level: float, *, device: str
) -> tuple[Tensor, Tensor]:
    n_points = int(np.prod(y_true_full.shape[-2:]))
    if n_observations <= 0 or n_observations > n_points:
        raise ValueError(f"n_observations must be in [1, {n_points}], got {n_observations}.")
    pos_mask = torch.zeros(n_points, dtype=torch.bool)
    pos_mask[torch.randperm(n_points)[:n_observations]] = True
    y_obs_part = y_true_full.reshape(1, 1, -1)[..., pos_mask].to(device)
    y_obs_part = y_obs_part + torch.randn_like(y_obs_part) * math.sqrt(float(noise_level))
    return pos_mask, y_obs_part


def run_surrogate_regression_FAPS(
    G: OFMModel,
    forward_model: torch.nn.Module,
    dims: Sequence[int],
    pos_mask: Tensor,
    y_obs_part: Tensor,
    noise_level: float,
    *,
    n_samples: int,
    annealing_steps: int,
    ode_steps: int,
    langevin_steps: int,
    langevin_lr: float,
    tau: float,
    low_rank_cov_rank: int,
    low_rank_cov_samples: int,
    low_rank_cov_ode_steps: int,
    grad_clip_norm: Optional[float],
    state_clamp: Optional[float],
    device: str,
) -> FAPSSamples:
    sampler = OFMSurrogateFAPSSampler(
        G=G,
        forward_model=forward_model,
        dims=dims,
        pos_mask=pos_mask,
        y_obs_part=y_obs_part,
        noise_level=noise_level,
        tau=tau,
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
        grad_clip_norm=grad_clip_norm,
        state_clamp=state_clamp,
        ode_method="euler",
        return_path=True,
    )


def visualize_results(
    posterior: FAPSSamples,
    x_true_full: Tensor,
    y_true_full: Tensor,
    y_obs_part: Tensor,
    pos_mask: Tensor,
    forward_model: torch.nn.Module,
    cfg: EvalConfig,
) -> None:
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    x_gt = x_true_full.squeeze().cpu().numpy()
    y_gt = y_true_full.squeeze().cpu().numpy()
    x_post_mean = posterior.mean[0].cpu().numpy()
    x_post_std = posterior.std[0].cpu().numpy()
    x_samples = posterior.samples[:, 0].cpu().numpy()
    n_plot_samples = min(6, x_samples.shape[0])

    # Posterior predictive mean in solution space: E[F(u)] over posterior samples.
    with torch.no_grad():
        x_samples_t = posterior.samples.to(device=cfg.device, dtype=torch.float32)
        pred_chunks = []
        pred_batch_size = 64
        for i in range(0, x_samples_t.shape[0], pred_batch_size):
            pred_chunks.append(forward_model(x_samples_t[i : i + pred_batch_size]))
        y_post_mean = torch.cat(pred_chunks, dim=0).mean(dim=0).squeeze().cpu().numpy()

    flat_x_gt = x_gt.reshape(-1)
    flat_x_mean = x_post_mean.reshape(-1)
    flat_y_gt = y_gt.reshape(-1)
    flat_y_mean = y_post_mean.reshape(-1)
    obs_mask_np = pos_mask.cpu().numpy().astype(bool)
    obs_vals = y_obs_part.detach().cpu().numpy().reshape(-1)

    input_mse = float(np.mean((flat_x_mean - flat_x_gt) ** 2))
    observed_solution_mse = float(np.mean((flat_y_mean[obs_mask_np] - flat_y_gt[obs_mask_np]) ** 2))
    full_solution_mse = float(np.mean((flat_y_mean - flat_y_gt) ** 2))
    gt_norm = float(np.linalg.norm(flat_x_gt))
    rel_l2_samples = np.linalg.norm(x_samples.reshape(x_samples.shape[0], -1) - flat_x_gt[None, :], axis=1) / max(
        gt_norm, 1e-12
    )
    rel_l2_samples_mean = float(np.mean(rel_l2_samples))
    rel_l2_samples_min = float(np.min(rel_l2_samples))
    rel_l2_samples_max = float(np.max(rel_l2_samples))

    mask_2d = obs_mask_np.reshape(*cfg.dims)
    obs_y, obs_x = np.where(mask_2d)
    x_err_map = np.abs(x_post_mean - x_gt)

    fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.5))
    im0 = ax[0, 0].imshow(x_gt, cmap="viridis")
    ax[0, 0].set_title("GT input field")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(x_post_mean, cmap="viridis")
    ax[0, 1].set_title("Posterior mean (input)")
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[0, 2].imshow(x_post_std, cmap="magma")
    ax[0, 2].set_title("Posterior std (input)")
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

    im3 = ax[1, 0].imshow(y_gt, cmap="viridis")
    ax[1, 0].set_title("GT solution field")
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im4 = ax[1, 1].imshow(y_post_mean, cmap="viridis")
    ax[1, 1].set_title("Posterior predictive mean solution")
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    plt.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

    ax[1, 2].imshow(y_gt, cmap="gray", alpha=0.35)
    ax[1, 2].scatter(obs_x, obs_y, c=obs_vals, cmap="viridis", s=26, edgecolors="k", linewidths=0.3)
    ax[1, 2].set_title(f"Observed solution points (n={obs_vals.size})")
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])

    fig.suptitle(
        f"{cfg.label} inverse regression | sample_idx={cfg.test_sample_idx} | "
        f"noise_var={cfg.noise_level:.3e}",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(cfg.save_dir / f"{cfg.save_prefix}_reconstruction.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.8))
    hist_x_gt, edges_x_gt = np.histogram(flat_x_gt, bins=60, range=(-4, 4), density=True)
    hist_x_post, edges_x_post = np.histogram(x_samples.reshape(-1), bins=60, range=(-4, 4), density=True)
    centers_x_gt = 0.5 * (edges_x_gt[1:] + edges_x_gt[:-1])
    centers_x_post = 0.5 * (edges_x_post[1:] + edges_x_post[:-1])
    ax[0].plot(centers_x_gt, hist_x_gt, c="k", lw=2.0, label="GT input")
    ax[0].plot(centers_x_post, hist_x_post, c="r", lw=2.0, ls="--", label="Posterior input")
    ax[0].set_title("Input value histogram")
    ax[0].set_xlabel("Value")
    ax[0].legend(loc="upper right")

    y_pred_obs = flat_y_mean[obs_mask_np]
    ax[1].scatter(obs_vals, y_pred_obs, s=22, alpha=0.8, c="#4C72B0", edgecolors="white", linewidths=0.4)
    lo = min(obs_vals.min(), y_pred_obs.min())
    hi = max(obs_vals.max(), y_pred_obs.max())
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax[1].set_title("Observed solution: noisy obs vs pred")
    ax[1].set_xlabel("Noisy observations")
    ax[1].set_ylabel("Predicted solution at obs")
    ax[1].grid(alpha=0.2)

    im = ax[2].imshow(x_err_map, cmap="viridis")
    ax[2].set_title("|Posterior mean input - GT input|")
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(cfg.save_dir / f"{cfg.save_prefix}_diagnostics.png", dpi=220)
    plt.close(fig)

    # Explicit posterior sample visualization (input field): 2 rows x 4 cols.
    fig, ax = plt.subplots(2, 4, figsize=(12.6, 6.2))
    ax = ax.reshape(-1)
    vmin_s = min(x_gt.min(), x_post_mean.min(), x_samples[:n_plot_samples].min())
    vmax_s = max(x_gt.max(), x_post_mean.max(), x_samples[:n_plot_samples].max())

    im = ax[0].imshow(x_gt, cmap="viridis", vmin=vmin_s, vmax=vmax_s)
    ax[0].set_title("GT input")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    im = ax[1].imshow(x_post_mean, cmap="viridis", vmin=vmin_s, vmax=vmax_s)
    ax[1].set_title("Posterior mean")
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    for i in range(6):
        panel = ax[i + 2]
        if i < n_plot_samples:
            panel.imshow(x_samples[i], cmap="viridis", vmin=vmin_s, vmax=vmax_s)
            panel.set_title(f"Sample #{i + 1}")
        else:
            panel.axis("off")
        panel.set_xticks([])
        panel.set_yticks([])

    fig.subplots_adjust(left=0.04, right=0.90, bottom=0.05, top=0.90, wspace=0.10, hspace=0.22)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    fig.colorbar(im, cax=cax)
    fig.suptitle(
        f"Posterior samples of input field | mean Rel_L2={rel_l2_samples_mean:.4e}",
        y=0.98,
    )
    fig.savefig(cfg.save_dir / f"{cfg.save_prefix}_posterior_samples.png", dpi=220)
    plt.close(fig)

    np.savez_compressed(
        cfg.save_dir / f"{cfg.save_prefix}_arrays.npz",
        posterior_samples_input=x_samples,
        posterior_mean_input=x_post_mean,
        posterior_std_input=x_post_std,
        gt_input=x_gt,
        gt_solution=y_gt,
        posterior_mean_solution=y_post_mean,
        obs_mask=obs_mask_np.reshape(*cfg.dims),
        obs_values=obs_vals,
        input_mse=input_mse,
        observed_solution_mse=observed_solution_mse,
        full_solution_mse=full_solution_mse,
        rel_l2_samples=rel_l2_samples,
        rel_l2_samples_mean=rel_l2_samples_mean,
        rel_l2_samples_min=rel_l2_samples_min,
        rel_l2_samples_max=rel_l2_samples_max,
    )

    print(f"Input MSE (posterior mean):         {input_mse:.6e}")
    print(f"Observed solution MSE:              {observed_solution_mse:.6e}")
    print(f"Full solution MSE (posterior mean): {full_solution_mse:.6e}")
    print(f"Mean Rel_L2 over posterior samples: {rel_l2_samples_mean:.6e}")
    print(f"Min Rel_L2 over posterior samples:  {rel_l2_samples_min:.6e}")
    print(f"Max Rel_L2 over posterior samples:  {rel_l2_samples_max:.6e}")
    print(f"Saved regression figures and arrays to: {cfg.save_dir}")
    print(f"Saved posterior sample figure to: {cfg.save_dir / f'{cfg.save_prefix}_posterior_samples.png'}")


def main() -> None:
    cfg = build_config(parse_args())
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)


    print(
        "Regression config: "
        f"pde={cfg.label}, sample_idx={cfg.test_sample_idx}, n_obs={cfg.n_observations}, "
        f"noise_var={cfg.noise_level}, n_samples={cfg.n_samples}, "
        f"anneal={cfg.annealing_steps}, ode_steps={cfg.ode_steps}, "
        f"langevin_steps={cfg.langevin_steps}, langevin_lr={cfg.langevin_lr}, "
        f"low_rank_cov_rank={cfg.low_rank_cov_rank}, grad_clip={cfg.grad_clip_norm}, "
        f"state_clamp={cfg.state_clamp}"
    )

    prior = build_prior_model(cfg)
    forward_model = build_forward_model(cfg)
    x_true_full, y_true_full = load_test_case(str(cfg.test_data_path), cfg.test_sample_idx)
    pos_mask, y_obs_part = create_random_observation(
        y_true_full=y_true_full,
        n_observations=cfg.n_observations,
        noise_level=cfg.noise_level,
        device=cfg.device,
    )
    t0 = time.perf_counter()
    posterior = run_surrogate_regression_FAPS(
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

    visualize_results(
        posterior=posterior,
        x_true_full=x_true_full,
        y_true_full=y_true_full,
        y_obs_part=y_obs_part,
        pos_mask=pos_mask,
        forward_model=forward_model,
        cfg=cfg,
    )
    elapsed = time.perf_counter() - t0
    print(f"Total regression runtime: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
