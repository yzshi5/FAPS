"""Masked regression on global weather using a trained MINO OFM prior."""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline
from torchdiffeq import odeint

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

Tensor = torch.Tensor

GLOBAL_VIEW = (100.0, 0.0)
GLOBAL_SMOOTH_FACTOR = 1.0


@dataclass(frozen=True)
class EvalConfig:
    test_data_path: Path
    checkpoint_path: Optional[Path]
    model_dir: Path
    flat_checkpoint_dir: Path
    results_dir: Path
    save_prefix: str
    device: str
    seed: int
    n_longs: int
    n_lats: int
    epochs: int
    sigma_min: float
    kernel_length: float
    kernel_variance: float
    kernel_nu: float
    mino_x_dim: int
    mino_query_longs: int
    mino_query_lats: int
    mino_co_domain: int
    mino_radius: float
    mino_dim: int
    mino_num_heads: int
    mino_enc_depth: int
    mino_dec_depth: int
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
    low_rank_cov_batch_size: int
    grad_clip_norm: Optional[float]
    state_clamp: Optional[float]

    @property
    def dims(self) -> list[int]:
        return [self.n_longs, self.n_lats]

    @property
    def mino_query_dims(self) -> tuple[int, int]:
        return (self.mino_query_longs, self.mino_query_lats)


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


def flip_xy(field_2d: np.ndarray) -> np.ndarray:
    """Swap x/y axes for visualization consistency with weather script."""
    return np.swapaxes(field_2d, -2, -1)


def plot_global_map(
    ax: plt.Axes,
    field_lonlat: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    title: str,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    smooth_factor: float = GLOBAL_SMOOTH_FACTOR,
):
    """Plot climate data on a cartopy globe (Climate_MINO_T style)."""
    ax.coastlines()
    field_lonlat = np.asarray(field_lonlat)
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)

    if smooth_factor == 1.0:
        mesh = ax.pcolormesh(
            longitudes,
            latitudes,
            field_lonlat.T,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
    else:
        temp_latlon = field_lonlat.T
        lat_asc = latitudes
        if latitudes[0] > latitudes[-1]:
            lat_asc = latitudes[::-1]
            temp_latlon = temp_latlon[::-1, :]

        interp_func = RectBivariateSpline(lat_asc, longitudes, temp_latlon)
        n_lats = max(int(smooth_factor * len(latitudes)), len(latitudes))
        n_lons = max(int(smooth_factor * len(longitudes)), len(longitudes))
        lat_new = np.linspace(lat_asc[0], lat_asc[-1], n_lats)
        lon_new = np.linspace(longitudes[0], longitudes[-1], n_lons, endpoint=False)
        temp_new = interp_func(lat_new, lon_new)

        if latitudes[0] > latitudes[-1]:
            lat_plot = lat_new[::-1]
            temp_plot = temp_new[::-1, :]
        else:
            lat_plot = lat_new
            temp_plot = temp_new

        mesh = ax.pcolormesh(
            lon_new,
            lat_plot,
            temp_plot,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )

    ax.set_title(title, fontsize=18)
    return mesh


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


@dataclass
class DAPSSamples:
    samples: Tensor
    mean: Tensor
    std: Tensor
    trajectory: Optional[List[Tuple[float, Tensor]]] = None


class OFMMaskedDAPSSampler:
    """DAPS-style posterior sampler for masking-based regression."""

    def __init__(
        self,
        G: OFMModel,
        dims: Sequence[int],
        pos_mask: Tensor,
        u_obs_part: Tensor,
        noise_level: float,
        *,
        tau: float = 1.0,
        anchor_std_base: float = 0.05,
        anchor_std_scale: float = 1.0,
        low_rank_cov_rank: int = 32,
        low_rank_cov_samples: int = 256,
        low_rank_cov_ode_steps: int = 24,
        low_rank_cov_batch_size: int = 8,
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
        self.anchor_std_base = float(anchor_std_base)
        self.anchor_std_scale = float(anchor_std_scale)
        self.low_rank_cov_rank = int(max(0, low_rank_cov_rank))
        self.low_rank_cov_samples = int(max(0, low_rank_cov_samples))
        self.low_rank_cov_ode_steps = int(max(1, low_rank_cov_ode_steps))
        self.low_rank_cov_batch_size = int(max(1, low_rank_cov_batch_size))
        self.low_rank_cov_jitter = float(max(0.0, low_rank_cov_jitter))

        if list(getattr(self.G.gp, "dims", [])) == self.dims:
            self.gp_dist = self.G.gp.base_dist
        else:
            self.gp_dist = self.G.gp.new_dist(self.dims)

        self.base_scale_tril = self.gp_dist.scale_tril.to(self.device, dtype=self.dtype)
        self.n_channels = int(self.u_obs_part.shape[1])

        self.gp_cov = self._estimate_low_rank_covariance() if self.low_rank_cov_rank > 0 else None
        if self.gp_cov is not None:
            eye = torch.eye(self.n_points, device=self.device, dtype=self.dtype)
            self.gp_cov = 0.5 * (self.gp_cov + self.gp_cov.T) + self.low_rank_cov_jitter * eye
            self.cov_scale_tril = torch.linalg.cholesky(self.gp_cov)
            print(
                f"Using covariance source: low-rank surrogate "
                f"(rank={self.low_rank_cov_rank}, samples={self.low_rank_cov_samples}, "
                f"batch_size={self.low_rank_cov_batch_size})"
            )
        else:
            self.cov_scale_tril = None
            print("Using covariance source: identity (no low-rank surrogate)")

    @torch.no_grad()
    def _estimate_low_rank_covariance(self) -> Optional[Tensor]:
        if self.low_rank_cov_samples < 2:
            return None
        n = self.low_rank_cov_samples
        chunk_size = min(self.low_rank_cov_batch_size, n)
        chunks: list[Tensor] = []
        processed = 0
        while processed < n:
            cur = min(chunk_size, n - processed)
            u0 = self.bridge_std(0.0) * self._sample_base_gp(cur)
            u1 = self.endpoint_anchor(
                u0,
                t_start=0.0,
                ode_steps=self.low_rank_cov_ode_steps,
                ode_method="euler",
                rtol=1e-5,
                atol=1e-5,
            )
            chunks.append(self._flatten(u1).reshape(cur * self.n_channels, self.n_points).cpu())
            processed += cur
            del u0, u1
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        u_flat = torch.cat(chunks, dim=0).to(dtype=self.dtype)
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
            cov_lr = cov_lr + res_var * torch.eye(self.n_points, dtype=self.dtype)
        return cov_lr.to(self.device, dtype=self.dtype)

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
        u_flat = self._flatten(u)
        grad = torch.zeros_like(u_flat)
        y = self.u_obs_part.expand(u.shape[0], -1, -1)
        residual = self._apply_mask(u) - y
        grad[..., self.pos_mask] = -residual / self.obs_noise_var
        return self._unflatten(grad)

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
        rtol: float = 1e-5,
        atol: float = 1e-5,
        return_path: bool = False,
        record_every: int = 4,
        init: Optional[Tensor] = None,
    ) -> DAPSSamples:
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

            u_hat = self.endpoint_anchor(
                u_t,
                t_start=t_cur,
                ode_steps=ode_steps,
                ode_method=ode_method,
                rtol=rtol,
                atol=atol,
            )
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
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return DAPSSamples(samples=samples, mean=mean, std=std, trajectory=trajectory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Masked posterior regression on global weather with a trained MINO OFM prior."
    )
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=REGRESSION_ROOT / "checkpoints" / "FAPS_prior")
    parser.add_argument("--flat-checkpoint-dir", type=Path, default=REGRESSION_ROOT / "checkpoints" / "FAPS_prior")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--n-longs", type=int, default=90)
    parser.add_argument("--n-lats", type=int, default=46)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--kernel-length", type=float, default=0.05)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-nu", type=float, default=0.5)
    parser.add_argument("--mino-x-dim", type=int, default=3)
    parser.add_argument("--mino-query-longs", type=int, default=32)
    parser.add_argument("--mino-query-lats", type=int, default=16)
    parser.add_argument("--mino-co-domain", type=int, default=1)
    parser.add_argument("--mino-radius", type=float, default=0.2)
    parser.add_argument("--mino-dim", type=int, default=256)
    parser.add_argument("--mino-num-heads", type=int, default=4)
    parser.add_argument("--mino-enc-depth", type=int, default=5)
    parser.add_argument("--mino-dec-depth", type=int, default=2)
    parser.add_argument("--test-sample-idx", type=int, default=100)
    parser.add_argument("--n-observations", type=int, default=125)
    parser.add_argument("--noise-level", type=float, default=1e-3)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--annealing-steps", type=int, default=40)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--langevin-steps", type=int, default=50)
    parser.add_argument("--langevin-lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--low-rank-cov-rank", type=int, default=32)
    parser.add_argument("--low-rank-cov-samples", type=int, default=512)
    parser.add_argument("--low-rank-cov-ode-steps", type=int, default=20)
    parser.add_argument("--low-rank-cov-batch-size", type=int, default=64)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--state-clamp", type=float, default=None)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=Path, default=REGRESSION_ROOT / "outputs" / "Regression_results" / "weather_reg")
    parser.add_argument("--save-prefix", type=str, default="weather_reg")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        test_data_path=args.test_data_path.expanduser(),
        checkpoint_path=args.checkpoint_path.expanduser() if args.checkpoint_path else None,
        model_dir=args.model_dir.expanduser(),
        flat_checkpoint_dir=args.flat_checkpoint_dir.expanduser(),
        results_dir=args.save_dir.expanduser(),
        save_prefix=args.save_prefix,
        device=args.device,
        seed=args.seed,
        n_longs=args.n_longs,
        n_lats=args.n_lats,
        epochs=args.epochs,
        sigma_min=args.sigma_min,
        kernel_length=args.kernel_length,
        kernel_variance=args.kernel_variance,
        kernel_nu=args.kernel_nu,
        mino_x_dim=args.mino_x_dim,
        mino_query_longs=args.mino_query_longs,
        mino_query_lats=args.mino_query_lats,
        mino_co_domain=args.mino_co_domain,
        mino_radius=args.mino_radius,
        mino_dim=args.mino_dim,
        mino_num_heads=args.mino_num_heads,
        mino_enc_depth=args.mino_enc_depth,
        mino_dec_depth=args.mino_dec_depth,
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
        low_rank_cov_batch_size=args.low_rank_cov_batch_size,
        grad_clip_norm=args.grad_clip_norm,
        state_clamp=args.state_clamp,
    )


def resolve_checkpoint_path(cfg: EvalConfig) -> Path:
    if cfg.checkpoint_path is not None:
        if cfg.checkpoint_path.exists():
            return cfg.checkpoint_path
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

    candidates = [
        cfg.model_dir / f"epoch_{cfg.epochs}.pt",
        cfg.flat_checkpoint_dir / f"weather_epoch_{cfg.epochs}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find weather prior checkpoint. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def build_prior_model(cfg: EvalConfig) -> OFMModel:
    checkpoint_path = resolve_checkpoint_path(cfg)
    print(f"Loading OFM prior checkpoint from: {checkpoint_path}")

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
    model = MINOBackboneWrapper(mino=mino, query_dims=cfg.mino_query_dims).to(cfg.device)
    for param in model.parameters():
        param.requires_grad = False

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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


def load_test_case(test_data_path: str, test_sample_idx: int) -> Tensor:
    array = np.load(test_data_path)
    x_test = torch.from_numpy(array).float()
    if x_test.ndim != 4 or x_test.shape[1] < 3:
        raise ValueError(f"Expected weather data shape [N, 3, 46, 90], got {tuple(x_test.shape)}")
    if test_sample_idx < 0 or test_sample_idx >= x_test.shape[0]:
        raise IndexError(f"test_sample_idx={test_sample_idx} out of range [0, {x_test.shape[0] - 1}]")
    sample = x_test[test_sample_idx : test_sample_idx + 1]
    # Channel 2 is climate field; permute to [N, 1, 90, 46] to match the prior.
    return sample[:, 2:3].permute(0, 1, 3, 2)


def load_test_coordinates(test_data_path: str, test_sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Load per-sample latitude/longitude vectors from weather tensor channels."""
    array = np.load(test_data_path)
    if test_sample_idx < 0 or test_sample_idx >= array.shape[0]:
        raise IndexError(f"test_sample_idx={test_sample_idx} out of range [0, {array.shape[0] - 1}]")
    sample = array[test_sample_idx]
    latitudes = sample[0, :, 0]
    longitudes = sample[1, 0, :]
    return latitudes.astype(np.float32), longitudes.astype(np.float32)


def create_random_observation(
    u_obs_full: Tensor,
    n_observations: int,
    noise_level: float,
    *,
    device: str,
) -> tuple[Tensor, Tensor]:
    n_points = int(np.prod(u_obs_full.shape[-2:]))
    if n_observations <= 0 or n_observations > n_points:
        raise ValueError(f"n_observations must be in [1, {n_points}], got {n_observations}.")

    pos_mask = torch.zeros(n_points, dtype=torch.bool)
    pos_idx = torch.randperm(n_points)[:n_observations]
    pos_mask[pos_idx] = True

    u_obs_part = u_obs_full.reshape(1, 1, -1)[..., pos_mask].to(device)
    noise_pattern = torch.randn_like(u_obs_part) * math.sqrt(float(noise_level))
    u_obs_part = u_obs_part + noise_pattern
    return pos_mask, u_obs_part


def run_masked_regression_daps(
    G: OFMModel,
    dims: Sequence[int],
    pos_mask: Tensor,
    u_obs_part: Tensor,
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
    low_rank_cov_batch_size: int,
    grad_clip_norm: Optional[float],
    state_clamp: Optional[float],
    device: str,
) -> DAPSSamples:
    sampler = OFMMaskedDAPSSampler(
        G=G,
        dims=dims,
        pos_mask=pos_mask,
        u_obs_part=u_obs_part,
        noise_level=noise_level,
        tau=tau,
        low_rank_cov_rank=low_rank_cov_rank,
        low_rank_cov_samples=low_rank_cov_samples,
        low_rank_cov_ode_steps=low_rank_cov_ode_steps,
        low_rank_cov_batch_size=low_rank_cov_batch_size,
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
    posterior: DAPSSamples,
    u_obs_full: Tensor,
    u_obs_part: Tensor,
    pos_mask: Tensor,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    save_dir: Path,
    save_prefix: str,
    dims: Sequence[int],
    test_sample_idx: int,
    noise_level: float,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    gt = u_obs_full.squeeze().cpu().numpy()
    post_mean = posterior.mean[0].cpu().numpy()
    post_std = posterior.std[0].cpu().numpy()
    samples = posterior.samples[:, 0].cpu().numpy()
    sample_show = samples[0]

    flat_gt = gt.reshape(-1)
    flat_mean = post_mean.reshape(-1)
    flat_std = post_std.reshape(-1)
    obs_mask_np = pos_mask.cpu().numpy().astype(bool)
    obs_idx = np.where(obs_mask_np)[0]
    obs_vals = u_obs_part.detach().cpu().numpy().reshape(-1)

    observed_mse = float(np.mean((flat_mean[obs_mask_np] - flat_gt[obs_mask_np]) ** 2))
    unobserved_mse = float(np.mean((flat_mean[~obs_mask_np] - flat_gt[~obs_mask_np]) ** 2))
    full_mse = float(np.mean((flat_mean - flat_gt) ** 2))

    mask_2d = obs_mask_np.reshape(dims[0], dims[1])
    mask_2d_vis = flip_xy(mask_2d)
    obs_y, obs_x = np.where(mask_2d)
    obs_y_vis, obs_x_vis = np.where(mask_2d_vis)
    err_map = np.abs(post_mean - gt)

    vmin = min(gt.min(), post_mean.min(), sample_show.min())
    vmax = max(gt.max(), post_mean.max(), sample_show.max())

    subplot_title_fs = 18
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.5))
    im0 = ax[0, 0].imshow(flip_xy(gt), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax[0, 0].set_title("Ground truth test field", fontsize=subplot_title_fs)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(flip_xy(post_mean), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax[0, 1].set_title("Posterior mean", fontsize=subplot_title_fs)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[0, 2].imshow(flip_xy(sample_show), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax[0, 2].set_title("One posterior sample", fontsize=subplot_title_fs)
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

    im3 = ax[1, 0].imshow(flip_xy(post_std), cmap="magma")
    ax[1, 0].set_title("Posterior std", fontsize=subplot_title_fs)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    plt.colorbar(im3, ax=ax[1, 0], fraction=0.046, pad=0.04)

    im4 = ax[1, 1].imshow(flip_xy(err_map), cmap="viridis")
    ax[1, 1].set_title("|Posterior mean - GT|", fontsize=subplot_title_fs)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    plt.colorbar(im4, ax=ax[1, 1], fraction=0.046, pad=0.04)

    ax[1, 2].imshow(flip_xy(gt), cmap="gray", alpha=0.4)
    ax[1, 2].scatter(obs_x_vis, obs_y_vis, c=obs_vals, cmap="RdBu_r", s=26, edgecolors="k", linewidths=0.3)
    ax[1, 2].set_title(f"Observed points (n={obs_idx.size})", fontsize=subplot_title_fs)
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])

    fig.suptitle(
        f"Weather masked regression | sample_idx={test_sample_idx} | noise_var={noise_level:.3e}",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(save_dir / f"{save_prefix}_reconstruction.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.8))
    hist_gt, edges_gt = np.histogram(flat_gt, bins=60, range=(-4, 4), density=True)
    hist_post, edges_post = np.histogram(samples.reshape(-1), bins=60, range=(-4, 4), density=True)
    centers_gt = 0.5 * (edges_gt[1:] + edges_gt[:-1])
    centers_post = 0.5 * (edges_post[1:] + edges_post[:-1])
    ax[0].plot(centers_gt, hist_gt, c="k", lw=2.0, label="GT test field")
    ax[0].plot(centers_post, hist_post, c="r", lw=2.0, ls="--", label="Posterior samples")
    ax[0].set_title("Value histogram")
    ax[0].set_xlabel("Value")
    ax[0].legend(loc="upper right")

    obs_pred = flat_mean[obs_mask_np]
    ax[1].scatter(obs_vals, obs_pred, s=22, alpha=0.8, c="#4C72B0", edgecolors="white", linewidths=0.4)
    lo = min(obs_vals.min(), obs_pred.min())
    hi = max(obs_vals.max(), obs_pred.max())
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax[1].set_title("Noisy obs vs posterior mean", fontsize=subplot_title_fs)
    ax[1].set_xlabel("Noisy observations", fontsize=subplot_title_fs)
    ax[1].set_ylabel("Posterior mean at obs", fontsize=subplot_title_fs)
    ax[1].grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_dir / f"{save_prefix}_diagnostics.png", dpi=220)
    plt.close(fig)

    # Additional global-view plots for easier climate interpretation.
    fig = plt.figure(figsize=(18, 8.5))
    ax = np.array(
        [
            fig.add_subplot(2, 3, 1, projection=ccrs.Orthographic(*GLOBAL_VIEW)),
            fig.add_subplot(2, 3, 2, projection=ccrs.Orthographic(*GLOBAL_VIEW)),
            fig.add_subplot(2, 3, 3, projection=ccrs.Orthographic(*GLOBAL_VIEW)),
            fig.add_subplot(2, 3, 4, projection=ccrs.Orthographic(*GLOBAL_VIEW)),
            fig.add_subplot(2, 3, 5, projection=ccrs.Orthographic(*GLOBAL_VIEW)),
            fig.add_subplot(2, 3, 6, projection=ccrs.Orthographic(*GLOBAL_VIEW)),
        ]
    ).reshape(2, 3)
    im0 = plot_global_map(
        ax[0, 0],
        gt,
        latitudes,
        longitudes,
        title="GT climate field",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        smooth_factor=GLOBAL_SMOOTH_FACTOR,
    )
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.04, pad=0.03)
    lon_idx, lat_idx = np.where(mask_2d)
    obs_lons = longitudes[lon_idx]
    obs_lats = latitudes[lat_idx]
    ax[0, 1].coastlines()
    im1 = ax[0, 1].scatter(
        obs_lons,
        obs_lats,
        c=obs_vals,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        s=18,
        edgecolors="k",
        linewidths=0.2,
        transform=ccrs.PlateCarree(),
    )
    ax[0, 1].set_title(f"Observations (n={obs_idx.size})", fontsize=subplot_title_fs)
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.04, pad=0.03)
    im2 = plot_global_map(
        ax[0, 2],
        sample_show,
        latitudes,
        longitudes,
        title="One posterior sample",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        smooth_factor=GLOBAL_SMOOTH_FACTOR,
    )
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.04, pad=0.03)
    im3 = plot_global_map(
        ax[1, 0],
        post_mean,
        latitudes,
        longitudes,
        title="Posterior mean",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        smooth_factor=GLOBAL_SMOOTH_FACTOR,
    )
    plt.colorbar(im3, ax=ax[1, 0], fraction=0.04, pad=0.03)
    im4 = plot_global_map(
        ax[1, 1],
        post_std,
        latitudes,
        longitudes,
        title="Posterior std",
        cmap="magma",
        smooth_factor=GLOBAL_SMOOTH_FACTOR,
    )
    plt.colorbar(im4, ax=ax[1, 1], fraction=0.04, pad=0.03)
    im5 = plot_global_map(
        ax[1, 2],
        err_map,
        latitudes,
        longitudes,
        title="|Posterior mean - GT|",
        cmap="viridis",
        smooth_factor=GLOBAL_SMOOTH_FACTOR,
    )
    plt.colorbar(im5, ax=ax[1, 2], fraction=0.04, pad=0.03)
    # fig.suptitle(
    #     f"Weather masked regression global maps | sample_idx={test_sample_idx} | noise_var={noise_level:.3e}",
    #     y=0.99,
    # )
    plt.tight_layout()
    fig.savefig(save_dir / f"{save_prefix}_global_maps.png", dpi=220)
    plt.close(fig)

    np.savez_compressed(
        save_dir / f"{save_prefix}_arrays.npz",
        posterior_samples=samples,
        posterior_mean=post_mean,
        posterior_std=post_std,
        gt_field=gt,
        obs_mask=obs_mask_np.reshape(dims[0], dims[1]),
        obs_values=obs_vals,
        observed_mse=observed_mse,
        unobserved_mse=unobserved_mse,
        full_mse=full_mse,
    )

    print(f"Observed MSE:   {observed_mse:.6e}")
    print(f"Unobserved MSE: {unobserved_mse:.6e}")
    print(f"Full MSE:       {full_mse:.6e}")
    print(f"Saved regression figures and arrays to: {save_dir}")


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    t0 = time.perf_counter()
    print(
        "Regression config: "
        f"dims={cfg.dims}, sample_idx={cfg.test_sample_idx}, "
        f"n_obs={cfg.n_observations}, noise_var={cfg.noise_level}, "
        f"n_samples={cfg.n_samples}, anneal={cfg.annealing_steps}, ode_steps={cfg.ode_steps}, "
        f"langevin_steps={cfg.langevin_steps}, langevin_lr={cfg.langevin_lr}, "
        f"low_rank_cov_rank={cfg.low_rank_cov_rank}, grad_clip={cfg.grad_clip_norm}, "
        f"state_clamp={cfg.state_clamp}"
    )
    fmot = build_prior_model(cfg)
    u_obs_full = load_test_case(cfg.test_data_path, cfg.test_sample_idx)
    latitudes, longitudes = load_test_coordinates(cfg.test_data_path, cfg.test_sample_idx)
    pos_mask, u_obs_part = create_random_observation(
        u_obs_full=u_obs_full,
        n_observations=cfg.n_observations,
        noise_level=cfg.noise_level,
        device=cfg.device,
    )

    posterior = run_masked_regression_daps(
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
        low_rank_cov_rank=cfg.low_rank_cov_rank,
        low_rank_cov_samples=cfg.low_rank_cov_samples,
        low_rank_cov_ode_steps=cfg.low_rank_cov_ode_steps,
        low_rank_cov_batch_size=cfg.low_rank_cov_batch_size,
        grad_clip_norm=cfg.grad_clip_norm,
        state_clamp=cfg.state_clamp,
        device=cfg.device,
    )

    visualize_results(
        posterior=posterior,
        u_obs_full=u_obs_full,
        u_obs_part=u_obs_part,
        pos_mask=pos_mask,
        latitudes=latitudes,
        longitudes=longitudes,
        save_dir=cfg.results_dir,
        save_prefix=cfg.save_prefix,
        dims=cfg.dims,
        test_sample_idx=cfg.test_sample_idx,
        noise_level=cfg.noise_level,
    )
    elapsed = time.perf_counter() - t0
    print(f"Total regression runtime: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
