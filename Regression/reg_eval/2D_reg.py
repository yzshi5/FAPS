"""Masked regression on 2D field data using a trained OFM prior."""

from __future__ import annotations

import argparse
import importlib.util
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

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REGRESSION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import faps_utils as _faps_utils

sys.modules.setdefault("ofm_utils", _faps_utils)

from faps_utils.fno import FNO
from faps_utils.ofm_ind_likelihood import OFMModel

# Load 2D posterior metrics module (filename starts with a digit).
_metrics2d_path = REPO_ROOT / "faps_utils" / "2D_metrics.py"
_metrics2d_spec = importlib.util.spec_from_file_location("metrics_2d", _metrics2d_path)
if _metrics2d_spec is None or _metrics2d_spec.loader is None:
    raise ImportError(f"Could not load 2D metrics module from: {_metrics2d_path}")
_metrics2d_mod = importlib.util.module_from_spec(_metrics2d_spec)
_metrics2d_spec.loader.exec_module(_metrics2d_mod)
compute_all_metrics_2d = _metrics2d_mod.compute_all_metrics

Tensor = torch.Tensor

NS_TEST_DATA_PATH = "/net/ghisallo/scratch1/yshi5/OFM/dataset/N_S/ns_test_10000.npy"
BH_TEST_DATA_PATH = "/net/wintermute/scratch/agao3/OpFlow/Unoflow/m_1_1_40_all.npy"

DATASET_DEFAULTS = {
    "ns": {
        "label": "NS",
        "test_data_path": NS_TEST_DATA_PATH,
        "model_dir": REGRESSION_ROOT / "checkpoints" / "FAPS_prior",
        "checkpoint_name": "ns_epoch_300.pt",
        "results_dir": REGRESSION_ROOT / "outputs" / "Regression_results" / "NS_reg",
        "device": "cuda:2" if torch.cuda.is_available() else "cpu",
        "test_sample_idx": 100,
        "n_observations": 64,
        "noise_level": 1e-2,
        "n_samples": 32,
        "low_rank_cov_samples": 256,
        "cmap": "RdBu_r",
    },
    "bh": {
        "label": "BH",
        "test_data_path": BH_TEST_DATA_PATH,
        "model_dir": REGRESSION_ROOT / "checkpoints" / "FAPS_prior",
        "checkpoint_name": "bh_epoch_300.pt",
        "results_dir": REGRESSION_ROOT / "outputs" / "Regression_results" / "BH_reg",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "test_sample_idx": 100,
        "n_observations": 256,
        "noise_level": 1e-3,
        "n_samples": 256,
        "low_rank_cov_samples": 256,
        "cmap": "viridis",
    },
    "custom": {
        "label": "2D",
        "test_data_path": None,
        "model_dir": REGRESSION_ROOT / "checkpoints" / "FAPS_prior",
        "checkpoint_name": None,
        "results_dir": REGRESSION_ROOT / "outputs" / "Regression_results" / "2D_reg",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "test_sample_idx": 0,
        "n_observations": 64,
        "noise_level": 1e-2,
        "n_samples": 128,
        "low_rank_cov_samples": 256,
        "cmap": "RdBu_r",
    },
}

@dataclass(frozen=True)
class EvalConfig:
    dataset: str
    label: str
    checkpoint_path: Optional[Path]
    model_dir: Path
    flat_checkpoint_dir: Path
    checkpoint_name: Optional[str]
    test_data_path: Path
    results_dir: Path
    device: str
    seed: int
    n_x: int
    n_y: int
    modes: int
    width: int
    mlp_width: int
    epochs: int
    sigma_min: float
    kernel_length: float
    kernel_variance: float
    kernel_nu: float
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
    save_prefix: str
    cmap: str

    @property
    def dims(self) -> list[int]:
        return [self.n_x, self.n_y]

    @property
    def label_lower(self) -> str:
        return self.label.lower()


@dataclass
class FAPSSamples:
    samples: Tensor
    mean: Tensor
    std: Tensor
    trajectory: Optional[List[Tuple[float, Tensor]]] = None


class OFMMaskedFAPSSampler:
    """FAPS-style posterior sampler for masking-based regression."""

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
        low_rank_cov_ode_steps: int = 20,
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
            remaining_dims = max(self.n_points - rank, 1)
            res_var = torch.sum(eigvals[rank:]) / remaining_dims
            cov_lr = cov_lr + res_var * torch.eye(self.n_points, device=self.device, dtype=self.dtype)
            has_res_var = True
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
    ) -> FAPSSamples:
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
        return FAPSSamples(samples=samples, mean=mean, std=std, trajectory=trajectory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked posterior regression on 2D data with a trained OFM prior.")
    parser.add_argument("--dataset", choices=DATASET_DEFAULTS.keys(), default="ns")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional direct path to a checkpoint.")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--flat-checkpoint-dir", type=Path, default=REGRESSION_ROOT / "checkpoints" / "FAPS_prior")
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--test-data-path", type=Path, default=None)
    parser.add_argument("--test-sample-idx", type=int, default=None)
    parser.add_argument("--n-observations", "--n-obs", type=int, default=None)
    parser.add_argument("--noise-level", type=float, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--annealing-steps", type=int, default=40)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--langevin-steps", type=int, default=50)
    parser.add_argument("--langevin-lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--low-rank-cov-rank", type=int, default=32)
    parser.add_argument("--low-rank-cov-samples", type=int, default=None)
    parser.add_argument("--low-rank-cov-ode-steps", type=int, default=20)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--state-clamp", type=float, default=None)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", "--results-dir", type=Path, default=None)
    parser.add_argument("--save-prefix", type=str, default=None)
    parser.add_argument("--n-x", type=int, default=64)
    parser.add_argument("--n-y", type=int, default=64)
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sigma-min", type=float, default=1e-4)
    parser.add_argument("--kernel-length", type=float, default=0.01)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    parser.add_argument("--kernel-nu", type=float, default=0.5)
    parser.add_argument("--cmap", type=str, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    defaults = DATASET_DEFAULTS[args.dataset]
    test_data_path = args.test_data_path or defaults["test_data_path"]
    if test_data_path is None:
        raise ValueError("--test-data-path is required when --dataset custom is used.")

    return EvalConfig(
        dataset=args.dataset,
        label=defaults["label"],
        checkpoint_path=args.checkpoint.expanduser() if args.checkpoint else None,
        model_dir=(args.model_dir or defaults["model_dir"]).expanduser(),
        flat_checkpoint_dir=args.flat_checkpoint_dir.expanduser(),
        checkpoint_name=args.checkpoint_name or defaults["checkpoint_name"],
        test_data_path=Path(test_data_path).expanduser(),
        results_dir=(args.save_dir or defaults["results_dir"]).expanduser(),
        device=args.device or defaults["device"],
        seed=args.seed,
        n_x=args.n_x,
        n_y=args.n_y,
        modes=args.modes,
        width=args.width,
        mlp_width=args.mlp_width,
        epochs=args.epochs,
        sigma_min=args.sigma_min,
        kernel_length=args.kernel_length,
        kernel_variance=args.kernel_variance,
        kernel_nu=args.kernel_nu,
        test_sample_idx=args.test_sample_idx if args.test_sample_idx is not None else defaults["test_sample_idx"],
        n_observations=args.n_observations if args.n_observations is not None else defaults["n_observations"],
        noise_level=args.noise_level if args.noise_level is not None else defaults["noise_level"],
        n_samples=args.n_samples if args.n_samples is not None else defaults["n_samples"],
        annealing_steps=args.annealing_steps,
        ode_steps=args.ode_steps,
        langevin_steps=args.langevin_steps,
        langevin_lr=args.langevin_lr,
        tau=args.tau,
        low_rank_cov_rank=args.low_rank_cov_rank,
        low_rank_cov_samples=(
            args.low_rank_cov_samples
            if args.low_rank_cov_samples is not None
            else defaults["low_rank_cov_samples"]
        ),
        low_rank_cov_ode_steps=args.low_rank_cov_ode_steps,
        grad_clip_norm=args.grad_clip_norm,
        state_clamp=args.state_clamp,
        save_prefix=args.save_prefix or f"{defaults['label'].lower()}_reg",
        cmap=args.cmap or defaults["cmap"],
    )


def resolve_checkpoint_path(cfg: EvalConfig) -> Path:
    if cfg.checkpoint_path is not None:
        if cfg.checkpoint_path.exists():
            return cfg.checkpoint_path
        raise FileNotFoundError(f"Could not find checkpoint: {cfg.checkpoint_path}")

    candidates = [
        cfg.model_dir / f"epoch_{cfg.epochs}.pt",
        cfg.flat_checkpoint_dir / f"{cfg.label_lower}_epoch_{cfg.epochs}.pt",
    ]
    if cfg.checkpoint_name:
        candidates.insert(0, cfg.model_dir / cfg.checkpoint_name)
        candidates.append(cfg.flat_checkpoint_dir / cfg.checkpoint_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find 2D prior checkpoint. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def build_prior_model(cfg: EvalConfig) -> OFMModel:
    checkpoint_path = resolve_checkpoint_path(cfg)
    print(f"Loading OFM prior checkpoint from: {checkpoint_path}")

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


def load_test_case(cfg: EvalConfig) -> Tensor:
    array = np.load(cfg.test_data_path)
    x_test = torch.from_numpy(array).float()
    if x_test.ndim == 3:
        x_test = x_test.unsqueeze(1)
    if x_test.ndim != 4:
        raise ValueError(f"Expected test data shape [N, C, H, W] or [N, H, W], got {tuple(x_test.shape)}")
    if tuple(x_test.shape[-2:]) != tuple(cfg.dims):
        raise ValueError(
            f"Configured dims {cfg.dims} do not match test data spatial shape {tuple(x_test.shape[-2:])}. "
            "Pass --n-x/--n-y to match the checkpoint and data."
        )
    if cfg.test_sample_idx < 0 or cfg.test_sample_idx >= x_test.shape[0]:
        raise IndexError(f"test_sample_idx={cfg.test_sample_idx} out of range [0, {x_test.shape[0] - 1}]")
    return x_test[cfg.test_sample_idx : cfg.test_sample_idx + 1, :1]


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


def run_masked_regression_FAPS(
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
    grad_clip_norm: Optional[float],
    state_clamp: Optional[float],
    device: str,
) -> FAPSSamples:
    sampler = OFMMaskedFAPSSampler(
        G=G,
        dims=dims,
        pos_mask=pos_mask,
        u_obs_part=u_obs_part,
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
    u_obs_full: Tensor,
    u_obs_part: Tensor,
    pos_mask: Tensor,
    save_dir: Path,
    save_prefix: str,
    dims: Sequence[int],
    cmap: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    gt = u_obs_full.squeeze().cpu().numpy()
    post_mean = posterior.mean[0].cpu().numpy()
    post_std = posterior.std[0].cpu().numpy()
    samples = posterior.samples[:, 0].cpu().numpy()
    metric_sample_idx = int(np.random.randint(samples.shape[0]))
    sample_show = samples[metric_sample_idx]

    flat_gt = gt.reshape(-1)
    flat_mean = post_mean.reshape(-1)
    flat_std = post_std.reshape(-1)
    obs_mask_np = pos_mask.cpu().numpy().astype(bool)
    obs_idx = np.where(obs_mask_np)[0]
    obs_vals = u_obs_part.detach().cpu().numpy().reshape(-1)

    observed_mse = float(np.mean((flat_mean[obs_mask_np] - flat_gt[obs_mask_np]) ** 2))
    unobserved_mse = float(np.mean((flat_mean[~obs_mask_np] - flat_gt[~obs_mask_np]) ** 2))
    full_mse = float(np.mean((flat_mean - flat_gt) ** 2))

    mask_2d = obs_mask_np.reshape(*dims)
    obs_y, obs_x = np.where(mask_2d)
    vmin = min(gt.min(), post_mean.min(), sample_show.min())
    vmax = max(gt.max(), post_mean.max(), sample_show.max())

    obs_pred = flat_mean[obs_mask_np]

    # 2D posterior metrics:
    # - CRPS / SSR: use all posterior samples.
    # - PSNR / SSIM / Relative L2: use one random posterior sample.
    gt_t = torch.from_numpy(gt[None, ...]).float()
    posterior_samples_t = torch.from_numpy(samples).float()
    one_sample_t = torch.from_numpy(sample_show[None, ...]).float()
    data_range = float(max(gt.max() - gt.min(), 1e-6))
    metrics_2d = compute_all_metrics_2d(
        posterior_samples=posterior_samples_t,
        one_sample=one_sample_t,
        target=gt_t,
        data_range=data_range,
    )
    crps_val = float(metrics_2d["CRPS"].detach().cpu().item())
    ssr_val = float(metrics_2d["SSR"].detach().cpu().item())
    psnr_val = float(metrics_2d["PSNR"].detach().cpu().item())
    ssim_val = float(metrics_2d["SSIM"].detach().cpu().item())
    rel_l2_val = float(metrics_2d["Relative_L2"].detach().cpu().item())

    fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.2))
    ax = ax.reshape(-1)
    subplot_title_fs = 21#13
    im0 = ax[0].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title("Ground truth test field", fontsize=subplot_title_fs)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    ax[1].imshow(gt, cmap="gray", alpha=0.4)
    obs_im = ax[1].scatter(obs_x, obs_y, c=obs_vals, cmap=cmap, s=26, edgecolors="k", linewidths=0.3)
    ax[1].set_title(f"Observed points (n={obs_idx.size})", fontsize=subplot_title_fs)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.colorbar(obs_im, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(sample_show, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2].set_title("One posterior sample", fontsize=subplot_title_fs)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(post_mean, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[3].set_title("Posterior mean", fontsize=subplot_title_fs)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    im4 = ax[4].imshow(post_std, cmap="magma")
    ax[4].set_title("Posterior std", fontsize=subplot_title_fs)
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

    ax[5].scatter(
        obs_vals,
        obs_pred,
        s=22,
        alpha=0.9,
        c="#4C72B0",
        edgecolors="white",
        linewidths=0.4,
    )
    lo = min(obs_vals.min(), obs_pred.min())
    hi = max(obs_vals.max(), obs_pred.max())
    ax[5].plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax[5].set_title("Noisy obs vs mean prediction", fontsize=subplot_title_fs)
    ax[5].set_xlabel("Noisy observations", fontsize=subplot_title_fs)
    ax[5].set_ylabel("Posterior mean at obs", fontsize=subplot_title_fs)
    ax[5].grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_dir / f"{save_prefix}_reconstruction.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    hist_gt, edges_gt = np.histogram(flat_gt, bins=60, range=(-4, 4), density=True)
    hist_post, edges_post = np.histogram(samples.reshape(-1), bins=60, range=(-4, 4), density=True)
    centers_gt = 0.5 * (edges_gt[1:] + edges_gt[:-1])
    centers_post = 0.5 * (edges_post[1:] + edges_post[:-1])
    ax[0].plot(centers_gt, hist_gt, c="k", lw=2.0, label="GT test field")
    ax[0].plot(centers_post, hist_post, c="r", lw=2.0, ls="--", label="Posterior samples")
    ax[0].set_title("Value histogram")
    ax[0].set_xlabel("Value")
    ax[0].legend(loc="upper right")

    ax[1].scatter(
        obs_vals,
        obs_pred,
        s=22,
        alpha=0.9,
        c="#4C72B0",
        edgecolors="white",
        linewidths=0.4,
        label="Posterior mean",
    )
    lo = min(obs_vals.min(), obs_pred.min())
    hi = max(obs_vals.max(), obs_pred.max())
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1.2)
    ax[1].set_title("Observed values: noisy obs vs mean prediction")
    ax[1].set_xlabel("Noisy observations")
    ax[1].set_ylabel("Posterior mean at obs")
    ax[1].grid(alpha=0.2)
    ax[1].legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(save_dir / f"{save_prefix}_diagnostics.png", dpi=220)
    plt.close(fig)

    np.savez_compressed(
        save_dir / f"{save_prefix}_arrays.npz",
        posterior_samples=samples,
        posterior_mean=post_mean,
        posterior_std=post_std,
        gt_field=gt,
        obs_mask=obs_mask_np.reshape(*dims),
        obs_values=obs_vals,
        observed_mse=observed_mse,
        unobserved_mse=unobserved_mse,
        full_mse=full_mse,
        CRPS=crps_val,
        SSR=ssr_val,
        PSNR=psnr_val,
        SSIM=ssim_val,
        Relative_L2=rel_l2_val,
        metric_sample_idx=metric_sample_idx,
    )

    print(f"Observed MSE:   {observed_mse:.6e}")
    print(f"Unobserved MSE: {unobserved_mse:.6e}")
    print(f"Full MSE:       {full_mse:.6e}")
    print(
        f"Posterior metrics (all-sample): CRPS={crps_val:.6e}, SSR={ssr_val:.6e}"
    )
    print(
        f"Posterior metrics (one-sample idx={metric_sample_idx}): "
        f"PSNR={psnr_val:.6e}, SSIM={ssim_val:.6e}, Relative_L2={rel_l2_val:.6e}"
    )
    print(f"Saved regression figures and arrays to: {save_dir}")


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(
        "Regression config: "
        f"dataset={cfg.label}, dims={cfg.dims}, sample_idx={cfg.test_sample_idx}, "
        f"n_obs={cfg.n_observations}, noise_var={cfg.noise_level}, "
        f"n_samples={cfg.n_samples}, anneal={cfg.annealing_steps}, ode_steps={cfg.ode_steps}, "
        f"langevin_steps={cfg.langevin_steps}, langevin_lr={cfg.langevin_lr}, "
        f"low_rank_cov_rank={cfg.low_rank_cov_rank}, low_rank_cov_samples={cfg.low_rank_cov_samples}, "
        f"grad_clip={cfg.grad_clip_norm}, "
        f"state_clamp={cfg.state_clamp}"
    )
    fmot = build_prior_model(cfg)
    u_obs_full = load_test_case(cfg)
    pos_mask, u_obs_part = create_random_observation(
        u_obs_full=u_obs_full,
        n_observations=cfg.n_observations,
        noise_level=cfg.noise_level,
        device=cfg.device,
    )
    
    t0 = time.perf_counter()
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
        low_rank_cov_rank=cfg.low_rank_cov_rank,
        low_rank_cov_samples=cfg.low_rank_cov_samples,
        low_rank_cov_ode_steps=cfg.low_rank_cov_ode_steps,
        grad_clip_norm=cfg.grad_clip_norm,
        state_clamp=cfg.state_clamp,
        device=cfg.device,
    )

    visualize_results(
        posterior=posterior,
        u_obs_full=u_obs_full,
        u_obs_part=u_obs_part,
        pos_mask=pos_mask,
        save_dir=cfg.results_dir,
        save_prefix=cfg.save_prefix,
        dims=cfg.dims,
        cmap=cfg.cmap,
    )
    elapsed = time.perf_counter() - t0
    print(f"Total regression runtime: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
