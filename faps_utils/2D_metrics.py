import torch
import torch.nn.functional as F
from typing import Dict


def _to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() if not torch.is_floating_point(x) else x


def _check_ensemble_inputs(
    samples: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    samples: [N, H, W]
    target:  [1, H, W]
    """
    samples = _to_float(samples)
    target = _to_float(target)

    if samples.ndim != 3:
        raise ValueError(f"`samples` must have shape [N, H, W], got {tuple(samples.shape)}.")
    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError(f"`target` must have shape [1, H, W], got {tuple(target.shape)}.")
    if samples.shape[1:] != target.shape[1:]:
        raise ValueError(
            f"Spatial shapes must match: samples {tuple(samples.shape)}, target {tuple(target.shape)}."
        )
    return samples, target


def _check_point_inputs(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    pred:   [1, H, W]
    target: [1, H, W]
    """
    pred = _to_float(pred)
    target = _to_float(target)

    if pred.ndim != 3 or pred.shape[0] != 1:
        raise ValueError(f"`pred` must have shape [1, H, W], got {tuple(pred.shape)}.")
    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError(f"`target` must have shape [1, H, W], got {tuple(target.shape)}.")
    if pred.shape != target.shape:
        raise ValueError(f"Shapes must match: pred {tuple(pred.shape)}, target {tuple(target.shape)}.")
    return pred, target


def relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Relative L2 error:
        ||pred - target||_2 / ||target||_2

    Inputs:
        pred:   [1, H, W]
        target: [1, H, W]

    Returns:
        scalar tensor
    """
    pred, target = _check_point_inputs(pred, target)
    num = torch.linalg.vector_norm(pred - target)
    den = torch.linalg.vector_norm(target).clamp_min(eps)
    return num / den


def crps_ensemble(
    samples: torch.Tensor,
    target: torch.Tensor,
    fair: bool = False,
) -> torch.Tensor:
    """
    Grid-point finite-ensemble CRPS averaged over all pixels.

    Standard finite-ensemble estimator:
        CRPS = mean_m |x_m - y|
               - 1/(2 M^2) sum_{m,k} |x_m - x_k|

    Implemented with the sorted Zamo-Naveau equivalent, avoiding O(M^2) memory.

    Inputs:
        samples: [N, H, W] posterior samples
        target:  [1, H, W] ground truth

    Args:
        fair:
            False: standard empirical CRPS with denominator M^2.
            True: fair finite-ensemble correction with denominator M(M-1).
                  Only valid for N >= 2.

    Returns:
        scalar tensor
    """
    samples, target = _check_ensemble_inputs(samples, target)
    M = samples.shape[0]

    if fair and M < 2:
        raise ValueError("Fair CRPS requires at least 2 ensemble members.")

    # First term: E |X - y|
    abs_error = (samples - target).abs().mean(dim=0)  # [H, W]

    # Second term: 0.5 E |X - X'|
    # For sorted x_(i), sum_{i,j} |x_i - x_j|
    # = 2 * sum_i (2i - M - 1) x_(i), using 1-based i.
    xs = torch.sort(samples, dim=0).values  # [M, H, W]
    coeff = torch.arange(M, device=samples.device, dtype=samples.dtype)
    coeff = (2 * coeff - M + 1).view(M, 1, 1)  # zero-based equivalent

    denom = M * (M - 1) if fair else M * M
    spread_term = (coeff * xs).sum(dim=0) / denom  # [H, W]

    crps_map = abs_error - spread_term
    return crps_map.mean()


def spread_skill_ratio(
    samples: torch.Tensor,
    target: torch.Tensor,
    finite_ensemble_correction: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Spread-skill ratio, averaged over spatial grid points.

    Skill:
        RMSE of ensemble mean against ground truth.

    Spread:
        RMS ensemble standard deviation.

    For a finite ensemble of size M, if target and ensemble members are exchangeable,
    E[(ensemble_mean - target)^2] = (1 + 1/M) * Var[X].
    The finite_ensemble_correction multiplies the unbiased ensemble variance by (M+1)/M,
    so a calibrated ensemble has SSR approximately 1.

    Inputs:
        samples: [N, H, W] posterior samples
        target:  [1, H, W] ground truth

    Returns:
        scalar tensor
    """
    samples, target = _check_ensemble_inputs(samples, target)
    M = samples.shape[0]

    if M < 2:
        raise ValueError("SSR requires at least 2 ensemble members.")

    ensemble_mean = samples.mean(dim=0, keepdim=True)  # [1, H, W]

    skill_sq = ((ensemble_mean - target) ** 2).mean()

    # Unbiased sample variance across posterior samples at each pixel.
    spread_var = samples.var(dim=0, unbiased=True)  # [H, W]

    if finite_ensemble_correction:
        spread_var = spread_var * ((M + 1.0) / M)

    spread_sq = spread_var.mean()

    return torch.sqrt(spread_sq.clamp_min(eps)) / torch.sqrt(skill_sq.clamp_min(eps))


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 20 log10(data_range) - 10 log10(MSE)

    Inputs:
        pred:   [1, H, W]
        target: [1, H, W]

    Args:
        data_range:
            1.0 for tensors in [0, 1].
            2.0 for tensors in [-1, 1].
            255.0 for uint8-scale image tensors.

    Returns:
        scalar tensor, in dB
    """
    pred, target = _check_point_inputs(pred, target)

    mse = torch.mean((pred - target) ** 2)
    data_range_t = torch.as_tensor(data_range, device=pred.device, dtype=pred.dtype)

    return 20.0 * torch.log10(data_range_t) - 10.0 * torch.log10(mse.clamp_min(eps))


def _gaussian_kernel_2d(
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - (window_size - 1) / 2.0

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()

    # [channels, 1, window_size, window_size] for grouped conv
    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Structural Similarity Index Measure, grayscale torch implementation.

    Inputs:
        pred:   [1, H, W]
        target: [1, H, W]

    Args:
        data_range:
            1.0 for [0, 1], 2.0 for [-1, 1], 255.0 for [0, 255].

    Returns:
        scalar tensor
    """
    pred, target = _check_point_inputs(pred, target)

    if window_size % 2 == 0:
        raise ValueError("`window_size` must be odd.")
    if pred.shape[-2] < window_size or pred.shape[-1] < window_size:
        raise ValueError(
            f"Input spatial size {tuple(pred.shape[-2:])} must be >= window_size={window_size}."
        )

    # Convert [1, H, W] to [B=1, C=1, H, W]
    x = pred.unsqueeze(0)
    y = target.unsqueeze(0)

    B, C, H, W = x.shape
    kernel = _gaussian_kernel_2d(
        channels=C,
        window_size=window_size,
        sigma=sigma,
        device=x.device,
        dtype=x.dtype,
    )

    pad = window_size // 2

    def filt(z: torch.Tensor) -> torch.Tensor:
        z = F.pad(z, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(z, kernel, groups=C)

    mu_x = filt(x)
    mu_y = filt(y)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = filt(x * x) - mu_x_sq
    sigma_y_sq = filt(y * y) - mu_y_sq
    sigma_xy = filt(x * y) - mu_xy

    data_range_t = torch.as_tensor(data_range, device=x.device, dtype=x.dtype)
    C1 = (k1 * data_range_t) ** 2
    C2 = (k2 * data_range_t) ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / denominator.clamp_min(eps)
    return ssim_map.mean()


def compute_all_metrics(
    posterior_samples: torch.Tensor,
    one_sample: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper.

    Args:
        posterior_samples: [N, H, W]
        one_sample:        [1, H, W], one posterior draw or point estimate
        target:            [1, H, W]
    """
    return {
        "CRPS": crps_ensemble(posterior_samples, target),
        "SSR": spread_skill_ratio(posterior_samples, target),
        "PSNR": psnr(one_sample, target, data_range=data_range),
        "SSIM": ssim(one_sample, target, data_range=data_range),
        "Relative_L2": relative_l2(one_sample, target),
    }
