import torch
from typing import Dict, Literal, Optional, Tuple


# ============================================================
# Basic utilities
# ============================================================

def _as_2d_float(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Require shape [N, n_x].
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    # Use float32 by default for numerical stability.
    x = x.float()

    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [N, n_x], got {tuple(x.shape)}.")
    if x.shape[0] < 2:
        raise ValueError(f"{name} must contain at least 2 samples, got N={x.shape[0]}.")

    return x


def _empirical_mean_cov(
    X: torch.Tensor,
    unbiased: bool = True,
    shrinkage: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate empirical mean and covariance.

    X: [N, D]
    """
    N, D = X.shape
    mu = X.mean(dim=0)
    Xc = X - mu

    denom = N - 1 if unbiased else N
    cov = (Xc.T @ Xc) / denom

    # Optional covariance shrinkage:
    # Sigma <- (1 - alpha) Sigma + alpha * avg_var * I
    # Useful when N is not much larger than D.
    if shrinkage > 0.0:
        avg_var = torch.diagonal(cov).mean().clamp_min(0.0)
        I = torch.eye(D, device=X.device, dtype=X.dtype)
        cov = (1.0 - shrinkage) * cov + shrinkage * avg_var * I

    return mu, cov


def _safe_cholesky(
    A: torch.Tensor,
    jitter: float = 1e-6,
    max_tries: int = 8,
) -> Tuple[torch.Tensor, float]:
    """
    Cholesky factorization with adaptive jitter.

    Returns:
        L such that A + jitter * I = L L^T
    """
    D = A.shape[-1]
    I = torch.eye(D, device=A.device, dtype=A.dtype)

    diag_scale = torch.diagonal(A).mean().abs().clamp_min(1.0)
    current_jitter = float(jitter) * float(diag_scale.detach().cpu())

    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(A + current_jitter * I)
        if int(info.item()) == 0:
            return L, current_jitter
        current_jitter *= 10.0

    raise RuntimeError(
        f"Cholesky failed even after jitter={current_jitter:.3e}. "
        "Try larger jitter or positive shrinkage, e.g. shrinkage=1e-3."
    )


# ============================================================
# 1. KL divergence between empirical Gaussian posteriors
# ============================================================

def kl_gaussian_from_samples_full(
    X_model: torch.Tensor,
    X_true: torch.Tensor,
    *,
    direction: Literal["model||true", "true||model"] = "model||true",
    shrinkage: float = 1e-4,
    jitter: float = 1e-6,
    unbiased_cov: bool = True,
) -> torch.Tensor:
    """
    KL divergence between two empirical Gaussian approximations.

    Inputs:
        X_model: [N, n_x] generated/model posterior samples
        X_true:  [M, n_x] ground-truth GP posterior samples

    Default:
        D_KL( model || true )

    Formula:
        KL(p || q) = 0.5 * [
            log |Sigma_q| / |Sigma_p|
            - d
            + tr(Sigma_q^{-1} Sigma_p)
            + (mu_q - mu_p)^T Sigma_q^{-1} (mu_q - mu_p)
        ]

    Returns:
        scalar tensor
    """
    X_model = _as_2d_float(X_model, "X_model")
    X_true = _as_2d_float(X_true, "X_true")

    if X_model.shape[1] != X_true.shape[1]:
        raise ValueError(
            f"n_x must match, got {X_model.shape[1]} and {X_true.shape[1]}."
        )

    mu_m, cov_m = _empirical_mean_cov(
        X_model,
        unbiased=unbiased_cov,
        shrinkage=shrinkage,
    )
    mu_t, cov_t = _empirical_mean_cov(
        X_true,
        unbiased=unbiased_cov,
        shrinkage=shrinkage,
    )

    if direction == "model||true":
        mu_p, cov_p = mu_m, cov_m
        mu_q, cov_q = mu_t, cov_t
    elif direction == "true||model":
        mu_p, cov_p = mu_t, cov_t
        mu_q, cov_q = mu_m, cov_m
    else:
        raise ValueError("direction must be 'model||true' or 'true||model'.")

    D = X_model.shape[1]

    Lq, _ = _safe_cholesky(cov_q, jitter=jitter)
    Lp, _ = _safe_cholesky(cov_p, jitter=jitter)

    logdet_q = 2.0 * torch.log(torch.diagonal(Lq)).sum()
    logdet_p = 2.0 * torch.log(torch.diagonal(Lp)).sum()

    # tr(Sigma_q^{-1} Sigma_p)
    q_inv_p = torch.cholesky_solve(cov_p, Lq)
    trace_term = torch.trace(q_inv_p)

    # (mu_q - mu_p)^T Sigma_q^{-1} (mu_q - mu_p)
    diff = (mu_q - mu_p).unsqueeze(-1)
    q_inv_diff = torch.cholesky_solve(diff, Lq)
    quad_term = (diff.T @ q_inv_diff).squeeze()

    kl = 0.5 * (logdet_q - logdet_p - D + trace_term + quad_term)

    # Small negative values can occur from numerical error.
    return torch.clamp(kl, min=0.0)


# ============================================================
# 2. Sliced Wasserstein Distance, pure torch
# ============================================================

def sliced_wasserstein_torch(
    X_model: torch.Tensor,
    X_true: torch.Tensor,
    *,
    n_proj: int = 256,
    p: int = 2,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sliced Wasserstein distance between two empirical distributions.

    Inputs:
        X_model: [N, n_x]
        X_true:  [M, n_x]

    For each random direction theta:
        project samples to 1D,
        sort projected values,
        compute 1D Wasserstein distance.

    Returns:
        scalar tensor
    """
    X_model = _as_2d_float(X_model, "X_model")
    X_true = _as_2d_float(X_true, "X_true")

    if X_model.shape[1] != X_true.shape[1]:
        raise ValueError(
            f"n_x must match, got {X_model.shape[1]} and {X_true.shape[1]}."
        )

    device = X_model.device
    dtype = X_model.dtype
    D = X_model.shape[1]

    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        theta = torch.randn(D, n_proj, device=device, dtype=dtype, generator=gen)
    else:
        theta = torch.randn(D, n_proj, device=device, dtype=dtype)

    theta = theta / torch.linalg.vector_norm(theta, dim=0, keepdim=True).clamp_min(1e-12)

    Xp = X_model @ theta
    Yp = X_true @ theta

    Xs = torch.sort(Xp, dim=0).values
    Ys = torch.sort(Yp, dim=0).values

    # If N != M, compare empirical quantiles on a common grid.
    if Xs.shape[0] != Ys.shape[0]:
        K = min(Xs.shape[0], Ys.shape[0])
        q = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype)

        def interp_sorted(vals: torch.Tensor, qgrid: torch.Tensor) -> torch.Tensor:
            n = vals.shape[0]
            pos = qgrid * (n - 1)
            lo = torch.floor(pos).long()
            hi = torch.ceil(pos).long()
            w = (pos - lo.to(dtype)).view(-1, 1)
            return (1.0 - w) * vals[lo] + w * vals[hi]

        Xs = interp_sorted(Xs, q)
        Ys = interp_sorted(Ys, q)

    wd_per_proj_p = torch.mean(torch.abs(Xs - Ys) ** p, dim=0)
    swd = torch.mean(wd_per_proj_p) ** (1.0 / p)

    return swd


def sliced_wasserstein_stable_torch(
    X_model: torch.Tensor,
    X_true: torch.Tensor,
    *,
    n_runs: int = 10,
    n_proj: int = 256,
    p: int = 2,
    seed: Optional[int] = 1234,
) -> torch.Tensor:
    """
    Average SWD over multiple random projection draws.
    Similar to your swd_stable function.
    """
    vals = []

    for r in range(n_runs):
        run_seed = None if seed is None else seed + r
        vals.append(
            sliced_wasserstein_torch(
                X_model,
                X_true,
                n_proj=n_proj,
                p=p,
                seed=run_seed,
            )
        )

    return torch.stack(vals).mean()


# ============================================================
# 3. RBF MMD, pure torch
# ============================================================

def mmd_rbf_torch(
    X_model: torch.Tensor,
    X_true: torch.Tensor,
    *,
    gamma: Optional[float] = 1.0,
    unbiased: bool = True,
    return_sqrt: bool = True,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    RBF-kernel MMD between two empirical sample sets.

    Inputs:
        X_model: [N, n_x]
        X_true:  [M, n_x]

    Kernel:
        k(x, y) = exp(-gamma * ||x - y||^2 / D)

    This matches the scaling in your code:
        exp(-gamma * dist_xy / D)

    Args:
        gamma:
            RBF bandwidth parameter. gamma=1.0 matches your provided code.
        unbiased:
            If True, removes diagonal terms from Kxx and Kyy.
        return_sqrt:
            If True, returns MMD.
            If False, returns MMD^2.
        chunk_size:
            Use this if N is large to reduce memory.

    Returns:
        scalar tensor
    """
    X_model = _as_2d_float(X_model, "X_model")
    X_true = _as_2d_float(X_true, "X_true")

    if X_model.shape[1] != X_true.shape[1]:
        raise ValueError(
            f"n_x must match, got {X_model.shape[1]} and {X_true.shape[1]}."
        )

    X = X_model
    Y = X_true

    m, D = X.shape
    n = Y.shape[0]

    gamma = 1.0 if gamma is None else float(gamma)

    def rbf_sum(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if chunk_size is None:
            dist2 = torch.cdist(A, B, p=2) ** 2
            return torch.exp(-gamma * dist2 / D).sum()

        total = torch.zeros((), device=A.device, dtype=A.dtype)
        for i in range(0, A.shape[0], chunk_size):
            Ai = A[i:i + chunk_size]
            dist2 = torch.cdist(Ai, B, p=2) ** 2
            total = total + torch.exp(-gamma * dist2 / D).sum()

        return total

    if unbiased:
        if m < 2 or n < 2:
            raise ValueError("Unbiased MMD requires at least 2 samples in each set.")

        # subtract diagonal k(x_i, x_i)=1
        Kxx_sum = rbf_sum(X, X) - m
        Kyy_sum = rbf_sum(Y, Y) - n
        Kxy_sum = rbf_sum(X, Y)

        mmd2 = (
            Kxx_sum / (m * (m - 1))
            + Kyy_sum / (n * (n - 1))
            - 2.0 * Kxy_sum / (m * n)
        )
    else:
        Kxx_sum = rbf_sum(X, X)
        Kyy_sum = rbf_sum(Y, Y)
        Kxy_sum = rbf_sum(X, Y)

        mmd2 = (
            Kxx_sum / (m * m)
            + Kyy_sum / (n * n)
            - 2.0 * Kxy_sum / (m * n)
        )

    mmd2 = torch.clamp(mmd2, min=0.0)

    if return_sqrt:
        return torch.sqrt(mmd2)
    else:
        return mmd2


# ============================================================
# 4. Convenience wrapper
# ============================================================

def gp_posterior_sample_metrics(
    X_model: torch.Tensor,
    X_true: torch.Tensor,
    *,
    kl_direction: Literal["model||true", "true||model", "sym"] = "model||true",
    kl_shrinkage: float = 1e-4,
    kl_jitter: float = 1e-6,
    swd_n_runs: int = 10,
    swd_n_proj: int = 256,
    mmd_gamma: Optional[float] = 1.0,
    seed: Optional[int] = 1234,
) -> Dict[str, torch.Tensor]:
    """
    Compute KL, SWD, and MMD for one 1D GP posterior comparison.

    Inputs:
        X_model: [N, n_x]
            Posterior samples generated by your model.

        X_true: [N, n_x]
            Ground-truth GP posterior samples.

    Returns:
        {
            "KL": scalar tensor,
            "SWD": scalar tensor,
            "MMD": scalar tensor,
        }
    """
    X_model = _as_2d_float(X_model, "X_model")
    X_true = _as_2d_float(X_true, "X_true")

    if kl_direction == "sym":
        kl_mt = kl_gaussian_from_samples_full(
            X_model,
            X_true,
            direction="model||true",
            shrinkage=kl_shrinkage,
            jitter=kl_jitter,
        )
        kl_tm = kl_gaussian_from_samples_full(
            X_model,
            X_true,
            direction="true||model",
            shrinkage=kl_shrinkage,
            jitter=kl_jitter,
        )
        kl = 0.5 * (kl_mt + kl_tm)
    else:
        kl = kl_gaussian_from_samples_full(
            X_model,
            X_true,
            direction=kl_direction,
            shrinkage=kl_shrinkage,
            jitter=kl_jitter,
        )

    swd = sliced_wasserstein_stable_torch(
        X_model,
        X_true,
        n_runs=swd_n_runs,
        n_proj=swd_n_proj,
        p=2,
        seed=seed,
    )

    mmd = mmd_rbf_torch(
        X_model,
        X_true,
        gamma=mmd_gamma,
        unbiased=True,
        return_sqrt=True,
    )

    return {
        "KL": kl,
        "SWD": swd,
        "MMD": mmd,
    }