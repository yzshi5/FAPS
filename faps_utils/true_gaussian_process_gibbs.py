import torch
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter
from faps_utils.util import make_grid
import numpy as np

## Load the model 


def gibbs_kernel_cov(grids, l0=0.05, l1=0.25, sigma=1.0, eps=1e-12):
    """Gibbs kernel covariance with input-dependent length-scale.

    l(x) = l0 + l1 * x, x in [0, 1]
    k(x, x') = sigma^2 * sqrt(2 l(x) l(x') / (l(x)^2 + l(x')^2))
               * exp(-(x - x')^2 / (l(x)^2 + l(x')^2))
    """
    grids = np.asarray(grids)
    x = grids[:, 0].astype(np.float64)
    lx = np.maximum(l0 + l1 * x, eps)
    l2_sum = lx[:, None] ** 2 + lx[None, :] ** 2
    prefactor = np.sqrt(np.maximum(2.0 * lx[:, None] * lx[None, :] / np.maximum(l2_sum, eps), eps))
    d2 = (x[:, None] - x[None, :]) ** 2
    return (sigma**2) * prefactor * np.exp(-d2 / np.maximum(l2_sum, eps))


class GibbsKernel(Kernel):
    """Non‑stationary SE with input‑dependent length‑scale:
       l(x) = l0 + l1 * x   (x in [0,1])
       k(x,x') = σ² √( 2 l(x) l(x') / (l(x)^2 + l(x')^2) )
                 * exp(‑ (x‑x')² / (l(x)^2 + l(x')^2))
    """

    def __init__(self, sigma=1.0, l0=0.05, l1=0.25):
        self.sigma = float(sigma)
        self.l0 = float(l0)
        self.l1 = float(l1)

    # required by scikit‑learn
    @property
    def hyperparameter_sigma(self):
        return Hyperparameter("sigma", "numeric", (1e-5, 1e5))

    def lengthscale(self, x):
        return self.l0 + self.l1 * x

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        lx = self.lengthscale(X)
        ly = self.lengthscale(Y)

        # pairwise squared distance matrix
        dists2 = (X[:, None, 0] - Y[None, :, 0]) ** 2
        ls_sum = lx[:, None, 0] ** 2 + ly[None, :, 0] ** 2

        prefactor = np.sqrt(2 * lx[:, None, 0] * ly[None, :, 0] / ls_sum)
        K = (self.sigma**2) * prefactor * np.exp(-dists2 / ls_sum)

        if eval_gradient:
            raise ValueError("Gradient not implemented for Gibbs kernel.")
        return K

    def diag(self, X):
        return np.full(X.shape[0], self.sigma ** 2)

    def is_stationary(self):
        return False



def _build_mvn_from_cov(base_mu, base_cov, device, max_tries=3):
    """
    Build a stable MultivariateNormal by validating Cholesky and
    increasing diagonal jitter when needed.
    """
    base_mu = base_mu.to(device=device, dtype=torch.float32)
    cov = base_cov.to(device=device, dtype=torch.float64)
    eye = torch.eye(cov.shape[0], device=device, dtype=torch.float64)

    jitter = 0.0
    for _ in range(max_tries):
        chol, info = torch.linalg.cholesky_ex(cov + jitter * eye)
        if torch.all(info == 0):
            chol = torch.tril(chol).to(torch.float32)
            return torch.distributions.MultivariateNormal(base_mu, scale_tril=chol)
        jitter = 1e-6 if jitter == 0.0 else jitter * 10.0

    raise RuntimeError(
        "Failed to build a valid GP covariance Cholesky factor after jitter escalation."
    )


class true_GPPrior_Gibbs(torch.distributions.distribution.Distribution):
    
    """ Wrapper around some torch utilities that makes prior sampling easy.
    """

    def __init__(
        self,
        l0=0.05,
        l1=0.25,
        sigma=1.0,
        mean=None,
        lengthscale=None,
        var=None,
        nu=0.5,
        device='cpu',
        dims=None,
    ):
        """
        kernel/mean/lengthscale/var: parameters of kernel
        you should choose right parameter to avoid numerical instability of the cov matrix
        """
        # Backward compatibility: if var is supplied, use sigma = sqrt(var).
        if var is not None:
            sigma = float(np.sqrt(var))
        self.l0 = float(l0)
        self.l1 = float(l1)
        self.sigma = float(sigma)

        # dims should be 1D [n_x] or 2D [n_x, n_x]
        n_points = np.prod(dims)
        grids = make_grid(dims)
        gibbs_cov = gibbs_kernel_cov(grids, l0=self.l0, l1=self.l1, sigma=self.sigma)

        self.dims = dims

        base_mu = torch.zeros(n_points).float()
        # add a small base jitter; helper can increase if needed
        base_cov = torch.tensor(gibbs_cov).float() + 1e-6 * torch.eye(gibbs_cov.shape[0])
        base_cov = base_cov.to(torch.float64) #can help improve numerical stability
        # be careful of numerical instability when calculating on GPU
        try:
            self.base_dist = _build_mvn_from_cov(base_mu, base_cov, device)
        except RuntimeError:
            self.base_dist = _build_mvn_from_cov(base_mu, base_cov, "cpu")

        self.device = device

    def check_input(self, x, dims=None):
        assert x.ndim == 2, f'Input {x.shape} should have shape (n_points, dim)'
        if dims:
            assert x.shape[1] == len(dims), f'Input {x.shape} should have shape (n_points, dim)'

    def new_dist(self, dims):
        """ Creates a Normal distribution at the points in x.
        x: locations to query at, a flattened grid; tensor (n_points, dim)

        returns: a gpytorch distribution corresponding to a Gaussian at x
        """
        n_points = np.prod(dims)
        grids = make_grid(dims)
        gibbs_cov = gibbs_kernel_cov(grids, l0=self.l0, l1=self.l1, sigma=self.sigma)

        base_mu = torch.zeros(n_points).float()
        base_cov = torch.tensor(gibbs_cov).float() + 1e-6 * torch.eye(gibbs_cov.shape[0])
        base_cov = base_cov.to(torch.float64)        

        try:
            base_dist = _build_mvn_from_cov(base_mu, base_cov, self.device)
        except RuntimeError:
            base_dist = _build_mvn_from_cov(base_mu, base_cov, "cpu")
            
        return base_dist
    
    def sample(self, dims, n_samples=1, n_channels=1):
        """ Draws samples from the GP prior.
        dims: list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
        n_samples: number of samples to draw
        n_channels: number of independent channels to draw samples for

        returns: samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        
        #x = x.to(self.device)
        if dims == self.dims:
            distr = self.base_dist
        else:
            distr = self.new_dist(dims)
        samples = distr.sample(sample_shape = torch.Size([n_samples * n_channels, ]))
        samples = samples.reshape(n_samples, n_channels, *dims)
        
        return samples
        
    
    def sample_from_prior(self, dims, n_samples=1, n_channels=1):
        """
        fixed prior
        """
        if dims == self.dims:
            distr = self.base_dist
        else:
            distr = self.new_dist(dims)
        samples = distr.sample(sample_shape = torch.Size([n_samples * n_channels, ]))
        samples = samples.reshape(n_samples, n_channels, *dims)
        
        return samples           
    
    def sample_train_data(self, dims, n_samples=1, n_channels=1, nbatch=1000):
        """
        calculation in cuda, but saved in cpu.
        iteratively 
        """
        samples_all = []

        sampled_num = 0
        nbatch = np.min([n_samples, nbatch])
              
        while sampled_num < n_samples:
            temp_sample = self.sample_from_prior(dims=dims, n_samples=nbatch, n_channels=n_channels).cpu()
            sampled_num += len(temp_sample)
            samples_all.append(temp_sample)
                
        samples_all = torch.vstack(samples_all)[:n_samples]
        return samples_all
        
    def prior_likelihood(self, x):
        """
        calculate the likelihood of the input.
        x shape:[n_batch, -1] 
        # only used in jacobian, already to(device), n_channels must be 1
        """
        x = torch.flatten(x, start_dim=1)
        logp = self.base_dist.log_prob(x)
        return logp
        
    ## for codomain data
    def prior_likelihood_codomain(self, x, n_channels=1):
        """
        calculate the likelihood of the input.
        x shape:[n_batch, -1] 
        # only used in jacobian, already to(device), n_channels must be 1
        """
        x = x.reshape(x.shape[0], n_channels, -1)
                                                         
        for i in range(n_channels):
            if i == 0:
                logp = self.base_dist.log_prob(torch.flatten(x[:,0],start_dim=1))
            else:
                logp += self.base_dist.log_prob(torch.flatten(x[:,i], start_dim=1))
        
        return logp    