from torch import Tensor as T
import torch


class NLBR:
    """
    Nonlinear Bayesian Regression (NLBR).
    """

    def __init__(
            self,
            Kx_train_train: T,  # N x N
            intercept_variance_ratio: float = 0.,
            regression_variance_ratio: float = 1e-6,
            variance_explained: float = 0.99,
    ):
        if variance_explained < 0. or variance_explained > 1.:
            raise ValueError("variance_explained must be in [0, 1]")
        self.intercept_variance_ratio = None
        self.regression_variance_ratio = None
        self.variance_explained = variance_explained
        self.N = Kx_train_train.size(0)
        # declare things to be stored
        self._evals = None  # eigenvales for the centered Gram matrix (N)
        self._evecs = None  # eigenvectors for all matrices (NxD)
        self._evals_Vinv = None  # eigenvalues for the variance matrix inverse (N)
        self._evals_bProj = None  # eigenvalues for the matrix computing the coord. rep. of the prior (N)
        self._Kx_mean = None  # mean of the kernel matrix (N)
        # initialize
        self._initialize(Kx_train_train)
        self.update_variance_ratios(intercept_variance_ratio, regression_variance_ratio)

    def update_variance_ratios(
            self,
            intercept_variance_ratio: float = None,
            regression_variance_ratio: float = None
    ):
        """
        Update the variance ratios and reinitialize the model.
        """
        if intercept_variance_ratio is not None:
            if intercept_variance_ratio < 0.:
                raise ValueError("intercept_variance_ratio must be nonnegative")
            self.intercept_variance_ratio = intercept_variance_ratio
        if regression_variance_ratio is not None:
            if regression_variance_ratio < 1e-6:
                raise ValueError("regression_variance_ratio must be positive")
            self.regression_variance_ratio = regression_variance_ratio
            evals = self._evals
            r2 = regression_variance_ratio
            if r2 < float("inf"):
                self._evals_Vinv = evals * r2 / (self.N*evals.pow(2.)*r2 + 1.)
                self._evals_bProj = evals.pow(-2.) / (r2*self.N)
            else:
                self._evals_Vinv = 1. / (self.N*evals)
                self._evals_bProj = torch.zeros_like(evals)

    def _initialize(self, Kx_train_train:T):
        N = self.N
        # Compute the mean of the kernel matrix
        Kx_mean = Kx_train_train.mean(dim=0)
        # Compute centered Gram matrix
        H = torch.eye(N) - 1. / N
        G = torch.einsum("ij, jk, lk -> il", H, Kx_train_train, H)
        # Compute covariance operator
        S = G / N
        # Compute eigendecomposition of covariance operator
        evals, evecs = torch.linalg.eigh(S)
        # Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx].clamp(min=0.)  # ensure positive eigenvalues
        evecs = evecs[:, idx]
        # Keep only up to variance explained (also roots out the zero eigenvalues)
        cumvar = torch.cumsum(evals, dim=0)
        totalvar = cumvar[-1]
        n_components = torch.searchsorted(cumvar, self.variance_explained * totalvar).item() + 1
        evals = evals[:n_components]
        evecs = evecs[:, :n_components]
        # Store
        self._evals = evals
        self._evecs = evecs
        self._Kx_mean = Kx_mean

    def ese(
            self,
            Kx_test_train: T,  # M x N
            u_train: T,  # M x N
            u_prior: T = None,  # M x N (optional prior)
    ):
        # Compute centered feature matrix
        Kx_test_train = Kx_test_train - self._Kx_mean.unsqueeze(0)
        Kx_test_train = Kx_test_train - Kx_test_train.mean(dim=1, keepdim=True)
        # Center u_train
        ubar = u_train.mean(dim=1)
        u_train = u_train - ubar.unsqueeze(-1)
        # Compute the coordinate representation of the prior
        if u_prior is None:
            u_prior = torch.zeros_like(u_train)
        a = u_prior.mean(dim=1)
        u_prior = u_prior - a.unsqueeze(-1)
        b = torch.einsum(
            "d,ld,nd,mn->ml",
            self._evals_bProj, self._evecs, self._evecs, u_prior
        )
        diff = u_train + b
        # Compute intercept
        r2 = self.intercept_variance_ratio
        if r2 < float("inf"):
            intercept = (a + self.N*r2*ubar) / (1 + self.N*r2)
        else:
            intercept = ubar
        # Compute inner product term
        ip = torch.einsum(
            "ml,d,ld,nd,mn->m",
            Kx_test_train, self._evals_Vinv, self._evecs, self._evecs, diff
        )
        # Compute the final estimate
        ese = intercept + ip
        return ese

    def ese_pairwise(
            self,
            Kx_test_train: T,  # M x N
            u_train: T  # L x N
    ):
        # view Kx_test_train as ML x N and u_train as LM x N
        M, N = Kx_test_train.size()
        L = u_train.size(0)
        Kx_test_train_exp = Kx_test_train.unsqueeze(1).expand(M, L, N).reshape(M*L, N)
        u_train_exp = u_train.unsqueeze(0).expand(M, L, N).reshape(M*L, N)
        ese = self.ese(Kx_test_train_exp, u_train_exp)
        return ese.reshape(M, L)


