from torch import Tensor as T
import torch


class BGFR:
    """
    Bayesian Global Fréchet Regression (BGFR).
    """

    def __init__(
            self,
            X_train: T,  # N x P
            X_test: T,  # M x P
            rx_train: T = None,  # ... x N
            rxy_train: T = None,  # ... x N
            # ry_train: T = None,  # N
            reg: float = 1e-5,  # regularization parameter
    ):
        """... can be nothing, 1 or M. M is used for localized prior."""
        P = X_train.size(1)
        N = X_train.size(0)
        M = X_test.size(0)
        # normalize weights
        rx_train = rx_train if rx_train is not None else torch.ones(N)
        rxy_train = rxy_train if rxy_train is not None else torch.ones(N)
        # ry_train = ry_train if ry_train is not None else torch.ones(N)
        rx = rx_train / rx_train.sum()
        rxy = rxy_train / rxy_train.sum()
        # ry = ry_train / ry_train.sum()

        # uniformize number of dimensions of weights to be broadcastable
        ndims = max(rx.dim(), rxy.dim()) - 1
        if rx.dim() < ndims + 1:
            for _ in range(ndims + 1 - rx.dim()):
                rx = rx.unsqueeze(0)
        if rxy.dim() < ndims + 1:
            for _ in range(ndims + 1 - rxy.dim()):
                rxy = rxy.unsqueeze(0)
        # if ry.dim() < ndims + 1:
        #     for _ in range(ndims + 1 - ry.dim()):
        #         ry = ry.unsqueeze(-1)


        # compute weighted moments
        mean = torch.einsum("np,...n->...p", X_train, rx)  # ... x P
        for _ in range(ndims):
            X_train = X_train.unsqueeze(0)  # ... x N x P
            X_test = X_test.unsqueeze(0)  # ... x M x P
        X_train_centered = X_train - mean.unsqueeze(-2)
        X_test_centered = X_test - mean.unsqueeze(-2)
        cov = torch.einsum(
            "...np,...nq,...n->...pq",
            X_train_centered,
            X_train_centered,
            rx,
        )  # ... x P x P
        regmat = reg * torch.eye(P, device=X_train.device) # ... x P x P
        for _ in range(ndims):
            regmat = regmat.unsqueeze(0)
        cov = cov + regmat

        invcov = torch.inverse(cov)  # ... x P x P
        # compute gfr weights
        gfr_weights = torch.einsum(
            "...mp, ...pq, ...nq -> ...mn",
            X_test_centered,
            invcov,
            X_train_centered,
        )  # ... x M x N
        gfr_weights = (1. + gfr_weights) * rxy.unsqueeze(-2)
        # gfr_weights = ry.unsqueeze(0) + gfr_weights * rxy.unsqueeze(0)
        self.gfr_weights = gfr_weights  # ... x M x N

    def ese(
            self,
            u_train: T,  # M x N
    ):
        return (self.gfr_weights * u_train).sum(-1)


class GFR(BGFR):

    def __init__(
            self,
            X_train: T,  # N x P
            X_test: T,  # M x P
    ):
        super().__init__(X_train, X_test)