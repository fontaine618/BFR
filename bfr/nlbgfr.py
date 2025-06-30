from torch import Tensor as T
import torch


class NLBGFR:
    """
    Nonlinear Bayesian Global Fréchet Regression (NLBGFR).
    """

    def __init__(
            self,
            K_train_train: T,  # N x N
            K_train_test: T,  # M x N
            rx_train: T = None,  # N
            rxy_train: T = None,  # N
            ry_train: T = None,  # N
            reg: float = 0.01
    ):
        N = K_train_train.size(0)
        M = K_train_test.size(0)
        # normalize weights
        rx_train = rx_train if rx_train is not None else torch.ones(N)
        rxy_train = rxy_train if rxy_train is not None else torch.ones(N)
        ry_train = ry_train if ry_train is not None else torch.ones(N)
        rx = rx_train / rx_train.sum()
        rxy = rxy_train / rxy_train.sum()
        ry = ry_train / ry_train.sum()
        # hat matrix
        H = torch.eye(N) - rx.unsqueeze(0)  # N x N
        # Gram matrix
        G = torch.einsum("ij, jk, lk -> il", H, K_train_train, H)  # N x N
        # covariance operator
        S = torch.einsum("i, ij->ij", rx, G) # N x N
        # average KME at training points
        Kbar = torch.einsum("ij, i->i", K_train_train, rx)  # N
        # doubly centered kernel matrix
        cK = K_train_test - Kbar.unsqueeze(0)  # M x N
        HcK = cK @ H.T  # M x N
        # weights
        GSinv = torch.linalg.solve(S.T + reg * torch.eye(N), G).T
        GinvGSinv = torch.linalg.solve(G + N * reg * torch.eye(N), GSinv)
        HcKtGinvGSinv = torch.einsum("mn, ni->mi", HcK, GinvGSinv)  # M x N
        self.weights = ry.unsqueeze(0) + HcKtGinvGSinv * rxy.unsqueeze(0)  # M x N

    def ese(
            self,
            u_train: T, # MxN
    ):
        return (self.weights * u_train).sum(1) # M


class NLGFR(NLBGFR):

    def __init__(
            self,
            K_train_train: T,  # N x N
            K_train_test: T,  # M x N
            reg: float = 0.01
    ):
        super().__init__(K_train_train, K_train_test, reg=reg)
