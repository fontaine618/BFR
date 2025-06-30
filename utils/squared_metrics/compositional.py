import torch
import math
from typing import Callable, Optional
from . import pairwise
from torch import Tensor as T


# decorator that normalizes the input before caling the functions
def normalize(sq_dist_func: Callable[[T, T, Optional], T]) -> Callable[[T, T, Optional], T]:
    def normalized(X: T, Y: T, **kwargs) -> T:
        X = X / X.sum(dim=-1, keepdim=True)  # normalize X
        Y = Y / Y.sum(dim=-1, keepdim=True)  # normalize Y
        return sq_dist_func(X, Y, **kwargs)
    return normalized

def sqrt(sq_dist_func: Callable[[T, T, Optional], T]) -> Callable[[T, T, Optional], T]:
    def sqrted(X: T, Y: T, **kwargs) -> T:
        X = X.sqrt()
        Y = Y.sqrt()
        return sq_dist_func(X, Y, **kwargs)
    return sqrted

@pairwise
@normalize
def squared_bray_curtis_dissimilarity(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    diff = (X - Y).abs()  # N x M x P
    return (diff.sum(dim=-1) / (X + Y).abs().sum(dim=-1)).pow(2)  # N x M


@pairwise
@normalize
def squared_canberra_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    num = (X - Y).abs()  # N x M x P
    denum = X.abs() + Y.abs()  # N x M x P
    nz = denum.gt(0.).sum(dim=-1)  # N x M
    # when denum is zero, it must be that 0=0 so we can ignore those terms
    denum = denum.clamp(min=1e-8)  # avoid division by zero
    return (num / denum).sum(dim=-1).div(nz).pow(2.)  # N x M

@sqrt
@pairwise
@normalize
def squared_hellinger_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    diff = (X - Y).pow(2)  # N x M x P
    return diff.mean(dim=-1)  # N x M


@sqrt
@pairwise
@normalize
def squared_cosine_sqrt_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    ip = (X * Y).sum(dim=-1)  # N x M
    return (1 - ip).pow(2)  # N x M


@pairwise
@normalize
def squared_cosine_norm_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    ip = (X / X.norm(dim=-1, keepdim=True) * Y / Y.norm(dim=-1, keepdim=True)).sum(dim=-1)  # N x M
    return (1 - ip).pow(2)  # N x M


@pairwise
@normalize
def squared_spherical_sqrt_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    X = X.sqrt()  # N x 1 x P
    Y = Y.sqrt()  # 1 x M x P
    ip = (X * Y).sum(dim=-1)  # N x M
    ip = ip.clamp(min=-0.999, max=0.999)  # avoid boundary issues
    return (torch.acos(ip)).pow(2)  # N x M


@pairwise
@normalize
def squared_spherical_norm_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    ip = (X / X.norm(dim=-1, keepdim=True) * Y / Y.norm(dim=-1, keepdim=True)).sum(dim=-1)  # N x M
    ip = ip.clamp(min=-0.999, max=0.999)  # avoid boundary issues
    return (torch.acos(ip)).pow(2)  # N x M


@pairwise
@normalize
def squared_aitchison_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
        offset: float = 0.5,  # small constant to avoid zero
) -> torch.Tensor:  # N x M
    X = X + offset  # N x 1 x P
    Y = Y + offset  # 1 x M x P
    # CLR
    X_clr = X.log()
    X_clr = X_clr - X_clr.mean(dim=-1, keepdim=True)  # N x 1 x P
    Y_clr = Y.log()
    Y_clr = Y_clr - Y_clr.mean(dim=-1, keepdim=True)  # 1 x M x P
    diff = (X_clr - Y_clr).pow(2)  # N x M x P
    return diff.sum(dim=-1)  # N x M