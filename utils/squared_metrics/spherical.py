import torch


def squared_geodesic_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    ip = (X.unsqueeze(1) * Y.unsqueeze(0)).sum(-1) # N x M
    # to allow differentiation, we need to avoid the boundary
    ip = ip.clamp(min=-0.99999, max=0.99999) # N x M
    return torch.acos(ip).pow(2.0) # N x M


def squared_chordal_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    ip = (X.unsqueeze(1) * Y.unsqueeze(0)).sum(-1) # N x M
    return 4.*(1 - ip).pow(2.0) # N x M