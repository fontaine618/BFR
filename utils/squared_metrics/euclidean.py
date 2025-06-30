import torch


def squared_L2_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    return (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(-1)  # N x M