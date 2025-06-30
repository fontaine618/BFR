import torch


def squared_exponential_kernel(
        D2: torch.Tensor,  # N x M
        length_scale: float = 1.0,
) -> torch.Tensor:  # N x M
    length_scale = torch.as_tensor(length_scale)
    return torch.exp(-D2 / (2 * length_scale ** 2)) / (2 * torch.pi * length_scale.pow(2)).pow(1 / 2)