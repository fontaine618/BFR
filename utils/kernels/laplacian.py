import torch


def laplacian_kernel(
        D2: torch.Tensor,  # N x M
        length_scale: float = 1.0,
) -> torch.Tensor:  # N x M
    """
    Laplacian kernel function.

    :param D2: Squared distance matrix (N x M).
    :param length_scale: Length scale parameter.
    :return: Laplacian kernel matrix.
    """
    length_scale = torch.as_tensor(length_scale)
    return torch.exp(-D2.sqrt() / length_scale) / length_scale