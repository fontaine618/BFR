import torch


def matern_kernel(
        D2: torch.Tensor,  # N x M
        length_scale: float = 1.0,
        nu: float = 1.5,
) -> torch.Tensor:  # N x M
    """
    Matern kernel function.

    :param D2: Squared distance matrix (N x M).
    :param length_scale: Length scale parameter.
    :param nu: Smoothness parameter.
    :return: Matern kernel matrix.
    """
    length_scale = torch.as_tensor(length_scale)
    sqrt_2nu = torch.as_tensor(2 * nu).sqrt()
    D = sqrt_2nu *  D2.sqrt() / length_scale
    K = torch.exp(-D)
    if nu == 0.5:
        pass
    elif nu == 1.5:
        K = K * (1 + D)
    elif nu == 2.5:
        K = K * (1 + D + D.pow(2) / 3)
    else:
        raise ValueError("Unsupported value for nu. Supported values are: 0.5, 1.5, and 2.5.")
    return K / length_scale