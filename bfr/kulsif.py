import torch


def kulsif(
        K_den_den: torch.Tensor, # N x N
        K_den_num: torch.Tensor, # ... x N x M
        reg: float = 0.,
        truncate: bool = True,
        reg_center: str = "one"
) -> torch.Tensor:
    """
    Kernel unconstrained Least Squares Importance Fitting (KuLSIF) by Kanamori et al. (2012)

    :param K_den_den: Kernel evaluation across denominator points (N x N)
    :param K_den_num: Kernel evaluation between denominator and numerator points (... x N x M)
    :param reg: Regularization parameter
    :param truncate: Whether to truncate the ratio to be non-negative
    :param reg_center: Centering of the regularization term, either "one" or "zero"
    :return: An N-dim tensor of estimated ratios at the denominator points
    """
    N = K_den_den.size(0)
    M = K_den_num.size(-1)
    # prepare quadratic form
    Q = K_den_den / N + reg * torch.eye(N, device=K_den_den.device)  # N x N
    K_den_num_sum = K_den_num.sum(-1) / (M * reg)  # ... x N
    if reg_center == "one":
        b = K_den_num_sum / N + torch.ones(N) / N  # ... x N
    elif reg_center == "zero":
        b = K_den_num_sum / N  # ... x N
    else:
        raise ValueError("reg_center must be 'one' or 'zero'")

    # unsqueeze Q to match dimensions if necessary
    ndims = K_den_num.dim() - 2  # number of dimensions before N
    for _ in range(ndims):
        Q = Q.unsqueeze(0)

    # Solve the linear system to get weights
    a = torch.matmul(torch.linalg.inv(Q), -b.unsqueeze(-1)).squeeze(-1) # ... x N
    # a = torch.linalg.solve(Q, -b)  # ... x N

    # Compute estimated ratio
    ratio = torch.einsum("...n, nm -> ...m", a, K_den_den) + K_den_num_sum
    if reg_center == "one":
        ratio = 1. + ratio
    if truncate:
        ratio = torch.clamp(ratio, min=0.)
    return ratio  # ... x N