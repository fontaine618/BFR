import torch


def squared_hamming_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    """
    Computes the squared Hamming distance between two binary matrices.
    The Hamming distance is defined as the number of differing elements.
    """
    return (X.unsqueeze(1) != Y.unsqueeze(0)).float().sum(dim=-1)  # N x M


def squared_jaccard_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    """
    Computes the squared Jaccard distance between two binary matrices.
    The Jaccard distance is defined as 1 minus the Jaccard index.
    """
    intersection = (X.unsqueeze(1) & Y.unsqueeze(0)).float().sum(dim=-1)  # N x M
    union = (X.unsqueeze(1) | Y.unsqueeze(0)).float().sum(dim=-1)  # N x M
    return (1 - intersection / union).pow(2)  # N x M


def squared_dice_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    """
    Computes the squared Dice distance between two binary matrices.
    The Dice distance is defined as 1 minus the Dice coefficient.
    """
    intersection = (X.unsqueeze(1) & Y.unsqueeze(0)).float().sum(dim=-1)  # N x M
    return (1 - (2 * intersection) / (X.sum(dim=-1, keepdim=True) + Y.sum(dim=-1, keepdim=True))).pow(2)  # N x M


def squared_overlap_distance(
        X: torch.Tensor,  # N x P
        Y: torch.Tensor,  # M x P
) -> torch.Tensor:  # N x M
    """
    Computes the squared overlap distance between two binary matrices.
    The overlap distance is defined as 1 minus the overlap coefficient.
    """
    intersection = (X.unsqueeze(1) & Y.unsqueeze(0)).float().sum(dim=-1)  # N x M
    return (1 - intersection / X.sum(dim=-1, keepdim=True)).pow(2)  # N x M