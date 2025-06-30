from torch import Tensor as T
from typing import Callable, Optional

# decorator that unsqueeze the input for pairwise evaluation
def pairwise(sq_dist_func: Callable[[T, T, Optional], T]) -> Callable[[T, T, Optional], T]:
    def pairwised(X: T, Y: T, **kwargs) -> T:
        X = X.unsqueeze(1)  # N x 1 x P
        Y = Y.unsqueeze(0)  # 1 x M x P
        return sq_dist_func(X, Y, **kwargs)
    return pairwised
