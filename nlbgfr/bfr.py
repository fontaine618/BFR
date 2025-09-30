from sympy.physics.units import velocity
from torch import Tensor as T
import torch
from .nlbr import NLBR
from typing import Callable


class BFR:
    """
    Bayesian Feature Regression (BFR) using gradient-based optimization.
    """

    def __init__(
            self,
            squared_distance: Callable,
            prior: Callable,
            Kx_train_train: T,  # N x N
            intercept_variance_ratio: float = 0.,
            regression_variance_ratio: float = 1e-6,
            variance_explained: float = 0.99,
    ):
        """

        :param squared_distance: A function that takes in (X: M x ..., Y: N x ...) and returns (M x N) tensor of squared distances
        :param prior: A function that takes in (Y: M x ...) and returns (M x N) tensor of prior expected squared distances
        :param Kx_train_train: A (N x N) tensor of the kernel matrix between training inputs
        :param intercept_variance_ratio:
        :param regression_variance_ratio:
        :param variance_explained:
        """
        self.squared_distance = squared_distance
        self.prior = prior
        self._nlbr = NLBR(
            Kx_train_train,
            intercept_variance_ratio=intercept_variance_ratio,
            regression_variance_ratio=regression_variance_ratio,
            variance_explained=variance_explained
        )

    def ese(
            self,
            Kx_test_train, u_train, Y
    ):
        """
        Compute the expected squared error (ESE) between the predicted and true outputs.

        :param Kx_test_train: M x N kernel matrix between test and train inputs
        :param u_train: M x N tensor of squared distances between test and train features
        :param Y: M x ... tensor of test feature locations
        :return: M x 1 tensor of expected squared errors for each test input
        """
        u_prior = self.prior(Y)
        return self._nlbr.ese(Kx_test_train, u_train, u_prior)

    def ppfm_apgd(
            self,
            Y_init: T,  # M x d
            Y_train: T,  # N x d
            Kx_test_train: T,  # M x N
            lr: float = 1e-2,
            max_iter: int = 1000,
            tol: float = 1e-6,
            momentum: float = 0.9,
            restart: bool = True,
            projection: Callable = None,
            transform: Callable = None,
            inv_transform: Callable = None
    ):
        """
        Optimize the feature locations using accelerated projected gradient descent with transformations.

        :param Y_init: M x ... tensor of initial feature locations
        :param Y_train: N x ... tensor of training feature locations
        :param Kx_test_train: M x N kernel matrix between test and train inputs
        :param lr: learning rate for the optimizer
        :param max_iter: maximum number of iterations
        :param tol: tolerance for convergence
        :param momentum: momentum parameter for acceleration
        :param restart: whether to restart momentum when the objective increases
        :param projection: A function that takes in (Y: M x ...) and returns (Y_proj: M x ...) projected onto the constraint set. Note that this is applied in the transformed space if transform is provided.
        :param transform: A function that transforms Y for optimization (e.g., to enforce constraints)
        :param inv_transform: A function that inverts back the transformation
        :return: M x ... tensor of posterior predictive Frechet means
        """
        if transform is None:
            transform = lambda Y: Y
        if inv_transform is None:
            inv_transform = lambda Y: Y
        if projection is None:
            projection = lambda Y: Y
        Z = torch.nn.Parameter(transform(Y_init).detach().clone())
        velocity = torch.zeros_like(Z)
        prev_loss = float("inf")
        for i in range(max_iter):
            if Z.grad is not None:
                Z.grad.zero_()
            Y = inv_transform(Z)
            u_train = self.squared_distance(Y, Y_train)
            ese = self.ese(Kx_test_train, u_train, Y)
            loss = ese.sum()
            loss.backward()
            with torch.no_grad():
                velocity.mul_(momentum).add_(Z.grad)
                Z.sub_(lr * velocity)
                Z.copy_(projection(Z))
            # if i % 10 == 0:
            #     print(f"Iter {i}: Loss = {loss.item():.6f}")
            if abs(prev_loss - loss.item()) < tol * abs(prev_loss):
                print(f"Iter {i}: Loss = {loss.item():.6f} (converged)")
                break
            if restart and loss.item() > prev_loss:
                # print(f"Iter {i}: Loss = {loss.item():.6f} (restart)")
                velocity.zero_()
            prev_loss = loss.item()
        return inv_transform(Z.detach())

