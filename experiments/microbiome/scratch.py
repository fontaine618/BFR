import torch
import os
import pandas as pd
import itertools
from nlbgfr.bfr import BFR
from nlbgfr.nlbr import NLBR
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.squared_metrics.spherical import squared_geodesic_distance, squared_chordal_distance
from utils.squared_metrics.euclidean import squared_L2_distance
import utils.squared_metrics.compositional as c
from experiments.microbiome.data_loader import T2D

sq_dist = c.squared_hellinger_distance

prior_study_name = "MetaCardis_2020_a"
train_eval_study_name = "QinJ_2012"

prior_study = T2D()
prior_study.filter_rare_taxa()
prior_study.filter_study(prior_study_name)
Yprior = torch.tensor(prior_study.relative_abundance.values, dtype=torch.float32)
Xprior = prior_study.features
Np = Yprior.size(0)

train = T2D()
train.filter_rare_taxa()
train.filter_study(train_eval_study_name)
train.assign_train_test(random_state=0)
train.filter_set("train")
Ytrain = torch.tensor(train.relative_abundance.values, dtype=torch.float32)
Xtrain = train.features
Nt = Ytrain.size(0)

eval = T2D()
eval.filter_rare_taxa()
eval.filter_study(train_eval_study_name)
eval.assign_train_test(random_state=0)
eval.filter_set("test")
Yeval = torch.tensor(eval.relative_abundance.values, dtype=torch.float32)
Xeval = eval.features
Ne = Yeval.size(0)

# define kernel function
def kernel(X1: pd.DataFrame, X2: pd.DataFrame) -> torch.Tensor:
    N1 = X1.shape[0]
    N2 = X2.shape[0]
    K = torch.zeros(N1, N2)
    # T2D
    K += X1['T2D'].values.reshape(-1, 1) == X2['T2D'].values.reshape(1, -1)
    # antibiotics
    K += X1['antibiotics'].values.reshape(-1, 1) == X2['antibiotics'].values.reshape(1, -1)
    # female
    K += X1['female'].values.reshape(-1, 1) == X2['female'].values.reshape(1, -1)
    # age
    K += (torch.tensor(X1['age'].values, dtype=torch.float32).reshape(-1, 1) -
        torch.tensor(X2['age'].values, dtype=torch.float32).reshape(1, -1)).pow(2.).div(100.0).neg().exp()
    # bmi
    K += (torch.tensor(X1['BMI'].values, dtype=torch.float32).reshape(-1, 1) -
        torch.tensor(X2['BMI'].values, dtype=torch.float32).reshape(1, -1)).pow(2.).div(25.0).neg().exp()
    return K

Kpt = kernel(Xprior, Xtrain)
Kpe = kernel(Xprior, Xeval)
Ktt = kernel(Xtrain, Xtrain)
Kpp = kernel(Xprior, Xprior)
Kte = kernel(Xtrain, Xeval)

# prepare prior function
nlgfr_prior = NLBR(
        Kx_train_train=Kpp,
        intercept_variance_ratio=float("inf"),
        regression_variance_ratio=float("inf"),
        variance_explained=0.99
    )
def prior(Y):
    return nlgfr_prior.ese_pairwise(Kpt.T, sq_dist(Y, Yprior)).T

# experiment parameters
seed = 0
variance_ratio = 1.
lr = 1e2
max_iter = 10000

bfr = BFR(
    squared_distance=sq_dist,
    prior=prior,
    Kx_train_train=Ktt,
    intercept_variance_ratio=variance_ratio,
    regression_variance_ratio=variance_ratio,
    variance_explained=0.99
)


# optimize
def clr(Y):
    log_Y = Y.clamp(min=1e-6).log()
    return log_Y - log_Y.mean(dim=-1, keepdim=True)
def inv_clr(Y):
    exp_Y = Y.exp()
    return exp_Y / exp_Y.sum(dim=-1, keepdim=True)
def center(Y):
    return Y - Y.mean(dim=-1, keepdim=True)

Y = bfr.ppfm_apgd(
    transform=clr,
    inv_transform=inv_clr,
    projection=center,
    Y_init=Yeval,
    Y_train=Ytrain,
    Kx_test_train=Kte.T,
    lr=lr,
    max_iter=max_iter,
    tol=1e-8
)

sq_dist(Y, Yeval).diag().mean().item()
# 1e-3: 0.01621929556131363
# 1e-2: 0.014428544789552689
# 0.1: 0.01350515615195036
# 1: 0.013895590789616108
# 10:
# 100: 0.014363091439008713

# self = bfr
# Y_init = Yeval
# Y_train = Ytrain
# Kx_test_train = Kte.T
# tol = 1e-6,
# momentum: float = 0.9,
# restart = True,
# projection = center
# transform = clr
# inv_transform = inv_clr