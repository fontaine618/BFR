import torch
import os
import pandas as pd
import itertools
from nlbgfr.bfr import BFR
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.squared_metrics.spherical import squared_geodesic_distance, squared_chordal_distance
from utils.squared_metrics.euclidean import squared_L2_distance


# ======================================================================================================================
# SETTINGS
FIGNAME = "spherical_rotated_prior"
DIRFIGURES = "./figures/"
os.makedirs(DIRFIGURES, exist_ok=True)
# Experiments
n_seeds = 100
variance_ratios = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 1e3, 1e4, float("inf")]
rotations = [0, 5, 10, 15]
# Fit parameters
max_iter = 1000
lr = 0.001
sq_dist = squared_geodesic_distance
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# DATA GENERATION

def generate_data(
        N=50,
        sd=0.3,
        seed=0,
        equally_spread=False,
):
    torch.manual_seed(seed)
    if equally_spread:
        X = torch.linspace(0, 1, N).reshape(-1, 1)
    else:
        X = torch.rand(N, 1)
    y1 = (1-X.pow(2.)).pow(0.5)*torch.cos(torch.pi*X)
    y2 = (1-X.pow(2.)).pow(0.5)*torch.sin(torch.pi*X)
    y3 = X
    mu = torch.stack([y1, y2, y3], dim=-1).squeeze(1)
    mu = mu / mu.norm(dim=-1, keepdim=True)  # normalize to unit sphere
    Y = mu + sd * torch.randn(N, 3)
    Y = Y / Y.norm(dim=-1, keepdim=True)  # normalize to unit sphere
    return X, Y, mu

def prior(Yc, y, n_samples=50, sd=0.3, seed=1, rotation=0):
    torch.manual_seed(seed)
    N = Yc.size(0)
    # rotate Yc
    if rotation != 0:
        theta = torch.as_tensor(rotation) * torch.pi / 180.
        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0.],
            [torch.sin(theta), torch.cos(theta), 0.],
            [0., 0., 1.]
        ])
        Yc = (R @ Yc.T).T
    samples = Yc.unsqueeze(0) + sd * torch.randn(n_samples, N, 3)
    samples = samples / samples.norm(dim=-1, keepdim=True)
    # average distance between samples and y
    avgD2 = torch.stack([
        sq_dist(samples[i, :, :], y) for i in range(n_samples)
    ], dim=0).mean(0)
    return avgD2.T


# ----------------------------------------------------------------------------------------------------------------------







# ======================================================================================================================
# RUN EXPERIMENTS


results = {}
for seed, variance_ratio, rotation in itertools.product(range(n_seeds), variance_ratios, rotations):
    print(f"Running NLBGFR - Seed {seed} - Variance ratio {variance_ratio} - Rotation {rotation}")

    # training data
    X_train, Y_train, Yc_train = generate_data(N=20, seed=seed, equally_spread=True)
    D2x_train_train = squared_L2_distance(X_train, X_train)

    # testing data
    X_test, Y_test, Yc_test = generate_data(N=100, seed=1000+seed, equally_spread=True)
    D2x_test_train = squared_L2_distance(X_test, X_train)

    # median heuristic for scales
    Dx_median = (torch.median(D2x_train_train[D2x_train_train > 0.])).sqrt().item()

    # compute kernel matrices
    Kx_train_train = squared_exponential_kernel(D2x_train_train, Dx_median)
    Kx_test_train = squared_exponential_kernel(D2x_test_train, Dx_median)

    bfr = BFR(
        squared_distance=sq_dist,
        prior=lambda Yp: prior(Yc_train, Yp, rotation=rotation),
        Kx_train_train=Kx_train_train,
        intercept_variance_ratio=variance_ratio,
        regression_variance_ratio=variance_ratio,
        variance_explained=0.99
    )

    # optimize
    Y = bfr.ppfm_apgd(
        projection=lambda Y: Y / Y.norm(dim=-1, keepdim=True),
        Y_init=Yc_test,
        Y_train=Y_train,
        Kx_test_train=Kx_test_train,
        lr=lr,
        max_iter=max_iter
    )

    pfm = Y.detach().cpu().numpy()
    # Prediction error
    pred_error = sq_dist(Y, Y_test).diag().mean().item()
    # Estimation error
    est_error = sq_dist(Y, Yc_test).diag().mean().item()
    # save results
    results[(seed, variance_ratio, rotation)] = {
        # "pfm": pfm,
        # "pred_error": pred_error,
        "est_error": est_error,
    }

# to pd dataframe and save
df = pd.DataFrame.from_dict(results, orient="index")
df.index = pd.MultiIndex.from_tuples(df.index, names=["seed", "variance_ratio", "rotation"])
df.to_csv(f"{DIRFIGURES}/{FIGNAME}.csv")
# ----------------------------------------------------------------------------------------------------------------------














