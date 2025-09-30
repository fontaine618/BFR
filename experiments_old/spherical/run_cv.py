import torch
# torch.cuda.set_per_process_memory_fraction(0.5, 0)
# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device("cpu")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import math
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from bfr.kulsif import kulsif
from bfr.nlbgfr import NLGFR, NLBGFR
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.squared_metrics.spherical import squared_geodesic_distance, squared_cosine_distance
from utils.squared_metrics.euclidean import squared_L2_distance
from utils.kernels.laplacian import laplacian_kernel

blue = "#001E44"
lightblue = "#96BEE6"

# ======================================================================================================================
# SETTINGS
DIRFIGURES = "./experiments/spherical/figures/"
os.makedirs(DIRFIGURES, exist_ok=True)
# Experiments
prior_invstrength = 0.1
n_seeds = 5
# Kernel scales
X_scales = [0.5]
Y_scales = [0.5]
# Regularization parameters
regs = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 3., 10.]
# regs2 = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, ]
# regs = [0.01]
regs2 = [0.0001]
# Fit parameters
max_iter = 1000
lr = 0.005
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
    mu = mu / mu.norm(dim=-1, keepdim=True)  # normalize to unit sphere (should be fine alreadY?
    Y = mu + sd * torch.randn(N, 3)
    Y = Y / Y.norm(dim=-1, keepdim=True)  # normalize to unit sphere
    return X, Y, mu



# ----------------------------------------------------------------------------------------------------------------------


def nlbgfr_predict(
        X_train, Y_train, Yc_train,
        X_test, Y_test, Yc_test,
        X_prior, Y_prior, Yc_prior,
        Dx_median, Dy_median,
        X_scale=0.5, Y_scale=0.5,
        reg=0.001, reg2=1e-5, lr=0.005, max_iter=1000,
        bayesian=True
):
    # compute squared distances in X
    D2x_train_train = squared_L2_distance(X_train, X_train)  # N x N
    D2x_train_prior = squared_L2_distance(X_train, X_prior)  # N x M
    D2x_train_test = squared_L2_distance(X_train, X_test)
    # compute squared distances in Y
    D2y_train_train = sq_dist(Y_train, Y_train)  # N x N
    D2y_train_prior = sq_dist(Y_train, Y_prior)  # N x M
    Y_scale_ = Dy_median * Y_scale

    # Kernels across training points
    kx_train_train = squared_exponential_kernel(D2x_train_train, X_scale)
    ky_train_train = laplacian_kernel(D2y_train_train, Y_scale_)
    # Kernels between training and prior points
    kx_train_prior = squared_exponential_kernel(D2x_train_prior, X_scale)
    ky_train_prior = laplacian_kernel(D2y_train_prior, Y_scale)
    # Kernel between training and testing points
    kx_train_test = squared_exponential_kernel(D2x_train_test, X_scale)
    # Product kernel
    K_train_train = kx_train_train * ky_train_train
    K_train_prior = kx_train_prior * ky_train_prior

    # Density ratio estimation
    ry_train = kulsif(ky_train_train, ky_train_prior, reg=reg)
    rxy_train = kulsif(K_train_train, K_train_prior, reg=reg)

    # Prepare weights
    if bayesian:
        nlbgfr = NLBGFR(kx_train_train, kx_train_test.T,
                        rxy_train=rxy_train, ry_train=ry_train,
                        reg=reg2)
    else:
        nlbgfr = NLGFR(kx_train_train, kx_train_test.T, reg=reg2)

    # Prepare optimizer
    Y = torch.nn.Parameter(Yc_test.detach().clone())
    optimizer = torch.optim.Rprop([Y], lr=lr)

    # Predict
    prev_loss = torch.inf
    prev_Y = Y.clone().detach()
    for i in range(max_iter):
        # compute squared distances
        optimizer.zero_grad()
        D2y_train_pred = sq_dist(Y, Y_train)  # M x N
        ese = nlbgfr.ese(D2y_train_pred)
        loss = ese.sum()
        loss.backward()
        # projected gradient step
        with torch.no_grad():
            Y -= lr * Y.grad
            Y.grad.zero_()
            Y.data = Y.data / Y.data.norm(dim=-1, keepdim=True)
        if i % 100 == 0:
            max_change = (Y - prev_Y).abs().max()
            print(f"Iteration {i} - Loss: {loss.item()} - Linf: {max_change.item()}")
        prev_loss = loss.item()
        prev_Y = Y.clone().detach()
    return Y.clone().detach(), rxy_train.clone().detach()

# ======================================================================================================================
# RUN EXPERIMENTS
nlbgfr_results = {}
nlbgfr_pred_error = {}
nlbgfr_cv_error = {}
nlgfr_results = {}
nlgfr_pred_error = {}
nlgfr_cv_error = {}
for reg, X_scale, Y_scale, reg2, seed in itertools.product(regs, X_scales, Y_scales, regs2, range(n_seeds)):
    print(f"X_scale: {X_scale} - Y_scale: {Y_scale} - reg: {reg} - reg2: {reg2}")

    # training data
    X_train, Y_train, Yc_train = generate_data(N=20, seed=seed)

    # prior data
    X_prior, Y_prior, Yc_prior = generate_data(N=100, seed=seed+1000, sd=0.3 * prior_invstrength,
                                               equally_spread=True)

    # testing data
    X_test, Y_test, Yc_test = generate_data(N=500, seed=seed+2000, equally_spread=True)

    # prepare rescaling
    D2y_train_train = sq_dist(Y_train, Y_train)  # N x N
    Dy_median = (torch.median(D2y_train_train[D2y_train_train > 0.])).sqrt().item()

    Y, rxy_train = nlbgfr_predict(
        X_train, Y_train, Yc_train,
        X_test, Y_test, Yc_test,
        X_prior, Y_prior, Yc_prior,
        None, Dy_median,
        X_scale=X_scale, Y_scale=Y_scale,
        reg=reg, reg2=reg2, lr=lr, max_iter=max_iter,
        bayesian=True
    )

    nlbgfr_results[(X_scale, Y_scale, reg, reg2, seed)] = Y.detach().cpu().numpy(), rxy_train.detach().cpu().numpy()
    nlbgfr_pred_error[(X_scale, Y_scale, reg, reg2, seed)] = (
        sq_dist(Y, Y_test).diag().mean().item(), # prediction error
        sq_dist(Y, Yc_test).diag().mean().item() # estimation error
    )

    # Cross-validation error
    cv_error = 0.
    n_folds = 10
    foldid = torch.arange(X_train.shape[0]) % n_folds  # Randomly assign folds
    for fold in range(n_folds):
        Y_fold, _ = nlbgfr_predict(
            X_train[foldid != fold], Y_train[foldid != fold], Yc_train[foldid != fold],
            X_train[foldid == fold], Y_train[foldid == fold], Yc_train[foldid == fold],
            X_prior, Y_prior, Yc_prior,
            None, Dy_median,
            X_scale=X_scale, Y_scale=Y_scale,
            reg=reg, reg2=reg2, lr=lr, max_iter=max_iter
        )
        cv_error += sq_dist(Y_fold, Y_train[foldid == fold]).diag().sum().item()
    nlbgfr_cv_error[(X_scale, Y_scale, reg, reg2, seed)] = cv_error

    # Prepare weights
    Y, rxy_train = nlbgfr_predict(
        X_train, Y_train, Yc_train,
        X_test, Y_test, Yc_test,
        X_prior, Y_prior, Yc_prior,
        None, Dy_median,
        X_scale=X_scale, Y_scale=Y_scale,
        reg=reg, reg2=reg2, lr=lr, max_iter=max_iter,
        bayesian=False
    )

    nlgfr_results[(X_scale, Y_scale, reg2, seed)] = Y.detach().cpu().numpy(), None
    nlgfr_pred_error[(X_scale, Y_scale, reg2, seed)] = (
        sq_dist(Y, Y_test).diag().mean().item(), # prediction error
        sq_dist(Y, Yc_test).diag().mean().item() # estimation error
    )
# ----------------------------------------------------------------------------------------------------------------------







# ======================================================================================================================
# PLOT RESULTS

# prepare frame
theta, phi = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
r = 1
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

exp="spherical"
X_scale = X_scales[0]
Y_scale = Y_scales[0]
reg2 = regs2[0]

plt.cla()
fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(121)
ax.axhline(y=1., color=blue, linestyle="--", label="NLGFR")
for seed in range(n_seeds):
    nlgfr_error = nlgfr_pred_error[(X_scale, Y_scale, reg2, seed)][0]
    rel_errors = {reg: nlbgfr_pred_error[(X_scale, Y_scale, reg, reg2, seed)][0]/nlgfr_error for reg in regs}
    ax.plot(regs, rel_errors.values(), color=lightblue, label="NLBGFR" if seed==0 else None)
    cv_errors = {reg: nlbgfr_cv_error[(X_scale, Y_scale, reg, reg2, seed)] for reg in regs}
    best = min(cv_errors, key=lambda x: x)
    ax.scatter(best, rel_errors[best], color=lightblue, marker='o', s=50)
# ax.axhline(y=nlgfr_pred_error[best][1], color=blue, linestyle="--", label="NLGFR")
# ax.plot(regs, errors, color=lightblue, label="NLBGFR")
ax.set_xscale("log")
ax.set_xlabel("KuLSIF Regularization")
ax.set_ylabel("Relative prediction error")
ax.legend()

seed = 0
ax = fig.add_subplot(122, projection='3d')
ax.plot_wireframe(x, y, z, color='k', alpha=0.05, linewidth=0.5)
cmap = ListedColormap(sns.color_palette("rocket_r", 256).as_hex())

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.7, 0.7)
ax.set_zlim(-0.7, 0.7)
ax.view_init(elev=30, azim=60)
ax.set_axis_off()
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.tight_layout()
ax.scatter(Yc_test[:, 0], Yc_test[:, 1], Yc_test[:, 2], c='k', marker='o', label="Ground truth", alpha=1., s=1)
# ax.scatter(Y_nlgfr[:, 0], Y_nlgfr[:, 1], Y_nlgfr[:, 2], c=blue, marker='o', label="NLGFR prediction", alpha=1.0, s=5)
ax.scatter(Y_prior[:, 0], Y_prior[:, 1], Y_prior[:, 2], c='k', marker='o', label="Prior data", alpha=0.2, s=5)
# sc = ax.scatter(Y_train[:, 0], Y_train[:, 1], Y_train[:, 2], s=ratio*20, marker='o', label="Training data", alpha=1., c="k")
# ax.scatter(Y_nlbgfr[:, 0], Y_nlbgfr[:, 1], Y_nlbgfr[:, 2], c=lightblue, marker='o', label="NLBGFR predictions", alpha=1.0, s=5)
ax.legend(framealpha=1.)
plt.savefig(f"{DIRFIGURES}/{exp}_cv.pdf")

# ----------------------------------------------------------------------------------------------------------------------