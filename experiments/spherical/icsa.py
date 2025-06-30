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

# ======================================================================================================================
# SETTINGS
DIRFIGURES = "./experiments/spherical/figures/"
os.makedirs(DIRFIGURES, exist_ok=True)
# Experiments
prior_invstrength = 0.1
# Kernel scales
X_scales = [0.5]
Y_scales = [0.5]
# Regularization parameters
regs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., ]
regs2 = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, ]
regs = [0.01]
# regs2 = [0.01]
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
        partial=False
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


# training data
X_train, Y_train, Yc_train = generate_data(N=20, seed=0)

# prior data
X_prior, Y_prior, Yc_prior = generate_data(N=100, seed=1, sd=0.3*prior_invstrength,
                                           equally_spread=True, partial=False)

# testing data
X_test, Y_test, Yc_test = generate_data(N=500, seed=2, equally_spread=True)
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# RUN EXPERIMENTS
nlbgfr_results = {}
nlbgfr_pred_error = {}
nlgfr_results = {}
nlgfr_pred_error = {}
for reg, X_scale, Y_scale, reg2 in itertools.product(regs, X_scales, Y_scales, regs2):
    print(f"X_scale: {X_scale} - Y_scale: {Y_scale} - reg: {reg} - reg2: {reg2}")

    # compute squared distances in X
    D2x_train_train = squared_L2_distance(X_train, X_train)  # N x N
    D2x_train_prior = squared_L2_distance(X_train, X_prior)  # N x M
    D2x_train_test = squared_L2_distance(X_train, X_test)
    # compute squared distances in Y
    D2y_train_train = sq_dist(Y_train, Y_train)  # N x N
    D2y_train_prior = sq_dist(Y_train, Y_prior)  # N x M
    Y_scale_ = (torch.median(D2y_train_train[D2y_train_train > 0.])).sqrt().item() * Y_scale

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
    nlbgfr = NLBGFR(kx_train_train, kx_train_test.T,
                    rxy_train=rxy_train, ry_train=ry_train,
                    reg=reg2)

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

    nlbgfr_results[(X_scale, Y_scale, reg, reg2)] = Y.detach().cpu().numpy(), rxy_train.detach().cpu().numpy()
    nlbgfr_pred_error[(X_scale, Y_scale, reg, reg2)] = (
        sq_dist(Y, Y_test).diag().mean().item(), # prediction error
        sq_dist(Y, Yc_test).diag().mean().item() # estimation error
    )



    # Prepare weights
    nlgfr = NLGFR(kx_train_train, kx_train_test.T, reg=reg2)

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
        ese = nlgfr.ese(D2y_train_pred)
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

    nlgfr_results[(X_scale, Y_scale, reg, reg2)] = Y.detach().cpu().numpy(), None
    nlgfr_pred_error[(X_scale, Y_scale, reg, reg2)] = (
        sq_dist(Y, Y_test).diag().mean().item(), # prediction error
        sq_dist(Y, Yc_test).diag().mean().item() # estimation error
    )

minval = math.inf
best = None
for k, v in nlgfr_pred_error.items():
    print(k, v)
    if v[1] < minval:
        best = k
        minval = v[1]
Y_nlgfr, _ = nlgfr_results[best]

minval = math.inf
best = None
for k, v in nlbgfr_pred_error.items():
    print(k, v)
    if v[1] < minval:
        best = k
        minval = v[1]
Y_nlbgfr, ratio = nlbgfr_results[best]
# ----------------------------------------------------------------------------------------------------------------------







# ======================================================================================================================
# PLOT RESULTS

# prepare frame
theta, phi = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
r = 1
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

exp="perfect"

plt.cla()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, color='k', alpha=0.1, linewidth=0.5)
cmap = ListedColormap(sns.color_palette("rocket_r", 256).as_hex())
ax.scatter(Y_train[:, 0], Y_train[:, 1], Y_train[:, 2], c="b", marker='o', label="train", alpha=1.)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.view_init(elev=30, azim=60)
plt.tight_layout()
ax.scatter(Yc_test[:, 0], Yc_test[:, 1], Yc_test[:, 2], c='k', marker='o', label="test (nlbgfr)", alpha=0.2, s=1)
plt.savefig(f"{DIRFIGURES}/{exp}1.png", bbox_inches="tight", dpi=300)
ax.scatter(Y_nlgfr[:, 0], Y_nlgfr[:, 1], Y_nlgfr[:, 2], c='b', marker='o', label="test (nlbgfr)", alpha=0.2)
plt.savefig(f"{DIRFIGURES}/{exp}2.png", bbox_inches="tight", dpi=300)
ax.scatter(Y_prior[:, 0], Y_prior[:, 1], Y_prior[:, 2], c='k', marker='o', label="prior", alpha=0.2)
plt.savefig(f"{DIRFIGURES}/{exp}3.png", bbox_inches="tight", dpi=300)
sc = ax.scatter(Y_train[:, 0], Y_train[:, 1], Y_train[:, 2], c=ratio, marker='o', label="train", alpha=1., cmap=cmap)
plt.savefig(f"{DIRFIGURES}/{exp}4.png", bbox_inches="tight", dpi=300)
ax.scatter(Y_nlbgfr[:, 0], Y_nlbgfr[:, 1], Y_nlbgfr[:, 2], c='g', marker='o', label="test (nlbgfr)", alpha=0.2)
plt.savefig(f"{DIRFIGURES}/{exp}5.png", bbox_inches="tight", dpi=300)

# for i, (exp, res) in enumerate(bgfr_results.items()):
#     # observed data + truth
#     plt.cla()
#     plt.figure(figsize=(5, 4))
#     ax = plt.gca()
#     ax.set_xlim(-2.5, 2.5)
#     ax.set_ylim(2., 4.)
#     ax.set_xlabel("Mean")
#     ax.set_ylabel("Std. deviation")
#     ax.axline((0, 3), slope=0.3, color="gray", linestyle="-")
#     ax.plot(Y_train[:, 0], Y_train[:, 1], "bo", markersize=3, alpha=0.2)
#     plt.savefig(f"{DIRFIGURES}/{exp}1.png", bbox_inches="tight", dpi=300)
#     ax.plot(Ygfr_pred[:, 0], Ygfr_pred[:, 1], "bo", markersize=3, alpha=0.2)
#     plt.savefig(f"{DIRFIGURES}/{exp}2.png", bbox_inches="tight", dpi=300)
#     ax.plot(res["Y_prior"][:, 0], res["Y_prior"][:, 1], "ko", alpha=0.5, markersize=3)
#     plt.savefig(f"{DIRFIGURES}/{exp}3.png", bbox_inches="tight", dpi=300)
#     ratio = res["best_ratio"]
#     df = pd.DataFrame({
#         "mu": Y_train[:, 0].detach().cpu().numpy(),
#         "sigma": Y_train[:, 1].detach().cpu().numpy(),
#         "ratio": ratio,
#     })
#     sns.scatterplot(data=df, x="mu", y="sigma", hue="ratio", palette="rocket_r", ax=ax, legend=True, zorder=10)
#     plt.savefig(f"{DIRFIGURES}/{exp}4.png", bbox_inches="tight", dpi=300)
#     Y = res["best_pred"]
#     ax.plot(Y[:, 0], Y[:, 1], "go", markersize=3, alpha=0.2)
#     plt.savefig(f"{DIRFIGURES}/{exp}5.png", bbox_inches="tight", dpi=300)

# ----------------------------------------------------------------------------------------------------------------------