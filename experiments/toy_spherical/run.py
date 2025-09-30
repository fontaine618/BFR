import torch
# torch.set_default_device("cpu")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import itertools
from matplotlib.colors import ListedColormap
from nlbgfr.bfr import BFR
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.squared_metrics.spherical import squared_geodesic_distance, squared_chordal_distance
from utils.squared_metrics.euclidean import squared_L2_distance

blue = "#001E44"
lightblue = "#96BEE6"

# ======================================================================================================================
# SETTINGS
FIGNAME = "spherical_nlbgfr_geodesic"
DIRFIGURES = "./figures/"
os.makedirs(DIRFIGURES, exist_ok=True)
# Experiments
n_seeds = 1
variance_ratios = [1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7, float("inf")]
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

def prior(Yc, y, n_samples=50, sd=0.3, seed=1):
    torch.manual_seed(seed)
    N = Yc.size(0)
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
for seed, variance_ratio in itertools.product(range(n_seeds), variance_ratios):
    print(f"Running NLBGFR - Seed {seed} - Variance ratio {variance_ratio}")

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
        prior=lambda Yp: prior(Yc_train, Yp),
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
    results[(seed, variance_ratio, "NLBGFR")] = {
        "pfm": pfm,
        "pred_error": pred_error,
        "est_error": est_error,
    }
# ----------------------------------------------------------------------------------------------------------------------











# ======================================================================================================================
# PLOT RESULTS

# create color scale for variance ratios
n_colors = len(variance_ratios)
cmap = sns.color_palette("flare", n_colors=n_colors)
cmap = ListedColormap(cmap)

# prepare frame
theta, phi = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
r = 1
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)



# plot 1: plot predictions
plt.cla()
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1, projection='3d')
ax = plt.gca()
ax.plot_wireframe(x, y, z, color='k', alpha=0.05, linewidth=0.5)
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
# plt.tight_layout()
ax.plot(Yc_test[:, 0], Yc_test[:, 1], Yc_test[:, 2], color=blue,
            markersize=3, alpha=1., label="Ground truth", linewidth=2, linestyle="dotted", zorder=11)
plt.plot(Y_train[:, 0], Y_train[:, 1], Y_train[:, 2], "ko", markersize=3, alpha=1., label="Training data", zorder=10)
for seed, variance_ratio in itertools.product(range(n_seeds), variance_ratios):
    pfm = results[(seed, variance_ratio, "NLBGFR")]["pfm"]
    if variance_ratio < float("inf"):
        ax.plot(pfm[:, 0], pfm[:, 1], pfm[:, 2],
                 markersize=3, alpha=1.,
                    color=cmap(variance_ratios.index(variance_ratio)),
                    marker="", linestyle="-", linewidth=2
            )
    else:
        ax.plot(pfm[:, 0], pfm[:, 1], pfm[:, 2],
                 markersize=3, alpha=1.,
                    color="grey",
                    marker="", linestyle="--", linewidth=2,
                label="NLGFR"
            )
plt.legend(loc="upper left")
cmap = sns.color_palette("flare", n_colors=n_colors)
cmap = ListedColormap(cmap[:-1])
sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(vmin=0, vmax=n_colors-1)
)

sm.set_array([])
cbar = plt.colorbar(sm, ticks=variance_ratios[:-1], ax=plt.gca())
cbar.set_label('BFR Variance Ratio', rotation=270, labelpad=15)
cbar.set_ticks(
    ticks=[i+0.5 for i in range(n_colors-1)],
    labels=[f"{vr:.0e}" for vr in variance_ratios[:-1]]
)
plt.tight_layout()
# plot 2: plot estimation error vs variance ratio
plt.subplot(1, 2, 2)
est_errors = [results[(0, vr, "NLBGFR")]["est_error"] for vr in variance_ratios[:-1]]
plt.axhline(y=est_errors[-1], color="grey", linestyle='--', label="NLGFR")
plt.plot(variance_ratios[:-1], est_errors, linestyle='-',
         color=cmap(variance_ratios.index(variance_ratios[1])), label="BFR")
plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Variance Ratio")
plt.ylabel("Estimation Error")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig(f"{DIRFIGURES}/{FIGNAME}.pdf")

# ----------------------------------------------------------------------------------------------------------------------














