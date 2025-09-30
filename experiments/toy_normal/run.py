import torch
# torch.set_default_device("cpu")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools

from matplotlib.colors import ListedColormap
from nlbgfr.bfr import BFR
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.squared_metrics.euclidean import squared_L2_distance

blue = "#001E44"
lightblue = "#96BEE6"

# ======================================================================================================================
# SETTINGS
FIGNAME = "normal_nlbgfr"
DIRFIGURES = "./figures/"
# DIRFIGURES = "./experiments/toy_normal/figures/"
os.makedirs(DIRFIGURES, exist_ok=True)
# Experiments
n_seeds = 1
variance_ratios = [1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7, float("inf")]
# Fit parameters
max_iter = 100
lr = 0.05
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# DATA GENERATION
def generate_data(
        N,
        a=(0., 1.),
        b=(3., 0.3),
        sd_mean=0.3,
        sd_sd=0.2,
        seed=0,
        equally_spread=False,
        sd_factor=1.
):
    """
    X ~ U[-2, 2]
    Y ~ N(mu(X), sigma(X))
    mu(X) = a[0] + a[1] * X + N(0, sd_mean)
    sigma(X) = b[0] + b[1] * X + N(0, sd_logsd)
    """
    torch.manual_seed(seed)
    if equally_spread:
        X = torch.linspace(-2, 2, N).reshape(-1, 1)  # regular grid on [-2, 2]
    else:
        X = torch.rand(N, 1) * 4. - 2.  # U[-2, 2]
    mu_fitted = a[0] + a[1] * X[:, 0]
    mu = mu_fitted + sd_factor * sd_mean * torch.randn(N)
    sigma_fitted = b[0] + b[1] * X[:, 0]# + 0.05 * X[:, 0]**2
    sigma = sigma_fitted + sd_factor * sd_sd * torch.randn(N)
    sigma_fitted = sigma_fitted.clamp(min=0.01)
    sigma = sigma.clamp(min=0.01)
    Y = torch.vstack((mu, sigma)).mT  # N x 2
    Yc = torch.vstack((mu_fitted, sigma_fitted)).mT
    return X, Y, Yc

X, Y, Yc = generate_data(N=20, seed=0)

def prior(
        Yc, y,
        sd_mean=0.3,
        sd_sd=0.2,
):
    """
    Computes the true E[U(y)|Xi]

    :param Yc: Nx2 tensor true centers
    :param y: Mx2 tensor of points to evaluate
    :param sd_mean: float
    :param sd_sd: float
    :return: MxN tensor of expected squared distances
    """
    N = Yc.size(0)
    M = y.size(0)
    variance_term = sd_mean**2 + sd_sd**2
    bias_term = (Yc.unsqueeze(0) - y.unsqueeze(1)).pow(2).sum(-1)
    return variance_term + bias_term  # M x N

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
        squared_distance=squared_L2_distance,
        prior=lambda Yp: prior(Yc_train, Yp),
        Kx_train_train=Kx_train_train,
        intercept_variance_ratio=variance_ratio,
        regression_variance_ratio=variance_ratio,
        variance_explained=0.99
    )

    # optimize
    Y = bfr.ppfm_apgd(
        Y_init=Yc_test,
        Y_train=Y_train,
        Kx_test_train=Kx_test_train,
        lr=lr,
        max_iter=max_iter
    )
    pfm = Y.detach().cpu().numpy()

    # Prediction error
    pred_error = squared_L2_distance(Y, Y_test).diag().mean().item()
    # Estimation error
    est_error = squared_L2_distance(Y, Yc_test).diag().mean().item()
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

# plot 1: plot predictions
plt.cla()
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.xlim(-2.5, 2.5)
plt.ylim(2.3, 4.)
plt.xlabel("Mean")
plt.ylabel("Std. deviation")
plt.plot(Yc_test[:, 0], Yc_test[:, 1], color=blue,
            markersize=3, alpha=1., label="Ground truth", linewidth=2, linestyle="dotted", zorder=11)
plt.plot(Y_train[:, 0], Y_train[:, 1], "ko", markersize=3, alpha=1., label="Training data", zorder=10)
for seed, variance_ratio in itertools.product(range(n_seeds), variance_ratios):
    pfm = results[(seed, variance_ratio, "NLBGFR")]["pfm"]
    est_error = results[(seed, variance_ratio, "NLBGFR")]["est_error"]
    if variance_ratio < float("inf"):
        plt.plot(pfm[:, 0], pfm[:, 1],
                 markersize=3, alpha=1.,
                    color=cmap(variance_ratios.index(variance_ratio)),
                    marker="", linestyle="-", linewidth=2
            )
    else:
        plt.plot(pfm[:, 0], pfm[:, 1],
                 markersize=3, alpha=1.,
                    color="grey",
                    marker="", linestyle="--", linewidth=2, label="NLGFR",
                 zorder=9
            )


plt.grid(True, which="both", ls="--", linewidth=0.5)
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
plt.xlabel("Variance Ratio")
plt.ylabel("Estimation Error")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig(f"{DIRFIGURES}/{FIGNAME}.pdf")

# ----------------------------------------------------------------------------------------------------------------------