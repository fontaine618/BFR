import torch
# torch.cuda.set_per_process_memory_fraction(0.5, 0)
# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device("cpu")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import itertools

from matplotlib.colors import ListedColormap
from bfr.kulsif import kulsif
from bfr.bgfr import BGFR, GFR
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.squared_metrics.euclidean import squared_L2_distance

blue = "#001E44"
lightblue = "#96BEE6"

# ======================================================================================================================
# SETTINGS
DIRFIGURES = "./experiments/normal/figures/"
os.makedirs(DIRFIGURES, exist_ok=True)
# Experiments
prior_invstrengths = {
    "normal": 1.0,
}
n_seeds = 100
# Kernel scales
X_scales = [0.5]
Y_scales = [0.5]
# Regularization parameters
regs = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10.]
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
    sigma_fitted = b[0] + b[1] * X[:, 0]
    sigma = sigma_fitted + sd_factor * sd_sd * torch.randn(N)
    sigma_fitted = sigma_fitted.clamp(min=0.01)
    sigma = sigma.clamp(min=0.01)
    Y = torch.vstack((mu, sigma)).mT  # N x 2
    Yc = torch.vstack((mu_fitted, sigma_fitted)).mT
    return X, Y, Yc
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# RUN EXPERIMENTS
gfr_results = {}
bgfr_results = {}
for seed in range(n_seeds):

    # training data
    X_train, Y_train, Yc_train = generate_data(N=20, seed=seed)
    D2x_train_train = squared_L2_distance(X_train, X_train)
    D2y_train_train = squared_L2_distance(Y_train, Y_train)

    # testing data
    X_test, Y_test, Yc_test = generate_data(N=500, seed=1000+seed, equally_spread=True)

    # median heuristic for scales
    Dx_median = (torch.median(D2x_train_train[D2x_train_train > 0.])).sqrt().item()
    Dy_median = (torch.median(D2y_train_train[D2y_train_train > 0.])).sqrt().item()

    # gfr
    gfr = GFR(X_train, X_test)
    Y = torch.nn.Parameter(Yc_test.detach().clone())
    optimizer = torch.optim.Rprop([Y], lr=lr)
    prev_loss = float("inf")
    prev_Y = Y.clone().detach()
    for i in range(max_iter):
        optimizer.zero_grad()
        U_trainpred = squared_L2_distance(Y, Y_train)  # M x N
        ese = gfr.ese(U_trainpred)
        loss = ese.sum()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            max_change = (Y - prev_Y).abs().max()
            print(f"Iteration {i} - Loss: {loss.item()} - Linf: {max_change.item()}")
        prev_loss = loss.item()
        prev_Y = Y.clone().detach()
    Ygfr_pred = Y.detach().cpu().numpy()
    # Prediction error
    gfr_pred_error = squared_L2_distance(Y, Y_test).diag().mean().item()
    # Estimation error
    gfr_est_error = squared_L2_distance(Y, Yc_test).diag().mean().item()
    # save results
    gfr_results[seed] = {
        "Ygfr_pred": Ygfr_pred,
        "gfr_pred_error": gfr_pred_error,
        "gfr_est_error": gfr_est_error,
    }
    for exp, prior_invstrength in prior_invstrengths.items():
        print(f"Experiment: {exp} - Prior strength: {prior_invstrength}")
        # prior data
        X_prior, Y_prior, Yc_prior = generate_data(N=100, seed=2000+seed, equally_spread=True, sd_factor=prior_invstrength)
        D2x_prior_train = squared_L2_distance(X_train, X_prior)
        D2y_prior_train = squared_L2_distance(Y_train, Y_prior)

        # Iterate over parameters
        exp_results = {}
        best = None
        best_pred = None
        best_ratio = None
        minval = float("inf")
        for X_scale, Y_scale, reg in itertools.product(X_scales, Y_scales, regs):
            print(f"X_scale: {X_scale} - Y_scale: {Y_scale} - reg: {reg}")
            # Kernel across training points
            kx = squared_exponential_kernel(D2x_train_train, X_scale * Dx_median)
            ky = squared_exponential_kernel(D2y_train_train, Y_scale * Dy_median)

            # Kernel between training and prior points
            kx_prior = squared_exponential_kernel(D2x_prior_train, X_scale * Dx_median)
            ky_prior = squared_exponential_kernel(D2y_prior_train, Y_scale * Dy_median)

            # Density ratio estimation
            rxy_train = kulsif(kx*ky, kx_prior*ky_prior, reg=reg)
            rx_train = kulsif(kx, kx_prior, reg=reg)

            # Prepare weights
            bgfr = BGFR(X_train, X_test, rxy_train=rxy_train, rx_train=rx_train)

            # Optimize
            Y = torch.nn.Parameter(Yc_test.detach().clone())
            optimizer = torch.optim.Rprop([Y], lr=lr)
            prev_loss = float("inf")
            prev_Y = Y.clone().detach()
            for i in range(max_iter):
                optimizer.zero_grad()
                U_priorpred = squared_L2_distance(Y, Y_prior)  # M x B
                U_trainpred = squared_L2_distance(Y, Y_train)  # M x N
                ese = bgfr.ese(U_trainpred)
                loss = ese.sum()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    max_change = (Y - prev_Y).abs().max()
                    print(f"Iteration {i} - Loss: {loss.item()} - Linf: {max_change.item()}")
                prev_loss = loss.item()
                prev_Y = Y.clone().detach()

            # Prediction error
            pred_error = squared_L2_distance(Y, Y_test).diag().mean().item()

            # Estimation error
            est_error = squared_L2_distance(Y, Yc_test).diag().mean().item()

            # Store results
            exp_results[(X_scale, Y_scale, reg)] = pred_error, est_error

            # Check for best result
            if est_error < minval:
                minval = est_error
                best = (X_scale, Y_scale, reg)
                best_pred, best_ratio = Y.detach().cpu().numpy(), rxy_train.detach().cpu().numpy()

        exp_results = pd.DataFrame.from_dict(exp_results).T
        exp_results.reset_index(inplace=True)
        exp_results.columns = ["X_scale", "Y_scale", "reg", "pred_error", "est_error"]
        # Store best result
        bgfr_results[(exp, seed)] = {
            "best": best,
            "best_pred": best_pred,
            "best_ratio": best_ratio,
            "results": exp_results,
            "Y_prior": Y_prior.detach().cpu().numpy(),
        }
# ----------------------------------------------------------------------------------------------------------------------







# ======================================================================================================================
# PLOT RESULTS
for exp, prior_invstrength in prior_invstrengths.items():

    # observed data + truth
    plt.cla()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), squeeze=True)

    # Plot estimation error
    ax = axes[0]
    ax.axhline(y=1., color=blue, linestyle="--", label="GFR")
    for seed in range(n_seeds):
        res = bgfr_results[(exp, seed)]["results"]
        res["rel_est_error"] = res["est_error"] / gfr_results[seed]["gfr_est_error"]
        sns.lineplot(data=res, x="reg", y="rel_est_error", ax=ax,
                     color=lightblue, label=("BGFR" if seed==0 else None), alpha=0.2)
        # add lowest value
        minval = res["rel_est_error"].min()
        minreg = res.loc[res["rel_est_error"] == minval, "reg"].values[0]
        ax.plot(minreg, minval, "o", color=lightblue, markersize=5, alpha=1.)
    ax.set_xscale("log")
    ax.set_xlabel("KuLSIF Regularization")
    ax.set_ylabel("Relative estimation error")
    ax.set_yscale("log")
    ax.legend()

    # Plot best (seed 0)
    res = bgfr_results[(exp, 0)]
    Ygfr_pred = gfr_results[0]["Ygfr_pred"]
    X_train, Y_train, Yc_train = generate_data(N=20, seed=0)
    ax = axes[1]
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(2., 4.)
    ax.set_xlabel("Mean")
    ax.set_ylabel("Std. deviation")
    ax.axline((0, 3), slope=0.3, color="gray", linestyle="-", label="Ground truth")
    # ax.plot(Y_train[:, 0], Y_train[:, 1], "bo", markersize=3, alpha=0.2, label="Training data")
    ax.plot(Ygfr_pred[:, 0], Ygfr_pred[:, 1], "o", markersize=3, alpha=1., label="GFR prediction", c=blue)
    ax.plot(res["Y_prior"][:, 0], res["Y_prior"][:, 1], "ko", alpha=0.25, markersize=3, label="Prior data")
    ratio = res["best_ratio"]
    df = pd.DataFrame({
        "mu": Y_train[:, 0].detach().cpu().numpy(),
        "sigma": Y_train[:, 1].detach().cpu().numpy(),
        "ratio": ratio,
    })
    Y = res["best_pred"]
    ax.plot(Y[:, 0], Y[:, 1], "o", markersize=3, alpha=1., label="BGFR prediction", c=lightblue)
    sp = sns.scatterplot(data=df, x="mu", y="sigma", color="black", #hue="ratio",
                         size="ratio", label="Training data",
                         ax=ax, legend=False, zorder=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{DIRFIGURES}/{exp}.pdf")

# ----------------------------------------------------------------------------------------------------------------------