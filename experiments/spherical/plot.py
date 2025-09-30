import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

blue = "#001E44"
lightblue = "#96BEE6"
orange = "#E77961"

# ======================================================================================================================
# SETTINGS
FIGNAME = "spherical_rotated_prior"
DIRFIGURES = "./figures/"
# DIRFIGURES = "./experiments/spherical/figures/"
os.makedirs(DIRFIGURES, exist_ok=True)
# Experiments
n_seeds = 100
variance_ratios = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 1e3, 1e4, float("inf")]
rotations = [0, 5, 10, 15]

results = pd.read_csv(f"{DIRFIGURES}/{FIGNAME}.csv", index_col=[0, 1, 2])
# ----------------------------------------------------------------------------------------------------------------------





# ======================================================================================================================
# PLOT RESULTS
fig, axes = plt.subplots(
    ncols=len(rotations),
    figsize=(10, 3),
    sharex=True,
    sharey=True
)
for i, rotation in enumerate(rotations):
    ax = axes[i]
    ax.set_title(f"Prior rotation {rotation}°", fontsize=14)
    est_errors_mean = []
    est_errors_se = []
    for vr in variance_ratios:
        est_errors = [results.loc[(seed, vr, rotation)]["est_error"] for seed in range(n_seeds)]
        est_errors_mean.append(np.mean(est_errors))
        est_errors_se.append(np.std(est_errors) / np.sqrt(len(est_errors)))
    est_errors_mean = np.array(est_errors_mean)
    est_errors_se = np.array(est_errors_se)
    ax.axhline(y=est_errors_mean[-1], color="grey", linestyle="--", label="NLGFR",
               xmin=0.05, xmax=0.95)
    ax.fill_between(
        variance_ratios,
        est_errors_mean[-1] - est_errors_se[-1],
        est_errors_mean[-1] + est_errors_se[-1],
        color="grey",
        alpha=0.3
    )
    ax.plot(variance_ratios[:-1], est_errors_mean[:-1], color=orange, label="BFR")
    ax.fill_between(
        variance_ratios[:-1],
        est_errors_mean[:-1] - est_errors_se[:-1],
        est_errors_mean[:-1] + est_errors_se[:-1],
        color=orange,
        alpha=0.3
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BFR Variance ratio", fontsize=12)
    if i == 0:
        ax.set_ylabel("Estimation error", fontsize=12)
        ax.legend(fontsize=10)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{DIRFIGURES}/{FIGNAME}.pdf")
# ----------------------------------------------------------------------------------------------------------------------














