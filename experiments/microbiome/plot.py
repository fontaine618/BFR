import torch
import os
import sys
import pandas as pd
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")



# ======================================================================================================================
# EXPERIMENTS
prior_dataset_name = "MetaCardis_2020_a"
train_test_dataset_names = ["QinJ_2012", "KarlssonFH_2013"]
dataset_display_names = {
    "QinJ_2012": "Qin et al. (2012)",
    "KarlssonFH_2013": "Karlsson et al. (2013)",
}

variance_ratios = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, float("inf")]
sq_dists = {
    "hellinger": "Hellinger",
    "aitchison": "Aitchison",
    "spherical": "Geodesic",
}
experiments = list(itertools.product(
    sq_dists.keys(),
    train_test_dataset_names,
    variance_ratios,
))
n_experiments = len(experiments)
n_seeds = 10
metrics = {
    "bc": "Bray-Curtis",
    "aitchison": "Aitchison",
    "hellinger": "Hellinger",
    "spherical": "Geodesic"
}
colors = {
    "hellinger": "#1f77b4",
    "aitchison": "#ff7f0e",
    "spherical": "#2ca02c",
}
# ----------------------------------------------------------------------------------------------------------------------




# ======================================================================================================================
# LOAD ALL RESULTS
results = []
for seed in range(n_seeds):
    DIR_RESULTS = (f"./experiments/microbiome/results/{seed}/")
    FILEPATH = os.path.join(DIR_RESULTS, "metrics.csv")
    if not os.path.exists(FILEPATH):
        continue
    df = pd.read_csv(FILEPATH)
    results.append(df)
results = pd.concat(results, ignore_index=True)
bfr = results[results["Variance Ratio"] < float("inf")]
nlgfr = results[results["Variance Ratio"] == float("inf")]
# ----------------------------------------------------------------------------------------------------------------------



# ======================================================================================================================
# PLOT RESULTS

# setup figure
plt.cla()
fig, axes = plt.subplots(
    nrows=len(metrics),
    ncols=len(train_test_dataset_names),
    figsize=(10, 10),
    sharex=True,
    # sharey="row",
)
# for each experiment, plot results
for ((row, metric), (col, dataset)) in itertools.product(
    enumerate(metrics.keys()),
    enumerate(train_test_dataset_names)
):
    ax = axes[row, col]
    ymax = 0.
    for dist_name, dist_display in sq_dists.items():
        # plot BFR results
        df_plot = bfr[
            (bfr["Metric"] == metrics[metric]) &
            (bfr["Distance"] == dist_name) &
            (bfr["Train/Test Dataset"] == dataset)
        ]
        if df_plot.shape[0] == 0:
            continue
        df_mean = df_plot.groupby("Variance Ratio")["Value"].mean().reset_index()
        df_std = df_plot.groupby("Variance Ratio")["Value"].sem().reset_index()
        ax.plot(
            df_mean["Variance Ratio"],
            df_mean["Value"],
            label=f"BFR ({dist_display})" if (row == 0 and col == 0) else None,
            color=colors[dist_name]
        )
        ax.fill_between(
            df_mean["Variance Ratio"],
            df_mean["Value"] - df_std["Value"],
            df_mean["Value"] + df_std["Value"],
            alpha=0.3,
            color=colors[dist_name]
        )
        # plot NLGFR results
        df_plot = nlgfr[
            (nlgfr["Metric"] == metrics[metric]) &
            (nlgfr["Distance"] == dist_name) &
            (nlgfr["Train/Test Dataset"] == dataset)
        ]
        if df_plot.shape[0] == 0:
            continue
        if df_plot.shape[0] > 0:
            mean_nlgfr = df_plot["Value"].mean()
            std_nlgfr = df_plot["Value"].sem()
            ax.axhline(
                mean_nlgfr,
                color=colors[dist_name],
                linestyle="--",
                label=f"NLGFR ({dist_display})" if (row == 0 and col == 0) else None,
            )
            ax.fill_between(
                [min(variance_ratios), max(variance_ratios)],
                [mean_nlgfr - std_nlgfr, mean_nlgfr - std_nlgfr],
                [mean_nlgfr + std_nlgfr, mean_nlgfr + std_nlgfr],
                alpha=0.3,
                color=colors[dist_name]
            )
            ymax = max(ymax, mean_nlgfr)
    # set log scale for x-axis
    ax.set_xscale("log")
    # ax.set_yscale("log")
    yl, yu = ax.get_ylim()
    ax.set_ylim(yl, min(yu, ymax * 1.1))
    # set labels
    if row == len(metrics) - 1:
        ax.set_xlabel("Variance Ratio")
    if col == 0:
        ax.set_ylabel(metrics[metric])
    if row == 0:
        ax.set_title(dataset_display_names[dataset])
    # if row == 0 and col == len(train_test_dataset_names) - 1:
    #     reorder = lambda hl, nc: (sum((lis[i::nc] for i in range(nc)), []) for lis in hl)
    #     h_l = ax.get_legend_handles_labels()
    #     ax.legend(*reorder(h_l, 2), loc="upper right", fontsize="small", ncol=2, )
    # grid only major
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)
plt.tight_layout()
fig.align_ylabels(axes[:, 0])
fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize="small")
plt.subplots_adjust(bottom=0.13)
plt.savefig("./experiments/microbiome/results/metrics.pdf")
# ----------------------------------------------------------------------------------------------------------------------


