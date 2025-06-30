from pyloseq.pyloseq import Pyloseq
import pandas as pd
import numpy as np
import scipy
import torch
import sys, argparse
import os
import utils.squared_metrics.compositional as c
from bfr.kulsif import kulsif
from bfr.bgfr import BGFR, GFR
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.kernels.laplacian import laplacian_kernel
from utils.squared_metrics.euclidean import squared_L2_distance
from pyloseq.zinb import zinb
from statsmodels.stats.multitest import multipletests
from experiments.dmbt1.simulated.gen_data import generate_longitudinal_microbiome_data


# ======================================================================================================================
# SETTINGS
DIR_RESULTS = "/home/simon/Documents/BFR/experiments/dmbt1/simulated/results5/"

# training squared distance
sq_dist = c.squared_bray_curtis_dissimilarity
sq_dist_name = "bc"
sq_dist_display = "Bray-Curtis"


# imputations
methods = {
    # "global_bc": "Global (Bray-Curtis)",
    # "global_canberra": "Global (Canberra)",
    "global_aitchison": "Global (Aitchison)",
    "global_hellinger": "Global (Hellinger)",
    # "global_cosine": "Global (Cosine)",
    # "global_spherical": "Global (Spherical)",
    # "markov_bc": "Markov (Bray-Curtis)",
    # "markov_canberra": "Markov (Canberra)",
    "markov_aitchison": "Markov (Aitchison)",
    "markov_hellinger": "Markov (Hellinger)",
    # "markov_cosine": "Markov (Cosine)",
    # "markov_spherical": "Markov (Spherical)",
    # "local_bc": "Local (Bray-Curtis)",
    # "local_canberra": "Local (Canberra)",
    "local_aitchison": "Local (Aitchison)",
    "local_hellinger": "Local (Hellinger)",
    # "local_cosine": "Local (Cosine)",
    # "local_spherical": "Local (Spherical)",
    # "autoregressive_bc": "Autoregressive (Bray-Curtis)",
    # "autoregressive_canberra": "Autoregressive (Canberra)",
    "autoregressive_aitchison": "Autoregressive (Aitchison)",
    "autoregressive_hellinger": "Autoregressive (Hellinger)",
    # "autoregressive_cosine": "Autoregressive (Cosine)",
    # "autoregressive_spherical": "Autoregressive (Spherical)",
    "locf": "LOCF"
}

# testing distances
metrics = {
    "bc": (c.squared_bray_curtis_dissimilarity, "Bray-Curtis"),
    # "canberra": (c.squared_canberra_distance, "Canberra"),
    "aitchison": (c.squared_aitchison_distance, "Aitchison"),
    "hellinger": (c.squared_hellinger_distance, "Hellinger"),
    "cosine": (c.squared_cosine_norm_distance, "Cosine"),
    "spherical": (c.squared_spherical_norm_distance, "Spherical"),
}
# ----------------------------------------------------------------------------------------------------------------------



# ======================================================================================================================
# COMPUTE METRICS


# setup dataframe to capture results
results = pd.DataFrame(columns=["Seed", "Method", "Metric", "Value"])

for seed in range(100):
    DIR_SEED = DIR_RESULTS + f"{seed}/"

    # load original data
    data = Pyloseq.load(DIR_SEED + "original")
    # get index of missing values
    sample_ids = data.sample_data.loc[data.sample_data["missing"]].index
    # get compositions
    true_compositions = torch.Tensor(data.otu_table.loc[sample_ids].values)

    # run trough all methods
    for method_name, method_display in methods.items():
        # load data
        ps = Pyloseq.load(DIR_SEED + method_name)
        imputed_compositions = torch.Tensor(ps.otu_table.loc[sample_ids].values)

        # compute metrics for each distance
        for metric_name, (metric_func, metric_display) in metrics.items():
            value = metric_func(true_compositions, imputed_compositions).diag().sqrt().mean()

            # add row to results
            results = pd.concat([
                results,
                pd.DataFrame({
                    "Seed": [seed],
                    "Method": [method_display],
                    "Metric": [metric_display],
                    "Value": [value.item()]
                })
            ], ignore_index=True)
# results.pivot(index="Method", columns="Metric", values="Value").sort_values("Bray-Curtis")
# Compute the rank of each method for each metric within seed
results["Rank"] = results.groupby(["Seed", "Metric"])["Value"].rank(method="min", ascending=True)
# print(results.pivot(index="Method", columns="Metric", values="Value"))
# Average rank
results_avg = results.groupby(["Method", "Metric"])["Rank"].mean().reset_index()
results_avg = results_avg.pivot(index="Method", columns="Metric", values="Rank")
# compute mean and std dev
df = results.groupby(["Method", "Metric"]).aggregate(
    Mean=("Value", "mean"),
    StdDev=("Value", "std"),
)
df["StdDev"] = df["StdDev"] / np.sqrt(100) 
# format mean (stddev) as string
df["Mean (StdDev)"] = df.apply(lambda x: f"{x['Mean']:.4f} ({x['StdDev']:.4f})", axis=1)
# reset index to have Method and Metric as columns
results_avg = df.reset_index().pivot(index="Method", columns="Metric", values="Mean (StdDev)")

# to latex table with 1 digit after 0
print(results_avg.to_latex())

# pivot the results for better readability: rows are metrics, columsn are methods
# results = results.pivot(index="Method", columns="Metric", values="Value")
#
# results.sort_values("Bray-Curtis")
# ----------------------------------------------------------------------------------------------------------------------