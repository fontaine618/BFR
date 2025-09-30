from pyloseq.pyloseq import Pyloseq
import pandas as pd
import numpy as np
import torch
import utils.squared_metrics.compositional as c

# ======================================================================================================================
# SETTINGS
DIR_RESULTS = "/experiments_old/dmbt1/simulated/results/"
n_seeds = 100


# imputations
methods = {
    "global_aitchison": "Global (Aitchison)",
    "global_hellinger": "Global (Hellinger)",
    "global_spherical": "Global (Geodesic)",
    "markov_aitchison": "Markov (Aitchison)",
    "markov_hellinger": "Markov (Hellinger)",
    "markov_spherical": "Markov (Geodesic)",
    "local_aitchison": "Local (Aitchison)",
    "local_hellinger": "Local (Hellinger)",
    "local_spherical": "Local (Geodesic)",
    "gfr_aitchison": "GFR (Aitchison)",
    "gfr_hellinger": "GFR (Hellinger)",
    "gfr_spherical": "GFR (Geodesic)",
    "locf": "LOCF"
}

# testing distances
metrics = {
    "bc": (c.squared_bray_curtis_dissimilarity, "Bray-Curtis"),
    # "canberra": (c.squared_canberra_distance, "Canberra"),
    "aitchison": (c.squared_aitchison_distance, "Aitchison"),
    "hellinger": (c.squared_hellinger_distance, "Hellinger"),
    # "cosine": (c.squared_cosine_norm_distance, "Cosine"),
    "spherical": (c.squared_spherical_norm_distance, "Geodesic"),
}
# ----------------------------------------------------------------------------------------------------------------------



# ======================================================================================================================
# COMPUTE METRICS


# setup dataframe to capture results
results = pd.DataFrame(columns=["Seed", "Method", "Metric", "Value"])

for seed in range(n_seeds):
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
df["StdDev"] = df["StdDev"] / np.sqrt(n_seeds)
# format mean (stddev) as string
df["Mean (StdDev)"] = df.apply(lambda x: f"{x['Mean']:.4f} ({x['StdDev']:.4f})", axis=1)
# reset index to have Method and Metric as columns
results_avg = df.reset_index().pivot(index="Method", columns="Metric", values="Mean (StdDev)")
print(results_avg.to_latex())
# ----------------------------------------------------------------------------------------------------------------------