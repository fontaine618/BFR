import pandas as pd
import numpy as np
import torch

# ======================================================================================================================
# SETTINGS
DIR_RESULTS = "/experiments_old/dmbt1/simulated/results5/"
n_otus = 109
# imputations
methods = {
    # "global_bc": "Global (Bray-Curtis)",
    "global_canberra": "Global (Canberra)",
    "global_aitchison": "Global (Aitchison)",
    "global_hellinger": "Global (Hellinger)",
    # "global_cosine": "Global (Cosine)",
    # "global_spherical": "Global (Spherical)",
    # "markov_bc": "Markov (Bray-Curtis)",
    "markov_canberra": "Markov (Canberra)",
    "markov_aitchison": "Markov (Aitchison)",
    "markov_hellinger": "Markov (Hellinger)",
    # "markov_cosine": "Markov (Cosine)",
    # "markov_spherical": "Markov (Spherical)",
    # "local_bc": "Local (Bray-Curtis)",
    "local_canberra": "Local (Canberra)",
    "local_aitchison": "Local (Aitchison)",
    "local_hellinger": "Local (Hellinger)",
    # "local_cosine": "Local (Cosine)",
    # "local_spherical": "Local (Spherical)",
    # "autoregressive_bc": "Autoregressive (Bray-Curtis)",
    "autoregressive_canberra": "Autoregressive (Canberra)",
    "autoregressive_aitchison": "Autoregressive (Aitchison)",
    "autoregressive_hellinger": "Autoregressive (Hellinger)",
    # "autoregressive_cosine": "Autoregressive (Cosine)",
    # "autoregressive_spherical": "Autoregressive (Spherical)",
    "locf": "LOCF",
    "complete": "Complete Cases",
    "original": "Oracle",
}
# ----------------------------------------------------------------------------------------------------------------------




# # ======================================================================================================================
# # DIFFERENTIAL ABUNDANCE ANALYSIS
results = pd.DataFrame(columns=["Seed", "Method", "Power", "FDR"])
for seed in range(5):
    DIR_SEED = DIR_RESULTS + f"{seed}/"
    beta = torch.Tensor(np.load(DIR_SEED + "original.beta.npy"))
    da = beta[:, 1, :].abs().gt(0.01).any(dim=1)
    da = pd.DataFrame({"true_differential": da}, index=[f"OTU{str(i+1).zfill(4)}" for i in range(n_otus)])

    # run trough all methods
    for method_name, method_display in methods.items():
        daa = pd.read_csv(DIR_SEED + method_name + ".da", index_col=0)
        # daa = daa.loc[daa["Timepoint"]<3]
        daa = da.join(daa, how="left")
        daa["differential"] = daa["differential"].fillna(False)
        # classification metrics
        tp = (daa["true_differential"] & daa["differential"]).sum()
        tn = (~daa["true_differential"] & ~daa["differential"]).sum()
        fp = (~daa["true_differential"] & daa["differential"]).sum()
        fn = (daa["true_differential"] & ~daa["differential"]).sum()
        p = tp + fn
        n = tn + fp
        power = tp / p if p > 0 else 0
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
        # add row to results
        results = pd.concat([
            results,
            pd.DataFrame({
                "Seed": [seed],
                "Method": [method_display],
                "Power": [power],
                "FDR": [fdr]
            })
        ], ignore_index=True)

results_avg = results.drop(columns=["Seed"]).groupby(["Method"]).mean()
print(results_avg.to_latex(float_format="%.3f", escape=False, na_rep="-"))
# # ----------------------------------------------------------------------------------------------------------------------

