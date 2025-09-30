import pandas as pd

# ======================================================================================================================
# SETTINGS
DIR_RESULTS = "/experiments_old/dmbt1/real/results/"
n_otus = 109
# imputations
methods = {
    # "global_bc": "Global (Bray-Curtis)",
    # "global_canberra": "Global (Canberra)",
    # "global_aitchison": "Global (Aitchison)",
    # "global_hellinger": "Global (Hellinger)",
    # "global_cosine": "Global (Cosine)",
    # "global_spherical": "Global (Spherical)",
    # "markov_bc": "Markov (Bray-Curtis)",
    # "markov_canberra": "Markov (Canberra)",
    # "markov_aitchison": "Markov (Aitchison)",
    "markov_hellinger": "Markov (Hellinger)",
    # "markov_cosine": "Markov (Cosine)",
    # "markov_spherical": "Markov (Spherical)",
    # "local_bc": "Local (Bray-Curtis)",
    # "local_canberra": "Local (Canberra)",
    # "local_aitchison": "Local (Aitchison)",
    "local_hellinger": "Local (Hellinger)",
    # "local_cosine": "Local (Cosine)",
    # "local_spherical": "Local (Spherical)",
    # "autoregressive_bc": "Autoregressive (Bray-Curtis)",
    # "autoregressive_canberra": "Autoregressive (Canberra)",
    # "autoregressive_aitchison": "Autoregressive (Aitchison)",
    # "autoregressive_hellinger": "Autoregressive (Hellinger)",
    # "autoregressive_cosine": "Autoregressive (Cosine)",
    # "autoregressive_spherical": "Autoregressive (Spherical)",
    "locf": "LOCF",
    "complete": "Complete Cases",
    # "original": "Oracle",
}
# ----------------------------------------------------------------------------------------------------------------------




# # ======================================================================================================================
# # DIFFERENTIAL ABUNDANCE ANALYSIS
results = list()

# run trough all methods
for method_name, method_display in methods.items():
    daa = pd.read_csv(DIR_RESULTS + method_name + ".da", index_col=0)
    # daa["differential"] = daa["differential"].fillna(False)
    daa["method"] = method_display
    daa["method_name"] = method_name
    results.append(daa.reset_index())

results = pd.concat(results, ignore_index=True)
# results["differential"] = results["p_value"] < 0.05

df = results.pivot(index=["OTU", "Week"], columns="method", values="differential")
# subset to only those OTUs that are differentially abundant in at least one method
df = df[df.any(axis=1)]

# # ----------------------------------------------------------------------------------------------------------------------

