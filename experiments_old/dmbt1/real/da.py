from pyloseq.pyloseq import Pyloseq
from pyloseq.da.zinb import zinb
from pyloseq.da.clrt import clrt
from pyloseq.da.lm import lm
from pyloseq.da.tdirmult import tdirmult
from statsmodels.stats.multitest import multipletests

import warnings

# ======================================================================================================================
# SETTINGS
DIR_RESULTS = "/experiments_old/dmbt1/real/results/"




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
    # "markov_hellinger": "Markov (Hellinger)",
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

# run trough all methods
for method_name, method_display in methods.items():
    print(f"Running differential abundance analysis for {method_name}...")
    # load data
    ps = Pyloseq.load(DIR_RESULTS + method_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # daa = lm(ps, groupby="Week")
        # daa = zinb(ps, groupby="Week", xvar="KO", ref_level=0)
        daa = tdirmult(ps, groupby="Week")

        daa = daa.dropna(subset=["p_value"])
        # daa["p_value_adj"] = multipletests(
        #     daa["p_value"], method="fdr_bh", alpha=0.05,
        # )[1]
        daa["differential"] = daa["p_value_adj"] < 0.05
        daa.set_index("OTU", inplace=True)
        daa.to_csv(DIR_RESULTS + f"{method_name}.da")
# # ----------------------------------------------------------------------------------------------------------------------
