from pyloseq.pyloseq import Pyloseq
import pandas as pd
import numpy as np
import statsmodels.api as sm


def lm(
        pyloseq_obj: Pyloseq,
        groupby: str = "Week",
):
    results = []
    for g in pyloseq_obj.sample_data[groupby].unique().tolist():
        psg = pyloseq_obj.subset_samples(inplace=False, **{groupby: g})
        psg.transform_otu("clr", inplace=True)  # Transform OTU table to CLR
        for otu in pyloseq_obj.otu_table.columns:
            df = pd.DataFrame({
                "clr": psg.otu_table[otu].values,
                "KO": psg.sample_data["KO"].values,
                "SCC": psg.sample_data["SCC"].values,
                "F": psg.sample_data["F"].values,
            })
            # linear regression model clr ~ KO + SCC + F
            model = sm.OLS.from_formula("clr ~ KO + SCC + F", data=df)
            results_model = model.fit()
            estimate = results_model.params["SCC"]
            # get p-value for KO
            p_value = results_model.pvalues["SCC"]
            out = otu, g, p_value, estimate, np.nan
            results.append(out)
    results_df = pd.DataFrame(results, columns=["OTU", groupby, "p_value", "estimate_nz", "estimate_z"])
    return results_df