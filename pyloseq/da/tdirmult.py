from pyloseq.pyloseq import Pyloseq
import pandas as pd
import numpy as np
from skbio.stats.composition import dirmult_ttest


def tdirmult(
        pyloseq_obj: Pyloseq,
        groupby: str = "Week",
):
    results = []
    for g in pyloseq_obj.sample_data[groupby].unique().tolist():
        psg = pyloseq_obj.subset_samples(inplace=False, **{groupby: g})
        res = dirmult_ttest(psg.otu_table, psg.sample_data["Genotype"], treatment="KO", reference="WT", p_adjust="bh")
        res[groupby] = g
        res.reset_index(inplace=True)
        res.rename(columns={"index": "OTU"}, inplace=True)
        results.append(res)
    # concatenate results
    results_df = pd.concat(results, axis=0)
    # choose columns to return
    results_df = results_df[["OTU", groupby, "pvalue", "Log2(FC)", "qvalue"]]
    results_df.rename(columns={"pvalue": "p_value", "Log2(FC)": "estimate_z", "qvalue": "p_value_adj"}, inplace=True)
    return results_df