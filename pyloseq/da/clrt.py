from pyloseq.pyloseq import Pyloseq
import pandas as pd
import numpy as np
import scipy


def clrt(
        pyloseq_obj: Pyloseq,
        groupby: str = "Week",
        xvar: str = "Genotype",
        ref_level: str = "WT",
):
    results = []
    for g in pyloseq_obj.sample_data[groupby].unique().tolist():
        psg = pyloseq_obj.subset_samples(inplace=False, **{groupby: g})
        psg.transform_otu("clr", inplace=True)  # Transform OTU table to CLR
        for otu in pyloseq_obj.otu_table.columns:
            df = pd.DataFrame({
                "clr": psg.otu_table[otu].values,
                xvar: psg.sample_data[xvar].values,
            })
            # run t test
            try:
                group1 = df[df[xvar] == ref_level]["clr"]
                group2 = df[df[xvar] != ref_level]["clr"]
                t_stat, p_value = scipy.stats.ttest_ind(group1, group2, equal_var=False)
            except Exception as e:
                print(f"Error fitting model for OTU {otu} for group {g}: {e}")
                results.append((otu, g, 1., np.nan, np.nan))
                continue
            estimate_nz = group1.mean() - group2.mean()  # Estimate for the non-zero part
            out = otu, g, p_value, estimate_nz, np.nan
            results.append(out)
    results_df = pd.DataFrame(results, columns=["OTU", groupby, "p_value", "estimate_nz", "estimate_z"])
    return results_df