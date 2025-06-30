from pyloseq.pyloseq import Pyloseq
import pandas as pd
import numpy as np
import scipy
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP


def zinb(
        pyloseq_obj: Pyloseq,
        groupby: str = "Week",
        xvar: str = "Genotype",
        ref_level: str = "WT",
):
    results = []
    for g in pyloseq_obj.sample_data[groupby].unique().tolist():
        dmbt1w = pyloseq_obj.subset_samples(inplace=False, **{groupby: g})
        for otu in pyloseq_obj.otu_table.columns:
            df = pd.DataFrame({
                "count": dmbt1w.otu_table[otu].values,
                "depth": dmbt1w.otu_table.sum(axis=1).values,
                groupby: dmbt1w.sample_data[xvar].values,
            })
            X = np.array([
                np.ones(df.shape[0]),  # Intercept
                df[groupby] != ref_level,  #Group indicator
            ]).T.astype(float)

            try:
                glm1 = ZeroInflatedNegativeBinomialP(
                    endog=df["count"].values,
                    exog=X,
                    exog_infl=X,
                    exposure=df["depth"].values,
                )
                fit1 = glm1.fit(maxiter=100, disp=0, warn_convergence=False)

                glm0 = ZeroInflatedNegativeBinomialP(
                    endog=df["count"].values,
                    exog=X[:, [0]],  # Intercept only
                    exog_infl=X[:, [0]],  # Intercept only
                    exposure=df["depth"].values,
                )
                fit0 = glm0.fit(maxiter=100, disp=0, warn_convergence=False)
            except Exception as e:
                print(f"Error fitting model for OTU {otu} for group {g}: {e}")
                results.append((otu, g, 1., np.nan, np.nan))
                continue

            lr = 2 * (fit1.llf - fit0.llf)
            # p value is chi squared 2
            p_value = scipy.stats.chi2.sf(lr, df=2)
            estimate_nz = fit1.params[3]  # Estimate for the non-zero part
            estimate_z = fit1.params[1]  # Estimate for the zero part
            out = otu, g, p_value, estimate_nz, estimate_z
            results.append(out)
    results_df = pd.DataFrame(results, columns=["OTU", groupby, "p_value", "estimate_nz", "estimate_z"])
    return results_df