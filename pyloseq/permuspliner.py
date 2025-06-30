"""
Implementation of different statistical analysis tools that are used to compare the significant difference between two trajectories.
(1) Splinectomy longitudinal statistical analysis tools
References:
- https://github.com/RRShieldsCutler/splinectomeR
- https://www.frontiersin.org/articles/10.3389/fmicb.2018.00785/full

The following 3 methods below are translated from R to Python and acomodated for our project.
Original package is called splinectomeR, implemented in R. For more details please use the reference links.
In short, the test compares whether the area between two polynomial lines is significantly different.
Our trajectory lines are the plynomial lines with degree 2 or 3.

(2) Linear regression statistical analysis tools
After the 3 methods from splinectomeR, we have the other one, based on comparing the two linear lines (line y=k*x+n).
To compare two linear lines, we compare the significant difference in the two coefficients that represent the line k and n.
In the microbiome trajectory, the only linear part is the part where infant is younger than 2 years. After that, the
trajectory plateaus, therefore, this test can be used only before 2 years.

Taken from microbiome-toolbox
- https://github.com/JelenaBanjac/microbiome-toolbox/blob/main/microbiome/statistical_analysis.py
"""
import numpy as np
import pandas as pd


def permuspliner(
    data,
    xvar,
    yvar,
    category,
    degree,
    cases,
    groups=None,
    perms=999,
    test_direction="more",
    ints=1000,
    quiet=False,
    cut_sparse=4,
):
    """Permuspliner tests for a significant difference between two groups overall.

    Permutation to test whether there is a non-zero trend among a set
    of individuals/samples over a continuous variable -- such as time. So, there
    does not need to be two groups in this test. The x variable datapoints are
    permuated within each case/individual, thus maintaining the distribution in
    the y component but shuffling the hypothesized trend.

    Note: the description is taken from splinectomeR

    Parameters
    ----------
    data: pd.DataFrame
        A dataframe object containing your data.
    xvar: str
        The independent variable; is continuous, e.g. y.
    yvar: str
        The dependent variable; is continuous, e.g. y_pred.
    category: str
        The column name of the category to be tested, e.g. country.
    cases:
        The column name defining the individual cases, e.g. subjectID.
    groups: dict
        If more than two groups, the two groups to compare as character vector, e.g. ["RUS", "FIN"].
    perms: int
        The number of permutations to generate
    cut_sparse: int
        Minimum number of total observations necessary per group to fit a spline (default 4)
    ints: int
        Number of x intervals over which to measure area
    quiet: bool
        Silence all text outputs

    Returns
    -------
    result: dict
        Returns a bunc of stuff, but the most important for you is pval.
    """
    perm_output = {}

    equation = lambda a, b: np.polyval(a, b)

    if test_direction not in ["more", "less"]:
        raise Exception(
            "Error in test direction option: must be either 'more' or 'less'"
        )

    in_df = data.copy()

    # Determine the two groups to compare
    if groups is None:
        if len(in_df.category.unique()) > 2:
            raise Exception(
                'More than two groups in category column. Define groups with (groups = c("Name1","Name2"))'
            )
        else:
            cats = in_df.category.unique()
            v1 = cats[0]
            v2 = cats[1]
    else:
        v1 = groups[0]
        v2 = groups[1]

    if quiet == False:
        print(f"\nGroups detected:", v1, "and", v2, ".\n")
        print(
            f"\nNow testing between variables",
            v1,
            "and",
            v2,
            "for a difference in the response labeled",
            yvar,
            "\n",
        )
        print(
            f"\nScalpel please: performing permusplinectomy with",
            perms,
            "permutations...\n",
        )

    # The experimentally reported response
    df_v1 = in_df[(in_df[category] == v1) & (~in_df[xvar].isna())]
    df_v2 = in_df[(in_df[category] == v2) & (~in_df[xvar].isna())]

    if len(df_v1[xvar]) < cut_sparse or len(df_v2[xvar]) < cut_sparse:
        raise Exception("Not enough data in each group to fit spline")

    # Prevent issues arising from identical case labels across groups
    if len(list(set(df_v1[cases]).intersection(set(df_v2[cases])))) > 0:
        raise Exception(
            "\nIt appears there may be identically labeled cases in both groups.\n...Please ensure that the cases are uniquely labeled between the two groups\n"
        )

    # Fit the splines for each group
    p_v1, _ = np.polyfit(
        df_v1[xvar], df_v1[yvar], degree, cov=True
    )  # parameters and covariance from of the fit of 1-D polynom.
    p_v2, _ = np.polyfit(
        df_v2[xvar], df_v2[yvar], degree, cov=True
    )  # parameters and covariance from of the fit of 1-D polynom.

    x0 = max(min(df_v1[xvar]), min(df_v2[xvar]))
    x1 = min(max(df_v1[xvar]), max(df_v2[xvar]))
    x0 = x0 + (
        (x1 - x0) * 0.1
    )  # Trim the first and last 10% to avoid low-density artifacts
    x1 = x1 - ((x1 - x0) * 0.1)
    xby = (x1 - x0) / (ints - 1)
    xx = np.arange(x0, x1, xby)  # Set the interval range

    # var1 , x
    v1_spl_f = pd.DataFrame(data={"var1": equation(p_v1, xx), "x": xx})
    v2_spl_f = pd.DataFrame(data={"var2": equation(p_v2, xx), "x": xx})

    real_spl_dist = (
        v1_spl_f.set_index("x").join(v2_spl_f.set_index("x"), on="x").reset_index()
    )
    real_spl_dist["abs_dist"] = np.abs(
        real_spl_dist.var1 - real_spl_dist.var2
    )  # Measure the real group distance
    real_area = (
        np.sum(real_spl_dist.abs_dist) / ints
    )  # Calculate the area between the groups

    if quiet == False:
        print(
            "\nArea between groups successfully calculated, now spinning up permutations...\n"
        )

    # Define the permutation function
    case_shuff = "case_shuff"  # Dummy label

    def spline_permute(randy, ix, perm_output):
        randy_meta = randy.drop_duplicates(subset=cases, keep="first").copy(
            deep=False
        )  # Pull out the individual IDs
        randy_meta[case_shuff] = np.random.choice(
            randy_meta[category].values,
            size=len(randy_meta[category].values),
            replace=True,
            p=None,
        )
        randy_meta = randy_meta[[cases, case_shuff]]
        randy = (
            randy.set_index(cases)
            .join(randy_meta.set_index(cases), on=cases)
            .reset_index()
        )

        randy_v1 = randy[(randy[case_shuff] == v1) & (~randy[xvar].isna())]
        randy_v2 = randy[(randy[case_shuff] == v2) & (~randy[xvar].isna())]

        # Fit the splines for the permuted groups
        p_randy_v1, _ = np.polyfit(randy_v1[xvar], randy_v1[yvar], degree, cov=True)
        p_randy_v2, _ = np.polyfit(randy_v2[xvar], randy_v2[yvar], degree, cov=True)

        # var1 , x
        randy_v1_fit = pd.DataFrame(data={"var1": equation(p_randy_v1, xx), "x": xx})
        randy_v2_fit = pd.DataFrame(data={"var2": equation(p_randy_v2, xx), "x": xx})

        spl_dist = (
            randy_v1_fit.set_index("x")
            .join(randy_v2_fit.set_index("x"), on="x")
            .reset_index()
        )
        spl_dist["abs_dist"] = np.abs(
            spl_dist.var1 - spl_dist.var2
        )  # Measure the real group distance

        real_area = (
            np.sum(real_spl_dist.abs_dist) / ints
        )  # Calculate the area between the groups

        transfer_perms = spl_dist[["var1", "var2", "abs_dist"]]
        transfer_perms = transfer_perms.rename(
            columns={
                "var1": f"v1perm_{ix}",
                "var2": f"v2perm_{ix}",
                "abs_dist": f"pdistance_{ix}",
            }
        )
        if ix > 0:
            perm_retainer = perm_output["perm_retainer"]
        else:
            perm_retainer = pd.DataFrame({"x": xx})
        perm_retainer = pd.concat([perm_retainer, transfer_perms], axis=1)
        perm_output["perm_retainer"] = perm_retainer
        perm_area = [
            np.nansum(spl_dist["abs_dist"]) / np.sum(~spl_dist["abs_dist"].isna())
        ]  # Calculate the area between permuted groups
        if ix > 0:
            permuted = perm_output["permuted"]
        else:
            permuted = []
        permuted = np.concatenate([permuted, perm_area])
        perm_output["permuted"] = permuted
        return perm_output

    # Run the permutation over desired number of iterations
    in_rand = pd.concat([df_v1, df_v2])

    pval = None

    for ix in range(perms):
        perm_output = spline_permute(in_rand, ix, perm_output)

    if quiet == False:
        print("...permutations completed...\n")

    if test_direction == "more":
        pval = (np.sum(perm_output["permuted"] >= real_area) + 1) / (perms + 1)

    elif test_direction == "less":
        pval = (sum(perm_output["permuted"] <= real_area) + 1) / (perms + 1)

    if quiet == False:
        print("p-value: ", pval)

    result = {
        "pval": pval,
        "category_1": v1,
        "category_2": v2,
        "v1_interpolated": v1_spl_f,
        "v2_interpolated": v2_spl_f,
        # "v1_spline": df_v1_spl, "v2_spline": df_v2_spl,
        "permuted_splines": perm_output["perm_retainer"],
        "true_distance": real_spl_dist,
        "v1_data": df_v1,
        "v2_data": df_v2,
    }

    return result

