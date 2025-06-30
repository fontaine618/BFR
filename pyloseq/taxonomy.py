import pandas as pd


def unclassified_to_nan(taxo: pd.DataFrame, string: str = "unclassified") -> pd.DataFrame:
    """
    Convert 'unclassified' entries in the taxonomy DataFrame to NaN.

    :param taxo: DataFrame containing taxonomy information.
    :return: DataFrame with 'unclassified' entries replaced by NaN.
    """
    # change all entries containing the string 'unclassified' to NaN
    taxo = taxo.map(lambda x: pd.NA if isinstance(x, str) and string in x.lower() else x)
    return taxo


def best_taxonomy(taxo: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best taxonomy for each OTU by keeping the most specific classification.

    :param taxo: DataFrame containing taxonomy information.
    :return: DataFrame with the best taxonomy for each OTU.
    """

    # Keep only the most specific classification for each OTU and output the level of specificity
    def fn(x):
        # Find the last non-NaN value in the row
        last_non_nan = x.last_valid_index()
        if last_non_nan is not None:
            # Return the value at the first non-NaN index
            return x[last_non_nan], last_non_nan
        else:
            # If all values are NaN, return NaN
            return pd.NA, pd.NA
    taxo = taxo.apply(fn, axis=1).to_frame()
    # split the tuple into two columns
    taxo = pd.DataFrame({
        "taxon": taxo[0].apply(lambda x: x[0] if isinstance(x, tuple) else x),
        "level": taxo[0].apply(lambda x: x[1] if isinstance(x, tuple) else pd.NA)
    }, index=taxo.index)

    return taxo