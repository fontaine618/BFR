from typing import Callable

import numpy as np
import pandas as pd


class Pyloseq:

    def __init__(
            self,
            otu_table: pd.DataFrame,  # N x K
            sample_data: pd.DataFrame,  # N x P
            tax_table: pd.DataFrame,  # K x D
    ):
        """
        Initialize the Pyloseq object with OTU table, sample data, and taxonomic table.

        :param otu_table: OTU table (N x K), where N is the number of samples, K is the number of OTUs.
        :param sample_data: Sample data (N x P), where N is the number of samples, P is the number of features.
        :param tax_table: Taxonomic table (K x D), where K is the number of OTUs, D is the number of taxonomic ranks.
        """
        self.otu_table = otu_table
        self.sample_data = sample_data
        self.tax_table = tax_table
        # check that the indices of otu_table and sample_data match
        self.otu_table.sort_index(inplace=True)
        self.sample_data.sort_index(inplace=True)
        if not self.otu_table.index.equals(self.sample_data.index):
            raise ValueError("Indices of otu_table and sample_data do not match.")
        # check that the indices of tax_table include all columns of otu_table, drop the ones that are not in otu_table
        self.tax_table = self.tax_table.loc[self.otu_table.columns]
        if not self.tax_table.index.equals(self.otu_table.columns):
            raise ValueError("Indices of tax_table do not match columns of otu_table.")

    def __repr__(self):
        """
        String representation of the Pyloseq object.
        """
        return f"Pyloseq(otu_table={self.otu_table.shape}, sample_data={self.sample_data.shape}, tax_table={self.tax_table.shape})"

    def aggregate_taxa(self, depth=str | int, inplace=True) -> "Pyloseq":
        """
        Aggregate OTU table by taxonomic rank (summation).

        :param depth: Taxonomic rank to aggregate by (e.g., "Phylum", "Class", "Order", "Family", "Genus", "Species", or column index).
        """
        if isinstance(depth, str):
            if depth not in self.tax_table.columns:
                raise ValueError(f"Taxonomic rank '{depth}' not found in tax_table columns.")
            taxa = self.tax_table[depth]
        elif isinstance(depth, int):
            if depth < 0 or depth >= self.tax_table.shape[1]:
                raise IndexError("Depth index out of bounds for tax_table.")
            taxa = self.tax_table.iloc[:, depth]
        else:
            raise TypeError("Depth must be a string (taxonomic rank) or an integer (column index).")

        # Aggregate OTU table by summing over the specified taxonomic rank
        agg_otu_table = self.otu_table.T
        agg_otu_table["taxa"] = taxa
        agg_otu_table = agg_otu_table.groupby("taxa").sum().T
        # drop columns of tax_table after the specified depth
        if isinstance(depth, str):
            depth_index = self.tax_table.columns.get_loc(depth)
        else:
            depth_index = depth
        tax_table = self.tax_table.iloc[:, :depth_index + 1]

        if inplace:
            # Update the OTU table with aggregated values
            self.otu_table = agg_otu_table
            self.tax_table = tax_table
            return self
        else:
            # Return a new Pyloseq object with aggregated values
            return Pyloseq(otu_table=agg_otu_table, sample_data=self.sample_data.copy(), tax_table=tax_table)

    def transform_otu(self, transform: Callable | str, inplace=True, **kwargs) -> "Pyloseq":
        """
        Apply a transformation function to the OTU table (row-wise).

        :param transform: A callable function to apply to each element of the OTU table or a string specifying a transformation type.
        :param kwargs: Additional keyword arguments for the transformation function. See note below.

        Supported transformations:
        - "rel" or "relative": Convert counts to relative abundances.
        - "clr" or "centered_log_ratio": Convert counts to centered log-ratio (CLR) transformation.

        Note: For CLR transformation, you can specify a 'shift' value in kwargs to avoid log(0) issues.
        """
        if isinstance(transform, str):
            if transform == "rel" or transform == "relative":
                # Convert counts to relative abundances
                otu_table = self.otu_table.div(self.otu_table.sum(axis=1), axis=0)
            elif transform == "clr" or transform == "centered_log_ratio":
                # Convert counts to centered log-ratio (CLR) transformation
                if "shift" not in kwargs:
                    shift = 0.5 * self.otu_table[self.otu_table > 0].min().min()
                else:
                    shift = kwargs["shift"]
                otu_table = self.otu_table.replace(0, shift)
                otu_table = otu_table.apply(lambda x: np.log(x), axis=1)
                otu_table = otu_table.sub(otu_table.mean(axis=1), axis=0)
            else:
                raise ValueError(f"Unknown transformation function '{transform}'. Supported: 'rel', 'clr'.")
        elif callable(transform):
            # Apply the custom function to each element of the OTU table
            otu_table = self.otu_table.map(transform)
        else:
            raise TypeError("Transform must be a callable function or a string specifying a transformation type.")

        if inplace:
            self.otu_table = otu_table
            return self
        else:
            return Pyloseq(otu_table=otu_table, sample_data=self.sample_data.copy(), tax_table=self.tax_table.copy())

    def filter_rare(self, detection: float = 0.01, prevalence: float = 0.1,
                    bin_taxa: bool = False, inplace=True) -> "Pyloseq":
        """
        Filter out rare OTUs based on detection and prevalence thresholds.

        :param detection: Minimum detection threshold (p/a).
        :param prevalence: Minimum prevalence threshold (fraction of samples).
        :param bin_taxa: If True, bin taxa that do not meet the thresholds into a single "Other" category.
        """
        prop = (self.otu_table.map(lambda x: x>=detection)).sum(axis=0) / self.otu_table.shape[0]
        keep = prop >= prevalence
        if not keep.any():
            raise ValueError("No OTUs meet the detection and prevalence thresholds.")
        if bin_taxa:
            # Bin rare taxa into a single "Other" category
            other_taxa = self.otu_table.loc[:, ~keep].sum(axis=1)
            otu_table = self.otu_table.loc[:, keep].copy()
            otu_table["Other"] = other_taxa
            tax_table = self.tax_table.loc[self.otu_table.columns]
            tax_table.loc["Other"] = ["Other"] * tax_table.shape[1]
        else:
            # Keep only the taxa that meet the thresholds
            otu_table = self.otu_table.loc[:, keep].copy()
            tax_table = self.tax_table.loc[self.otu_table.columns]

        if inplace:
            self.otu_table = otu_table
            self.tax_table = tax_table
            return self
        else:
            return Pyloseq(otu_table=otu_table, sample_data=self.sample_data.copy(), tax_table=tax_table)

    def _subset_samples_boolean(self, mask: pd.Series, inplace=True) -> "Pyloseq":
        """
        Subset the Pyloseq object to include only samples specified by a boolean mask.

        :param mask: Boolean Series indicating which samples to keep.
        """
        if not isinstance(mask, pd.Series):
            raise TypeError("Mask must be a pandas Series.")
        if mask.index.equals(self.sample_data.index):
            otu_table = self.otu_table.loc[mask]
            sample_data = self.sample_data[mask]
        else:
            raise ValueError("Mask index does not match sample_data index.")

        if inplace:
            self.otu_table = otu_table
            self.sample_data = sample_data
            return self
        else:
            return Pyloseq(otu_table=otu_table, sample_data=sample_data, tax_table=self.tax_table.copy())

    def _subset_samples_query(self, inplace=True, **kwargs) -> "Pyloseq":
        """
        Subset the Pyloseq object to include only specified samples.

        :param kwargs: Keyword arguments to filter samples in sample_data.
        """
        if not kwargs:
            raise ValueError("No filtering criteria provided. Please specify at least one criterion.")

        # Create a boolean mask for filtering samples
        mask = pd.Series([True] * self.sample_data.shape[0], index=self.sample_data.index)
        for key, value in kwargs.items():
            if key not in self.sample_data.columns:
                raise KeyError(f"Column '{key}' not found in sample_data.")
            mask &= (self.sample_data[key] == value)

        if not mask.any():
            raise ValueError("No samples match the specified criteria. Please check your filtering criteria.")
        return self._subset_samples_boolean(mask, inplace=inplace)

    def _subset_samples_index(self, index: list[str], inplace=True) -> "Pyloseq":
        """
        Subset the Pyloseq object to include only samples specified by a list of IDs.

        :param index: List of sample IDs to keep.
        """
        if not isinstance(index, list):
            raise TypeError("IDs must be a list of sample IDs.")
        mask = self.sample_data.index.isin(index)
        return self._subset_samples_boolean(mask, inplace=inplace)

    def subset_samples(self, inplace=True, **kwargs) -> "Pyloseq":
        """
        Subset the Pyloseq object to include only samples that match the specified criteria.

        :param kwargs: Keyword arguments to filter samples in sample_data.
        """
        if "index" in kwargs:
            return self._subset_samples_index(kwargs.pop("index"), inplace=inplace)
        elif "mask" in kwargs:
            return self._subset_samples_boolean(kwargs.pop("mask"), inplace=inplace)
        else:
            return self._subset_samples_query(**kwargs, inplace=inplace)

    def concat(self, obj: "Pyloseq") -> "Pyloseq":
        """
        Join another Pyloseq object to the current one. Requires different sample index.
        Assumes matching taxononmy (though it checks if the number of OTUs match).

        :param obj: Another Pyloseq object to join.
        """
        if not isinstance(obj, Pyloseq):
            raise TypeError("The object to join must be a Pyloseq instance.")

        # Check if the sample indices are all different
        if self.sample_data.index.isin(obj.sample_data.index).any():
            raise ValueError("Sample indices must be different for concat.")

        # Check if the dimensions match for number of features and number of OTUs
        if self.otu_table.shape[1] != obj.otu_table.shape[1]:
            raise ValueError("The number of OTUs must match for concat.")
        if self.sample_data.shape[1] != obj.sample_data.shape[1]:
            raise ValueError("The number of features in sample_data must match for concat.")

        otu_table = pd.concat([self.otu_table, obj.otu_table], axis=0)
        sample_data = pd.concat([self.sample_data, obj.sample_data], axis=0)
        return Pyloseq(otu_table=otu_table, sample_data=sample_data, tax_table=self.tax_table.copy())

        # # Concatenate the OTU tables, sample data
        # self.otu_table = pd.concat([self.otu_table, obj.otu_table], axis=0)
        # self.sample_data = pd.concat([self.sample_data, obj.sample_data], axis=0)
        #
        # # Sort the indices of the concatenated dataframes
        # self.otu_table.sort_index(inplace=True)
        # self.sample_data.sort_index(inplace=True)
        #
        # return self

    def copy(self) -> "Pyloseq":
        """
        Create a copy of the Pyloseq object.
        """
        return Pyloseq(
            otu_table=self.otu_table.copy(),
            sample_data=self.sample_data.copy(),
            tax_table=self.tax_table.copy()
        )

    def save(self, path: str) -> None:
        """
        Save the Pyloseq object to a file.

        :param path: Path to save the Pyloseq object.
        """
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")
        self.otu_table.to_csv(f"{path}.otu")
        self.sample_data.to_csv(f"{path}.meta")
        self.tax_table.to_csv(f"{path}.tax")

    @classmethod
    def load(cls, path: str) -> "Pyloseq":
        """
        Load a Pyloseq object from a file.

        :param path: Path to the Pyloseq object file.
        """
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")
        otu_table = pd.read_csv(f"{path}.otu", index_col=0)
        sample_data = pd.read_csv(f"{path}.meta", index_col=0)
        tax_table = pd.read_csv(f"{path}.tax", index_col=0)
        return cls(otu_table=otu_table, sample_data=sample_data, tax_table=tax_table)