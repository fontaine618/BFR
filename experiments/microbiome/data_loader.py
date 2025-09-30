import pandas as pd
import numpy as np

AVAILABLE_STUDIES = ["MetaCardis_2020_a", "QinJ_2012", "KarlssonFH_2013"]


class T2D:

    def __init__(
            self,
            meta_path: str = 'experiments/microbiome/t2d_meta.csv',
            rel_path: str = 'experiments/microbiome/t2d_rel.csv',
    ):
        self._meta = pd.read_csv(meta_path, index_col=1)
        self._rel = pd.read_csv(rel_path, index_col=0)
        # make sure the samples are aligned
        indices = self._meta.index.intersection(self._rel.index)
        self._meta = self._meta.loc[indices]
        self._rel = self._rel.loc[indices]
        self._rel = self._rel.div(self._rel.sum(axis=1), axis=0).fillna(0)

    def filter_rare_taxa(
            self,
            min_prevalence: float = 0.05,
    ):
        """Filter taxa that are present in less than `min_prevalence` fraction of samples."""
        prevalence = (self._rel > 0).sum(axis=0) / self._rel.shape[0]
        to_keep = prevalence[prevalence >= min_prevalence].index
        dropped = self._rel.columns.difference(to_keep)
        self._rel = self._rel.copy()
        if len(dropped) > 0:
            self._rel['Other'] = self._rel[dropped].sum(axis=1)
        self._rel = self._rel[to_keep.tolist() + (['Other'] if len(dropped) > 0 else [])]
        self._rel = self._rel.div(self._rel.sum(axis=1), axis=0).fillna(0)

    def filter_study(
            self,
            study: str,
    ):
        """Filter samples to only include those from the specified study."""
        if study not in AVAILABLE_STUDIES:
            raise ValueError(f"Study '{study}' not available. Choose from {AVAILABLE_STUDIES}.")
        self._meta = self._meta[self._meta['study_name'] == study]
        self._rel = self._rel.loc[self._meta.index]

    def assign_train_test(
            self,
            test_size: float = 0.5,
            random_state: int = 0,
    ):
        """Assign samples to train and test sets."""
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(self._meta.index)
        test_set_size = int(len(shuffled_indices) * test_size)
        test_indices = shuffled_indices[:test_set_size]
        self._meta['set'] = 'train'
        self._meta.loc[test_indices, 'set'] = 'test'
        self._meta['set'] = self._meta['set'].astype('category')
        self._meta['set'] = self._meta['set'].cat.reorder_categories(['train', 'test'])

    def filter_set(
            self,
            set_name: str = 'train',
    ):
        """Filter samples to only include those in the specified set (train or test)."""
        if set_name not in ['train', 'test']:
            raise ValueError("set_name must be either 'train' or 'test'.")
        if 'set' not in self._meta.columns:
            raise ValueError("No set assignment found. Please run assign_train_test() first.")
        self._meta = self._meta[self._meta['set'] == set_name]
        self._rel = self._rel.loc[self._meta.index]
        self._meta = self._meta.drop(columns=['set'])

    @property
    def relative_abundance(self) -> pd.DataFrame:
        return self._rel

    @property
    def features(self) -> pd.DataFrame:
        """['study_name', 'subject_id', 'body_site', 'antibiotics_current_use',
       'study_condition', 'disease', 'age', 'gender', 'country', 'BMI', 'T2D']"""
        df = self._meta[["antibiotics_current_use", "T2D", "age", "gender", "BMI"]].copy()

        # dummy coding for categorical variables

        df["antibiotics"] = (df["antibiotics_current_use"] == 'yes').astype(float)
        df.loc[df["antibiotics_current_use"].isna(), "antibiotics"] = np.nan
        df = df.drop(columns=["antibiotics_current_use"])

        df["female"] = (df["gender"] == "female").astype(float)
        df.loc[df["gender"].isna(), "female"] = np.nan
        df = df.drop(columns=["gender"])

        # impute missing values with median/mode
        df["age"] = df["age"].fillna(df["age"].median())
        df["BMI"] = df["BMI"].fillna(df["BMI"].median())
        df["T2D"] = df["T2D"].fillna(df["T2D"].median())
        df["antibiotics"] = df["antibiotics"].fillna(df["antibiotics"].median())
        df["female"] = df["female"].fillna(df["female"].median())
        return df







