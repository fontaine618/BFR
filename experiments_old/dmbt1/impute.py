from pyloseq.pyloseq import Pyloseq
import pandas as pd
import numpy as np
import torch
import os
import utils.squared_metrics.compositional as c
from bfr.kulsif import kulsif
from bfr.bgfr import BGFR, GFR
from nlbgfr import NLGFR
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.kernels.laplacian import laplacian_kernel
from utils.squared_metrics.euclidean import squared_L2_distance
import pyloseq.taxonomy as tax


# ======================================================================================================================
# IMPUTATION FUNCTIONS

def locf(previous: Pyloseq, current: Pyloseq, **kwargs) -> Pyloseq:
    "Last Observation Carried Forward (LOCF) imputation"
    current = current.copy()
    which_missing = ~current.sample_data["Observed"].values
    current.otu_table.iloc[which_missing, :] = previous.otu_table.iloc[which_missing, :].values
    return current


def global_impute(previous: Pyloseq, current: Pyloseq, **kwargs) -> Pyloseq:
    # Prior data (contains observed + imputed)
    X = torch.Tensor(previous.sample_data[["KO", "SCC", "F"]].values)
    X_prior = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_prior = torch.Tensor(previous.otu_table.values)
    # Observed data
    observed = current.subset_samples(Observed=True, inplace=False)
    X = torch.Tensor(observed.sample_data[["KO", "SCC", "F"]].values)
    X_observed = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_observed = torch.Tensor(observed.otu_table.values)
    # Missing data
    missing = current.subset_samples(Observed=False, inplace=False)
    X = torch.Tensor(missing.sample_data[["KO", "SCC", "F"]].values)
    X_missing = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_missing = torch.Tensor(missing.otu_table.values)
    # Compute distance matrices
    D2x_train_train = squared_L2_distance(X_observed[:, :3], X_observed[:, :3])
    D2x_prior_train = squared_L2_distance(X_observed[:, :3], X_prior[:, :3])
    D2y_train_train = sq_dist(Y_observed, Y_observed)
    D2y_prior_train = sq_dist(Y_observed, Y_prior)
    Y_scale = (torch.median(D2y_train_train[D2y_train_train > 0.])).sqrt() * Y_scale_factor
    # Kernel across training points
    kx = squared_exponential_kernel(D2x_train_train, X_scale)
    ky = laplacian_kernel(D2y_train_train, Y_scale)
    # Kernel between training and prior points
    kx_prior = squared_exponential_kernel(D2x_prior_train, X_scale)
    ky_prior = laplacian_kernel(D2y_prior_train, Y_scale)
    # Density ratio estimation
    rxy_train = kulsif(kx * ky, kx_prior * ky_prior, reg=reg)
    rx_train = kulsif(kx, kx_prior, reg=reg)
    # Prepare weights
    # bgfr = BGFR(X_observed, X_missing, rxy_train=rxy_train, rx_train=rx_train, reg=1e-5)
    bgfr = BGFR(X_observed, X_missing, rxy_train=rxy_train, reg=1e-5)
    # Predict
    logit = torch.nn.Parameter(torch.zeros_like(Y_missing))
    optimizer = torch.optim.Rprop([logit], lr=lr)
    prev_loss = float("inf")
    prev_Y = logit.clone().detach()
    for i in range(max_iter):
        optimizer.zero_grad()
        Y = logit.exp()
        Y = Y / Y.sum(dim=1, keepdim=True)  # normalize to probabilities
        U_trainpred = sq_dist(Y, Y_observed)  # M x N
        ese = bgfr.ese(U_trainpred)
        loss = ese.sum()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            max_change = (Y - prev_Y).abs().max()
            print(f"Iteration {i} - Loss: {loss.item()} - Linf: {max_change.item()}")
        prev_loss = loss.item()
        prev_logit = logit.clone().detach()
    # Impute
    Y_pred = Y.detach().cpu().numpy()
    Y_pred = (library_size * Y_pred).round().astype(int)  # convert to counts
    imputed = current.subset_samples(Observed=False, inplace=False)
    imputed.otu_table = pd.DataFrame(
        Y_pred,
        index=imputed.otu_table.index,
        columns=imputed.otu_table.columns
    )
    return observed.concat(imputed)


def markov_impute(previous: Pyloseq, current: Pyloseq, **kwargs) -> Pyloseq:
    # Prior data (contains observed + imputed)
    X = torch.Tensor(previous.sample_data[["KO", "SCC", "F"]].values)
    X_prior = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_prior = torch.Tensor(previous.otu_table.values)
    # Observed data
    observed = current.subset_samples(Observed=True, inplace=False)
    X = torch.Tensor(observed.sample_data[["KO", "SCC", "F"]].values)
    X_observed = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_observed = torch.Tensor(observed.otu_table.values)
    # Missing data
    missing = current.subset_samples(Observed=False, inplace=False)
    X = torch.Tensor(missing.sample_data[["KO", "SCC", "F"]].values)
    X_missing = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_missing = torch.Tensor(missing.otu_table.values)
    # Compute distance matrices
    D2x_train_train = squared_L2_distance(X_observed[:, :3], X_observed[:, :3])
    D2x_prior_train = squared_L2_distance(X_observed[:, :3], X_prior[:, :3])
    D2y_train_train = sq_dist(Y_observed, Y_observed)
    D2y_prior_train = sq_dist(Y_observed, Y_prior)
    Y_scale = (torch.median(D2y_train_train[D2y_train_train > 0.])).sqrt() * Y_scale_factor
    # Kernel across training points
    kx = squared_exponential_kernel(D2x_train_train, X_scale)
    ky = laplacian_kernel(D2y_train_train, Y_scale)
    # Kernel between training and prior points
    kx_prior = squared_exponential_kernel(D2x_prior_train, X_scale)
    ky_prior = laplacian_kernel(D2y_prior_train, Y_scale)
    ids_missing = current.sample_data.loc[~current.sample_data["Observed"], "SubjectID"].values
    # get index of ids_missing within the current sample_data
    i_missing = [current.sample_data["SubjectID"].values.tolist().index(id) for id in ids_missing]
    Y_pred_list = []
    for j, (i, id) in enumerate(zip(i_missing, ids_missing)):
        print(f"Imputing {j + 1}/{len(i_missing)}: {id}")

        # Density ratio estimation
        rxy_train = kulsif(kx * ky, kx_prior[:, [i]] * ky_prior[:, [i]], reg=reg)
        rx_train = kulsif(kx, kx_prior[:, [i]], reg=reg)
        # Prepare weights
        # bgfr = BGFR(X_observed, X_missing, rxy_train=rxy_train, rx_train=rx_train, reg=1e-5)
        bgfr = BGFR(X_observed, X_missing, rxy_train=rxy_train, rx_train=None, reg=1e-5)
        # Predict
        logit = torch.nn.Parameter(torch.zeros(1, Y_missing.shape[1]))
        optimizer = torch.optim.Rprop([logit], lr=lr)
        prev_loss = float("inf")
        prev_Y = logit.clone().detach()
        for i in range(max_iter):
            optimizer.zero_grad()
            Y = logit.exp()
            Y = Y / Y.sum(dim=1, keepdim=True)  # normalize to probabilities
            U_trainpred = sq_dist(Y, Y_observed)  # M x N
            ese = bgfr.ese(U_trainpred)
            loss = ese.sum()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                max_change = (Y - prev_Y).abs().max()
                print(f"Iteration {i} - Loss: {loss.item()} - Linf: {max_change.item()}")
            prev_loss = loss.item()
            prev_logit = logit.clone().detach()
        # Impute
        Y_pred_list.append(Y.detach().cpu().numpy())
    Y_pred = np.concatenate(Y_pred_list, axis=0)

    Y_pred = (library_size * Y_pred).round().astype(int)  # convert to counts
    imputed = current.subset_samples(Observed=False, inplace=False)
    imputed.otu_table = pd.DataFrame(
        Y_pred,
        index=imputed.otu_table.index,
        columns=imputed.otu_table.columns
    )
    return observed.concat(imputed)


def local_impute(previous: Pyloseq, current: Pyloseq, data: Pyloseq, **kwargs) -> Pyloseq:
    # Prior data
    X = torch.Tensor(data.sample_data[["KO", "SCC", "F"]].values)
    X_prior = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_prior = torch.Tensor(data.otu_table.values)
    sid = data.sample_data["SubjectID"].values
    # Observed data
    observed = current.subset_samples(Observed=True, inplace=False)
    X = torch.Tensor(observed.sample_data[["KO", "SCC", "F"]].values)
    X_observed = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_observed = torch.Tensor(observed.otu_table.values)
    # Missing data
    missing = current.subset_samples(Observed=False, inplace=False)
    X = torch.Tensor(missing.sample_data[["KO", "SCC", "F"]].values)
    X_missing = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_missing = torch.Tensor(missing.otu_table.values)
    # Compute distance matrices
    D2x_train_train = squared_L2_distance(X_observed[:, :3], X_observed[:, :3])
    D2x_prior_train = squared_L2_distance(X_observed[:, :3], X_prior[:, :3])
    D2y_train_train = sq_dist(Y_observed, Y_observed)
    D2y_prior_train = sq_dist(Y_observed, Y_prior)
    Y_scale = (torch.median(D2y_train_train[D2y_train_train > 0.])).sqrt() * Y_scale_factor
    # Kernel across training points
    kx = squared_exponential_kernel(D2x_train_train, X_scale)
    ky = laplacian_kernel(D2y_train_train, Y_scale)
    # Kernel between training and prior points
    kx_prior = squared_exponential_kernel(D2x_prior_train, X_scale)
    ky_prior = laplacian_kernel(D2y_prior_train, Y_scale)
    ids_missing = current.sample_data.loc[~current.sample_data["Observed"], "SubjectID"].values
    i_missing = [current.sample_data["SubjectID"].values.tolist().index(id) for id in ids_missing]
    Y_pred_list = []
    for j, (i, id) in enumerate(zip(i_missing, ids_missing)):
        print(f"Imputing {j + 1}/{len(i_missing)}: {id}")

        which = (sid == id) * (data.sample_data["Observed"].values)  # only use actual observations

        # Density ratio estimation
        rxy_train = kulsif(kx * ky, kx_prior[:, which] * ky_prior[:, which], reg=reg)
        rx_train = kulsif(kx, kx_prior[:, which], reg=reg)
        ry_train = kulsif(ky, ky_prior[:, which], reg=reg)
        # Prepare weights
        # bgfr = BGFR(X_observed, X_missing, rxy_train=rxy_train, rx_train=rx_train, reg=1e-5)
        bgfr = BGFR(X_observed, X_missing, rxy_train=rxy_train, rx_train=None, reg=1e-5)
        # Predict
        logit = torch.nn.Parameter(torch.zeros(1, Y_missing.shape[1]))
        optimizer = torch.optim.Rprop([logit], lr=lr)
        prev_loss = float("inf")
        prev_Y = logit.clone().detach()
        for i in range(max_iter):
            optimizer.zero_grad()
            Y = logit.exp()
            Y = Y / Y.sum(dim=1, keepdim=True)  # normalize to probabilities
            U_trainpred = sq_dist(Y, Y_observed)  # M x N
            ese = bgfr.ese(U_trainpred)
            loss = ese.sum()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                max_change = (Y - prev_Y).abs().max()
                print(f"Iteration {i} - Loss: {loss.item()} - Linf: {max_change.item()}")
            prev_loss = loss.item()
            prev_logit = logit.clone().detach()
        # Impute
        Y_pred_list.append(Y.detach().cpu().numpy())
    Y_pred = np.concatenate(Y_pred_list, axis=0)

    Y_pred = (library_size * Y_pred).round().astype(int)  # convert to counts
    imputed = current.subset_samples(Observed=False, inplace=False)
    imputed.otu_table = pd.DataFrame(
        Y_pred,
        index=imputed.otu_table.index,
        columns=imputed.otu_table.columns
    )
    return observed.concat(imputed)


def gfr(previous: Pyloseq, current: Pyloseq, **kwargs) -> Pyloseq:
    # Observed data
    observed = current.subset_samples(Observed=True, inplace=False)
    X = torch.Tensor(observed.sample_data[["KO", "SCC", "F"]].values)
    X_observed = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_observed = torch.Tensor(observed.otu_table.values)
    # Missing data
    missing = current.subset_samples(Observed=False, inplace=False)
    X = torch.Tensor(missing.sample_data[["KO", "SCC", "F"]].values)
    X_missing = torch.hstack([X, X[:, [0]] * X[:, [1]]])
    Y_missing = torch.Tensor(missing.otu_table.values)
    # Prepare weights
    gfr = GFR(X_observed, X_missing, reg=1e-5)
    # Predict
    logit = torch.nn.Parameter(torch.zeros_like(Y_missing))
    optimizer = torch.optim.Rprop([logit], lr=lr)
    prev_loss = float("inf")
    prev_Y = logit.clone().detach()
    for i in range(max_iter):
        optimizer.zero_grad()
        Y = logit.exp()
        Y = Y / Y.sum(dim=1, keepdim=True)  # normalize to probabilities
        U_trainpred = sq_dist(Y, Y_observed)  # M x N
        ese = gfr.ese(U_trainpred)
        loss = ese.sum()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            max_change = (Y - prev_Y).abs().max()
            print(f"Iteration {i} - Loss: {loss.item()} - Linf: {max_change.item()}")
        prev_loss = loss.item()
        prev_logit = logit.clone().detach()
    # Impute
    Y_pred = Y.detach().cpu().numpy()
    Y_pred = (library_size * Y_pred).round().astype(int)  # convert to counts
    imputed = current.subset_samples(Observed=False, inplace=False)
    imputed.otu_table = pd.DataFrame(
        Y_pred,
        index=imputed.otu_table.index,
        columns=imputed.otu_table.columns
    )
    return observed.concat(imputed)


def autoregressive_impute(previous: Pyloseq, current: Pyloseq, data: Pyloseq, **kwargs) -> Pyloseq:
    # Observed data
    observed = current.subset_samples(Observed=True, inplace=False)
    X = torch.Tensor(observed.sample_data[["KO", "SCC", "F"]].values)
    X_observed = torch.hstack([X, X[:, [0]] * X[:, [1]]]) # No x 3
    Y_observed = torch.Tensor(observed.otu_table.values) # No x K
    # Missing data
    missing = current.subset_samples(Observed=False, inplace=False)
    X = torch.Tensor(missing.sample_data[["KO", "SCC", "F"]].values)
    X_missing = torch.hstack([X, X[:, [0]] * X[:, [1]]]) # Nm x 3
    Y_missing = torch.Tensor(missing.otu_table.values) # Nm x K
    # Previous composition
    Y_prior = torch.Tensor(previous.otu_table.values) # NxK
    ids_missing = current.sample_data.loc[~current.sample_data["Observed"], "SubjectID"].values
    i_missing = ids_missing.astype(int)
    b_missing = ~current.sample_data["Observed"].values
    Y_prior_observed = Y_prior[~b_missing, :]  # No x K
    Y_prior_missing = Y_prior[b_missing, :]  # Nm x K

    # Compute distance matrices
    D2x_train_train = squared_L2_distance(X_observed[:, :3], X_observed[:, :3])  # No x No
    D2x_train_test = squared_L2_distance(X_observed[:, :3], X_missing[:, :3])  # No x Nm
    D2y_train_train = sq_dist(Y_prior_observed, Y_prior_observed)  # No x No
    D2y_prior_train = sq_dist(Y_prior_observed, Y_prior_missing)  # No x Np
    Y_scale = (torch.median(D2y_train_train[D2y_train_train > 0.])).sqrt() * Y_scale_factor
    # Kernel across training points
    kx_train_train = squared_exponential_kernel(D2x_train_train, X_scale)
    ky_train_train = laplacian_kernel(D2y_train_train, Y_scale)
    # Kernel between training and testing points
    kx_train_test = squared_exponential_kernel(D2x_train_test, X_scale)
    ky_train_test = laplacian_kernel(D2y_prior_train, Y_scale)
    # Product kernel
    K_train_train = kx_train_train * ky_train_train
    K_train_test = kx_train_test * ky_train_test

    # Prepare weights
    bgfr = NLGFR(K_train_train, K_train_test.T, reg=reg)
    # Predict
    logit = torch.nn.Parameter(torch.zeros_like(Y_missing))
    optimizer = torch.optim.Rprop([logit], lr=lr)
    prev_loss = float("inf")
    prev_Y = logit.clone().detach()
    for i in range(max_iter):
        optimizer.zero_grad()
        Y = logit.exp()
        Y = Y / Y.sum(dim=1, keepdim=True)  # normalize to probabilities
        U_trainpred = sq_dist(Y, Y_observed)  # M x N
        ese = bgfr.ese(U_trainpred)
        loss = ese.sum()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            max_change = (Y - prev_Y).abs().max()
            print(f"Iteration {i} - Loss: {loss.item()} - Linf: {max_change.item()}")
        prev_loss = loss.item()
        prev_logit = logit.clone().detach()
    # Impute
    Y_pred = Y.detach().cpu().numpy()
    Y_pred = (library_size * Y_pred).round().astype(int)  # convert to counts
    imputed = current.subset_samples(Observed=False, inplace=False)
    imputed.otu_table = pd.DataFrame(
        Y_pred,
        index=imputed.otu_table.index,
        columns=imputed.otu_table.columns
    )
    return observed.concat(imputed)
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# METHODS
experiments = [
    (global_impute, "Global (Aitchison)", "global_aitchison", c.squared_aitchison_distance, 0.001),
    (markov_impute, "Markov (Aitchison)", "markov_aitchison", c.squared_aitchison_distance, 0.1),
    (local_impute, "Local (Aitchison)", "local_aitchison", c.squared_aitchison_distance, 0.1),
    (gfr, "GFR (Aitchison)", "gfr_aitchison", c.squared_aitchison_distance, 0.001),
    (global_impute, "Global (Hellinger)", "global_hellinger", c.squared_hellinger_distance, 0.001),
    (markov_impute, "Markov (Hellinger)", "markov_hellinger", c.squared_hellinger_distance, 0.1),
    (local_impute, "Local (Hellinger)", "local_hellinger", c.squared_hellinger_distance, 0.1),
    (gfr, "GFR (Hellinger)", "gfr_hellinger", c.squared_hellinger_distance, 0.001),
    (global_impute, "Global (Spherical)", "global_spherical", c.squared_spherical_norm_distance, 0.001),
    (markov_impute, "Markov (Spherical)", "markov_spherical", c.squared_spherical_norm_distance, 0.1),
    (local_impute, "Local (Spherical)", "local_spherical", c.squared_spherical_norm_distance, 0.1),
    (gfr, "GFR (Spherical)", "gfr_spherical", c.squared_spherical_norm_distance, 0.001),
    (locf, "LOCF", "locf", None, None)
]

# ----------------------------------------------------------------------------------------------------------------------


for SEED in range(100):
    # ======================================================================================================================
    # SETTINGS
    DIR_RESULTS = f"/experiments_old/dmbt1/results/{SEED}/"
    prop_missing = 0.2
    os.makedirs(DIR_RESULTS, exist_ok=True)

    X_scale = 0.7
    Y_scale_factor = 0.7
    reg = 0.001

    max_iter = 100
    lr = 0.05
    library_size = 9484
    # ----------------------------------------------------------------------------------------------------------------------


    # ======================================================================================================================
    # DATA
    DIRDATA = "/home/simon/Documents/BFR/experiments/dmbt1/data/"

    samples = pd.read_csv(DIRDATA + "meta.csv", index_col=None)
    demo = pd.read_csv(DIRDATA + "demo.csv", index_col=None)
    mice = pd.read_csv(DIRDATA + "mice.csv", index_col=None)

    samples[["SubjectID", "Week"]] = samples["Group"].str.split("_", expand=True)
    samples["Week"] = samples["Week"].apply(lambda x: "0w" if x is None else x)
    samples["Week"] = samples["Week"].str.replace("w", "").astype(int)
    samples = samples.rename(columns={"Diagnosis2": "Diagnosis", "Gender": "Sex"})
    # uniformize sample IDs
    samples["SubjectID"] = samples["SubjectID"].astype(str).str.zfill(4)
    samples["SampleID"] = samples["SubjectID"] + "_" + samples["Week"].astype(str).str.zfill(2)
    samples = samples.set_index("SampleID")
    # samples.drop(["Group"], axis=1, inplace=True)

    taxo = pd.read_csv(DIRDATA + "taxo.csv", index_col=0)
    taxo = taxo.drop(["Size"], axis=1)

    taxo = tax.unclassified_to_nan(taxo)

    otu = pd.read_csv(DIRDATA + "otu.csv", index_col=1)
    otu = otu.drop(["label", "numOtus"], axis=1)
    otu.sort_index(inplace=True)
    otu.index = samples.sort_values("Group").index

    data = Pyloseq(
        otu_table=otu,
        sample_data=samples,
        tax_table=taxo
    )
    best_tax = tax.best_taxonomy(data.tax_table)
    data = data.filter_rare(detection=1, prevalence=0.05, bin_taxa=False)



    # ----------------------------------------------------------------------------------------------------------------------




    # ======================================================================================================================
    # TESTING DATA
    week4p = data.sample_data["Week"].gt(0)
    np.random.seed(SEED)
    missing = np.random.uniform(size=data.sample_data.shape[0]) < prop_missing
    data.sample_data["Testing"] = week4p * missing
    data.save(f"{DIR_RESULTS}/original") # save indicators
    # Mask testing data
    data.otu_table.loc[data.sample_data["Testing"]] = np.nan
    # ----------------------------------------------------------------------------------------------------------------------




    # ======================================================================================================================
    # ADD MISSING DATA
    meta = samples[["SubjectID", "Genotype", "Diagnosis", "Sex"]]
    meta = meta.drop_duplicates().reset_index(drop=True)
    subject_ids = data.sample_data["SubjectID"].unique()
    timepoints = [0, 4, 8, 12, 16, 22]
    all_samples = pd.DataFrame({
        "SubjectID": [sid for sid in subject_ids for _ in timepoints],
        "Week": [tp for _ in subject_ids for tp in timepoints],
    })
    all_samples["SampleID"] = all_samples["SubjectID"] + "_" + all_samples["Week"].astype(str).str.zfill(2)
    # merge in meta
    all_samples = all_samples.merge(meta, on="SubjectID", how="left")
    all_samples = all_samples.set_index("SampleID")
    all_samples["Missing"] = np.isin(all_samples.index.values, data.sample_data.index.values, invert=True)
    missing = all_samples[all_samples["Missing"]].copy()
    missing["Group"] = ""
    missing["Testing"] = False
    # table of nans to mimic otu
    missing_otu = pd.DataFrame(
        np.nan,
        index=missing.index,
        columns=data.otu_table.columns
    )
    missing = Pyloseq(
        otu_table=missing_otu,
        sample_data=missing,
        tax_table=data.tax_table
    )
    # merge the missing data with the original data
    data.sample_data["Missing"] = False
    data = data.concat(missing)
    data.sample_data["Observed"] = ~data.sample_data["Missing"] & ~data.sample_data["Testing"]
    # ----------------------------------------------------------------------------------------------------------------------




    # ======================================================================================================================
    # PREPARE FEATURES
    data.sample_data["KO"] = data.sample_data["Genotype"].apply(lambda x: 1 if x == "KO" else 0)
    data.sample_data["SCC"] = data.sample_data["Diagnosis"].apply(lambda x: 1 if x == "SCC" else 0)
    data.sample_data["F"] = data.sample_data["Sex"].apply(lambda x: 1 if x == "F" else 0)
    # ----------------------------------------------------------------------------------------------------------------------





    # ======================================================================================================================
    # IMPUTATION

    timepoints = [0, 4, 8, 12, 16, 22]

    for impute, imp_title, imp_name, sq_dist, reg in experiments:
        print(f"Running imputation: {imp_title}")

        imputed_data = {
            0: data.subset_samples(Week=0, inplace=False),
        }
        # debug
        previous = imputed_data[0]
        current = data.subset_samples(Week=4, inplace=False)


        for t in range(1, 6):
            previous=imputed_data[timepoints[t - 1]]
            current=data.subset_samples(Week=timepoints[t], inplace=False)
            print(f"Imputing timepoint {t} (Week {timepoints[t]})")
            imputed_data[timepoints[t]] = impute(
                previous=previous,
                current=current,
                data=data
            )

        # merge
        imputed = imputed_data[0]
        for t in range(1, 6):
            imputed = imputed.concat(imputed_data[timepoints[t]])

        imputed.save(DIR_RESULTS + imp_name)
    # ----------------------------------------------------------------------------------------------------------------------

