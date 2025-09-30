from pyloseq.pyloseq import Pyloseq
import pandas as pd
import numpy as np
import torch
import os
import utils.squared_metrics.compositional as c
from bfr.kulsif import kulsif
from bfr.bgfr import BGFR, GFR
from bfr.nlbgfr import NLGFR
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.kernels.laplacian import laplacian_kernel
from utils.squared_metrics.euclidean import squared_L2_distance
from experiments_old.dmbt1.simulated.gen_data import generate_longitudinal_microbiome_data

for seed in range(100):

    # ======================================================================================================================
    # SETTINGS
    DIR_RESULTS = "/experiments_old/dmbt1/simulated/results/"

    DIR_SEED = DIR_RESULTS + f"{seed}/"
    os.makedirs(DIR_SEED, exist_ok=True)

    X_scale = 0.5
    Y_scale_factor = 0.5
    reg = 0.0005

    max_iter = 100
    lr = 0.05
    # ----------------------------------------------------------------------------------------------------------------------


    # ======================================================================================================================
    # DATA
    data, beta = generate_longitudinal_microbiome_data(
        seed=seed,
        seed_signal=1,
        noise_scale=1.,
        random_intercept_scale=1.0,
        effect_size=2.,
        smoothness=0.7,
        dispersion=0.5
    )
    data.save(DIR_SEED + "original")
    np.save(DIR_SEED + "original.beta", beta.cpu().numpy())

    beta.norm(10, dim=2)
    # ----------------------------------------------------------------------------------------------------------------------


    # ======================================================================================================================
    # IMPUTATION FUNCTIONS
    def complete_cases(previous: Pyloseq, current: Pyloseq, **kwargs) -> Pyloseq:
        return current.subset_samples(missing=False, inplace=False)


    def locf(previous: Pyloseq, current: Pyloseq, **kwargs) -> Pyloseq:
        "Last Observation Carried Forward (LOCF) imputation"
        current = current.copy()
        which_missing = current.sample_data["missing"].values
        current.otu_table.iloc[which_missing, :] = previous.otu_table.iloc[which_missing, :].values
        return current


    def global_impute(previous: Pyloseq, current: Pyloseq, **kwargs) -> Pyloseq:
        # Prior data
        X = torch.Tensor(previous.sample_data[["KO", "SCC"]].values)
        X_prior = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_prior = torch.Tensor(previous.otu_table.values)
        # Observed data
        observed = current.subset_samples(missing=False, inplace=False)
        X = torch.Tensor(observed.sample_data[["KO", "SCC"]].values)
        X_observed = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_observed = torch.Tensor(observed.otu_table.values)
        # Missing data
        missing = current.subset_samples(missing=True, inplace=False)
        X = torch.Tensor(missing.sample_data[["KO", "SCC"]].values)
        X_missing = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_missing = torch.Tensor(missing.otu_table.values)
        # Compute distance matrices
        D2x_train_train = squared_L2_distance(X_observed[:, :2], X_observed[:, :2])
        D2x_prior_train = squared_L2_distance(X_observed[:, :2], X_prior[:, :2])
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
        bgfr = BGFR(X_observed, X_missing, rxy_train=rxy_train, rx_train=rx_train, reg=1e-5)
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
        Y_pred = (10000. * Y_pred).round().astype(int)  # convert to counts
        imputed = current.subset_samples(missing=True, inplace=False)
        imputed.otu_table = pd.DataFrame(
            Y_pred,
            index=imputed.otu_table.index,
            columns=imputed.otu_table.columns
        )
        return observed.concat(imputed)


    def gfr(previous: Pyloseq, current: Pyloseq, **kwargs) -> Pyloseq:
        # Observed data
        observed = current.subset_samples(missing=False, inplace=False)
        X = torch.Tensor(observed.sample_data[["KO", "SCC"]].values)
        X_observed = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_observed = torch.Tensor(observed.otu_table.values)
        # Missing data
        missing = current.subset_samples(missing=True, inplace=False)
        X = torch.Tensor(missing.sample_data[["KO", "SCC"]].values)
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
        Y_pred = (10000. * Y_pred).round().astype(int)  # convert to counts
        imputed = current.subset_samples(missing=True, inplace=False)
        imputed.otu_table = pd.DataFrame(
            Y_pred,
            index=imputed.otu_table.index,
            columns=imputed.otu_table.columns
        )
        return observed.concat(imputed)


    def markov_impute(previous: Pyloseq, current: Pyloseq, **kwargs) -> Pyloseq:
        # Prior data
        X = torch.Tensor(previous.sample_data[["KO", "SCC"]].values)
        X_prior = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_prior = torch.Tensor(previous.otu_table.values)
        # Observed data
        observed = current.subset_samples(missing=False, inplace=False)
        X = torch.Tensor(observed.sample_data[["KO", "SCC"]].values)
        X_observed = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_observed = torch.Tensor(observed.otu_table.values)
        # Missing data
        missing = current.subset_samples(missing=True, inplace=False)
        X = torch.Tensor(missing.sample_data[["KO", "SCC"]].values)
        X_missing = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_missing = torch.Tensor(missing.otu_table.values)
        # Compute distance matrices
        D2x_train_train = squared_L2_distance(X_observed[:, :2], X_observed[:, :2])
        D2x_prior_train = squared_L2_distance(X_observed[:, :2], X_prior[:, :2])
        D2y_train_train = sq_dist(Y_observed, Y_observed)
        D2y_prior_train = sq_dist(Y_observed, Y_prior)
        Y_scale = (torch.median(D2y_train_train[D2y_train_train > 0.])).sqrt() * Y_scale_factor
        # Kernel across training points
        kx = squared_exponential_kernel(D2x_train_train, X_scale)
        ky = laplacian_kernel(D2y_train_train, Y_scale)
        # Kernel between training and prior points
        kx_prior = squared_exponential_kernel(D2x_prior_train, X_scale)
        ky_prior = laplacian_kernel(D2y_prior_train, Y_scale)
        ids_missing = current.sample_data.loc[current.sample_data["missing"], "SubjectID"].values
        i_missing = ids_missing.astype(int)
        Y_pred_list = []
        for j in range(len(i_missing)):
            id = ids_missing[j]
            i = i_missing[j]

            # Density ratio estimation
            rxy_train = kulsif(kx * ky, kx_prior[:, [i]] * ky_prior[:, [i]], reg=reg)
            rx_train = kulsif(kx, kx_prior[:, [i]], reg=reg)
            # Prepare weights
            bgfr = BGFR(X_observed, X_missing, rxy_train=rxy_train, rx_train=rx_train, reg=1e-5)
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

        Y_pred = (10000. * Y_pred).round().astype(int)  # convert to counts
        imputed = current.subset_samples(missing=True, inplace=False)
        imputed.otu_table = pd.DataFrame(
            Y_pred,
            index=imputed.otu_table.index,
            columns=imputed.otu_table.columns
        )
        return observed.concat(imputed)


    def local_impute(previous: Pyloseq, current: Pyloseq, data: Pyloseq, **kwargs) -> Pyloseq:
        # Prior data
        X = torch.Tensor(data.sample_data[["KO", "SCC"]].values)
        X_prior = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_prior = torch.Tensor(data.otu_table.values)
        sid = data.sample_data["SubjectID"].values
        # Observed data
        observed = current.subset_samples(missing=False, inplace=False)
        X = torch.Tensor(observed.sample_data[["KO", "SCC"]].values)
        X_observed = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_observed = torch.Tensor(observed.otu_table.values)
        # Missing data
        missing = current.subset_samples(missing=True, inplace=False)
        X = torch.Tensor(missing.sample_data[["KO", "SCC"]].values)
        X_missing = torch.hstack([X, X[:, [0]] * X[:, [1]]])
        Y_missing = torch.Tensor(missing.otu_table.values)
        # Compute distance matrices
        D2x_train_train = squared_L2_distance(X_observed[:, :2], X_observed[:, :2])
        D2x_prior_train = squared_L2_distance(X_observed[:, :2], X_prior[:, :2])
        D2y_train_train = sq_dist(Y_observed, Y_observed)
        D2y_prior_train = sq_dist(Y_observed, Y_prior)
        Y_scale = (torch.median(D2y_train_train[D2y_train_train > 0.])).sqrt() * Y_scale_factor
        # Kernel across training points
        kx = squared_exponential_kernel(D2x_train_train, X_scale)
        ky = laplacian_kernel(D2y_train_train, Y_scale)
        # Kernel between training and prior points
        kx_prior = squared_exponential_kernel(D2x_prior_train, X_scale)
        ky_prior = laplacian_kernel(D2y_prior_train, Y_scale)
        ids_missing = current.sample_data.loc[current.sample_data["missing"], "SubjectID"].values
        i_missing = ids_missing.astype(int)
        Y_pred_list = []
        for j in range(len(i_missing)):
            id = ids_missing[j]
            i = i_missing[j]
            which = (sid == id)*(~data.sample_data["missing"].values)

            # Density ratio estimation
            rxy_train = kulsif(kx * ky, kx_prior[:, which] * ky_prior[:, which], reg=reg)
            rx_train = kulsif(kx, kx_prior[:, which], reg=reg)
            # Prepare weights
            bgfr = BGFR(X_observed, X_missing, rxy_train=rxy_train, rx_train=rx_train, reg=1e-5)
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

        Y_pred = (10000. * Y_pred).round().astype(int)  # convert to counts
        imputed = current.subset_samples(missing=True, inplace=False)
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
        # (global_impute, "Global (Aitchison)", "global_aitchison", c.squared_aitchison_distance),
        # (markov_impute, "Markov (Aitchison)", "markov_aitchison", c.squared_aitchison_distance),
        # (local_impute, "Local (Aitchison)", "local_aitchison", c.squared_aitchison_distance),
        # (global_impute, "Global (Hellinger)", "global_hellinger", c.squared_hellinger_distance),
        # (markov_impute, "Markov (Hellinger)", "markov_hellinger", c.squared_hellinger_distance),
        # (local_impute, "Local (Hellinger)", "local_hellinger", c.squared_hellinger_distance),
        # (global_impute, "Global (Spherical)", "global_spherical", c.squared_spherical_norm_distance),
        # (markov_impute, "Markov (Spherical)", "markov_spherical", c.squared_spherical_norm_distance),
        # (local_impute, "Local (Spherical)", "local_spherical", c.squared_spherical_norm_distance),
        # (complete_cases, "Complete cases", "complete", None),
        (gfr, "GFR (Aitchison)", "gfr_aitchison", c.squared_aitchison_distance),
        (gfr, "GFR (Hellinger)", "gfr_hellinger", c.squared_hellinger_distance),
        (gfr, "GFR (Spherical)", "gfr_spherical", c.squared_spherical_norm_distance),
        # (locf, "LOCF", "locf", None)
    ]


    # ----------------------------------------------------------------------------------------------------------------------

    # ======================================================================================================================
    # IMPUTATION
    for impute, imp_title, imp_name, sq_dist in experiments:
        print(f"Running imputation: {imp_title}...")

        imputed_data = {
            0: data.subset_samples(Timepoint=0, inplace=False),
            1: data.subset_samples(Timepoint=1, inplace=False),
            5: data.subset_samples(Timepoint=5, inplace=False)
        }
        # t = 3
        # previous = imputed_data[t - 1]
        # current = data.subset_samples(Timepoint=t, inplace=False)


        for t in range(2, 5):
            imputed_data[t] = impute(
                previous=imputed_data[t - 1],
                current=data.subset_samples(Timepoint=t, inplace=False),
                data=data
            )

        # merge
        imputed = imputed_data[0]
        for t in range(1, 6):
            imputed = imputed.concat(imputed_data[t])

        imputed.save(DIR_SEED + imp_name)
    # ----------------------------------------------------------------------------------------------------------------------

