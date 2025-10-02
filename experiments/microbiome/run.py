import torch
import os
import sys
import pandas as pd
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('/storage/home/spf5519/work/BFR/')
from nlbgfr.bfr import BFR
from nlbgfr.nlbr import NLBR
import utils.squared_metrics.compositional as c
from experiments.microbiome.data_loader import T2D



# ======================================================================================================================
# EXPERIMENTS
prior_dataset_name = "MetaCardis_2020_a"
train_test_dataset_names = ["QinJ_2012", "KarlssonFH_2013"]
variance_ratios = {
    "QinJ_2012": [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, float("inf")],
    "KarlssonFH_2013": [3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, float("inf")]
}
n_variance_ratios = max(len(v) for v in variance_ratios.values())
sq_dists = {
    "hellinger": c.squared_hellinger_distance,
    "aitchison": c.squared_aitchison_distance,
    "spherical": c.squared_spherical_norm_distance,
}
experiments = list(itertools.product(
    sq_dists.keys(),
    train_test_dataset_names,
    range(n_variance_ratios),
))
n_experiments = len(experiments)
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# SETTINGS
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DIR_RESULTS = f"./results/{SEED}/"
# DIR_RESULTS = f"experiments/microbiome/results/{SEED}/"
os.makedirs(DIR_RESULTS, exist_ok=True)
# ----------------------------------------------------------------------------------------------------------------------



# ======================================================================================================================
# HELPER FUNCTIONS

def kernel(X1: pd.DataFrame, X2: pd.DataFrame) -> torch.Tensor:
    N1 = X1.shape[0]
    N2 = X2.shape[0]
    K = torch.zeros(N1, N2)
    K += X1['T2D'].values.reshape(-1, 1) == X2['T2D'].values.reshape(1, -1)
    K += X1['antibiotics'].values.reshape(-1, 1) == X2['antibiotics'].values.reshape(1, -1)
    K += X1['female'].values.reshape(-1, 1) == X2['female'].values.reshape(1, -1)
    K += (torch.tensor(X1['age'].values, dtype=torch.float32).reshape(-1, 1) -
        torch.tensor(X2['age'].values, dtype=torch.float32).reshape(1, -1)).pow(2.).div(100.0).neg().exp()
    K += (torch.tensor(X1['BMI'].values, dtype=torch.float32).reshape(-1, 1) -
        torch.tensor(X2['BMI'].values, dtype=torch.float32).reshape(1, -1)).pow(2.).div(25.0).neg().exp()
    return K

# optimize
def clr(Y):
    log_Y = Y.clamp(min=1e-6).log()
    return log_Y - log_Y.mean(dim=-1, keepdim=True)

def inv_clr(Y):
    exp_Y = Y.exp()
    return exp_Y / exp_Y.sum(dim=-1, keepdim=True)

def center(Y):
    return Y - Y.mean(dim=-1, keepdim=True)

def fit_params(sq_dist_name):
    lr = {
        "hellinger": 1e1,
        "aitchison": 1e-2,
        "spherical": 1e0,
    }[sq_dist_name]
    return {
        "transform": clr,
        "inv_transform": inv_clr,
        "projection": center,
        "max_iter": 10000,
        "lr": lr,
        "tol": 1e-8
    }

def prediction_metrics(Ytrue: torch.Tensor, Ypred: torch.Tensor):
    metrics = {
        "bc": (c.squared_bray_curtis_dissimilarity, "Bray-Curtis"),
        "aitchison": (c.squared_aitchison_distance, "Aitchison"),
        "hellinger": (c.squared_hellinger_distance, "Hellinger"),
        "spherical": (c.squared_spherical_norm_distance, "Geodesic"),
    }
    results = pd.DataFrame(columns=["Metric", "Value"])
    for metric_name, (metric_func, metric_display) in metrics.items():
        value = metric_func(Ytrue, Ypred).mean().item()
        results = pd.concat([results, pd.DataFrame({
            "Metric": [metric_display],
            "Value": [value]
        })], ignore_index=True)
    return results
# ----------------------------------------------------------------------------------------------------------------------




# ======================================================================================================================
# RUN EXPERIMENT
def run_experiment(
        prior_dataset_name: str,
        train_test_dataset_name: str,
        variance_ratio: float,
        sq_dist_name: str,
        seed: int = 0,
):
    sq_dist = sq_dists[sq_dist_name]
    
    # load prior data
    prior_study = T2D("t2d_meta.csv", "t2d_rel.csv")
    prior_study.filter_rare_taxa()
    prior_study.filter_study(prior_dataset_name)
    Yprior = torch.tensor(prior_study.relative_abundance.values, dtype=torch.float32)
    Xprior = prior_study.features
    Np = Yprior.size(0)

    # load training data
    train = T2D("t2d_meta.csv", "t2d_rel.csv")
    train.filter_rare_taxa()
    train.filter_study(train_test_dataset_name)
    train.assign_train_test(random_state=seed)
    train.filter_set("train")
    Ytrain = torch.tensor(train.relative_abundance.values, dtype=torch.float32)
    Xtrain = train.features
    Nt = Ytrain.size(0)

    # load evaluation data
    eval = T2D("t2d_meta.csv", "t2d_rel.csv")
    eval.filter_rare_taxa()
    eval.filter_study(train_test_dataset_name)
    eval.assign_train_test(random_state=seed)
    eval.filter_set("test")
    Yeval = torch.tensor(eval.relative_abundance.values, dtype=torch.float32)
    Xeval = eval.features
    Ne = Yeval.size(0)

    # prepare kernel matrices
    Kpt = kernel(Xprior, Xtrain)
    Kpe = kernel(Xprior, Xeval)
    Ktt = kernel(Xtrain, Xtrain)
    Kpp = kernel(Xprior, Xprior)
    Kte = kernel(Xtrain, Xeval)

    # prepare prior function
    nlgfr_prior = NLBR(
            Kx_train_train=Kpp,
            intercept_variance_ratio=float("inf"),
            regression_variance_ratio=float("inf"),
            variance_explained=0.99
        )
    def prior(Y):
        return nlgfr_prior.ese_pairwise(Kpt.T, sq_dist(Y, Yprior)).T

    # prepare BFR model
    bfr = BFR(
        squared_distance=sq_dist,
        prior=prior,
        Kx_train_train=Ktt,
        intercept_variance_ratio=variance_ratio,
        regression_variance_ratio=variance_ratio,
        variance_explained=0.99
    )

    # run prediction
    Yinit = Yeval.clone().fill_(1.0 / Yeval.size(1))
    Ypred = bfr.ppfm_apgd(
        Y_init=Yinit,
        Y_train=Ytrain,
        Kx_test_train=Kte.T,
        **fit_params(sq_dist_name)
    )

    # compute metrics
    results = prediction_metrics(Yeval, Ypred)
    results["Prior Dataset"] = prior_dataset_name
    results["Train/Test Dataset"] = train_test_dataset_name
    results["Variance Ratio"] = variance_ratio
    results["Distance"] = sq_dist_name
    results["Seed"] = seed
    return results
# ----------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# MAIN LOOP
all_results = pd.DataFrame()
for i in range(n_experiments):
    sq_dist_name, train_test_dataset_name, i_variance_ratio = experiments[i]
    variance_ratio = variance_ratios[train_test_dataset_name][i_variance_ratio]
    print(f"Running experiment {i+1}/{n_experiments}: "
          f"Train/Test={train_test_dataset_name}, Variance Ratio={variance_ratio}, Distance={sq_dist_name}")
    results = run_experiment(
        prior_dataset_name=prior_dataset_name,
        train_test_dataset_name=train_test_dataset_name,
        variance_ratio=variance_ratio,
        sq_dist_name=sq_dist_name,
        seed=SEED
    )
    print(results)
    all_results = pd.concat([all_results, results], ignore_index=True)
    all_results.to_csv(DIR_RESULTS + "metrics.csv", index=False)
# ----------------------------------------------------------------------------------------------------------------------
