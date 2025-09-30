import torch
from pyloseq.pyloseq import Pyloseq
import pandas as pd

n_subjects: int = 65
n_timepoints: int = 6
n_otus: int = 109
sparsity: float = 0.05
smoothness: float = 0.95
random_intercept_scale: float = 0.
noise_scale: float = 0.
seed_signal: int = 1
seed: int = 1
dispersion: float = 1.
effect_size: float = 3.

def generate_longitudinal_microbiome_data(
        n_subjects: int = 65,
        n_timepoints: int = 6,
        n_otus: int = 109,
        sparsity: float = 0.05,
        smoothness: float = 0.95,
        random_intercept_scale: float = 1.,
        noise_scale: float = 1.,
        seed_signal: int = 0,
        seed: int = 0,
        dispersion: float = 1.,
        effect_size: float = 3.,
) -> (Pyloseq, torch.Tensor):
    sample_meta = pd.DataFrame({
        "SubjectID": [i // n_timepoints for i in range(n_subjects * n_timepoints)],
        "Timepoint": [i % n_timepoints for i in range(n_subjects * n_timepoints)],
    })
    sample_meta["SubjectID"] = sample_meta["SubjectID"].astype(str).str.zfill(3)
    sample_meta["SampleID"] = sample_meta["SubjectID"] + "_" + sample_meta["Timepoint"].astype(str)
    sample_meta = sample_meta.set_index("SampleID")
    sample_meta = sample_meta.sort_values(["Timepoint", "SubjectID"], ascending=True)
    subject_meta = pd.DataFrame({
        "SubjectID": [i for i in range(n_subjects)],
        "KO": [i % 2 for i in range(n_subjects)],
        "SCC": [0] * (n_subjects // 2) + [1] * (n_subjects - n_subjects // 2)
    }).set_index("SubjectID")
    subject_meta.index = subject_meta.index.astype(str).str.zfill(3)
    # design matrix
    X = torch.Tensor(subject_meta.values)
    X = torch.cat([torch.ones(X.shape[0], 1), X], dim=1)
    X = torch.cat([X, X[:, [1]] * X[:, [2]]], dim=1)
    # varying coefficients
    torch.manual_seed(seed_signal)
    p = 4
    beta = torch.randn(n_otus, p, n_timepoints)
    # smooth acroos timepoints
    smooth_matrix = (torch.arange(n_timepoints).float().unsqueeze(1) - torch.arange(n_timepoints).float().unsqueeze(0)).abs()
    smooth_matrix = smoothness ** smooth_matrix
    beta = beta @ smooth_matrix
    order = beta[:, 0, :].mean(-1).argsort(descending=True)
    beta = beta[order, :, :]
    # normalize size of beta
    B = beta[:, 1:, :]
    B = B / B.abs().max(dim=2, keepdim=True).values
    B *= effect_size
    beta[:, 1:, :] = B
    # group sparsity
    group_mask = torch.rand((n_otus, p)).lt(sparsity).float()
    group_mask[:, 0] = 1.0
    beta = beta * group_mask.unsqueeze(-1)
    beta[:, 3, :] = 0.
    # fitted mean
    mu = X @ beta
    # random intercepts (subjects)
    torch.manual_seed(seed)
    theta = torch.randn(n_otus, n_subjects) * random_intercept_scale
    # individualized means
    logits = mu + theta.unsqueeze(-1) + torch.randn(n_otus, n_subjects, n_timepoints) * noise_scale
    # dispersion
    logits *= dispersion
    # generate counts
    counts = torch.distributions.Multinomial(logits=logits.permute(1, 2, 0), total_count=10000).sample().permute(2, 0, 1)
    # concatenate along timepoints
    counts = counts.permute(0, 2, 1).reshape(n_otus, n_subjects * n_timepoints)
    counts = pd.DataFrame(counts.numpy().T, index=sample_meta.index, columns=[f"OTU{str(i+1).zfill(4)}" for i in range(n_otus)])
    # missing values
    p = X.sum(1)-3.
    p = p.exp() / (1 + p.exp())
    missing = torch.rand(p.shape) < p
    subject_meta["p_missing"] = p.numpy()
    subject_meta["missing"] = missing.numpy()
    meta = sample_meta.join(subject_meta, on="SubjectID")
    meta["missing"] = meta["missing"] & (meta["Timepoint"] > 1) & (meta["Timepoint"] < 5)
    pseq = Pyloseq(
        otu_table=counts,
        sample_data=meta,
        tax_table=pd.DataFrame(index=[f"OTU{str(i+1).zfill(4)}" for i in range(n_otus)], data={"Kingdom": "Bacteria"})
    )
    return (pseq, beta)