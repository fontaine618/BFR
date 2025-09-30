import torch
from bfr.kulsif import kulsif
from utils.kernels.squared_exponential import squared_exponential_kernel
from utils.squared_metrics.euclidean import squared_L2_distance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

N = 500
M = 500

mux = torch.zeros(1, 2)
muy = torch.ones(1, 2)

X = torch.randn(N, 2) + mux
Y = torch.randn(M, 2) + muy


plt.cla()
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c="b", marker="o", label="X")
plt.scatter(Y[:, 0], Y[:, 1], c="r", marker="x", label="Y")
plt.legend()
plt.show()


logprobaxx = torch.distributions.Normal(0., 1.).log_prob(X - mux).sum(-1)
logprobaxy = torch.distributions.Normal(0., 1.).log_prob(X - muy).sum(-1)
logratio = logprobaxx - logprobaxy
ratio = torch.exp(logratio)

df = pd.DataFrame({
    "x1": X[:, 0],
    "x2": X[:, 1],
    "logprobaxx": logprobaxx,
    "logprobaxy": logprobaxy,
    "logratio": logratio,
    "ratio": ratio,
})

plt.cla()
plt.figure(figsize=(10, 10))
sns.scatterplot(data=df, x="x1", y="x2", hue="logratio", palette="coolwarm", legend="auto")
plt.show()