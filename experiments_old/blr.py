import torch
import math

torch.manual_seed(42)
# Generate regression coefficient
p = 5
sig = 0.5
lam = 2.
sqrtlam = torch.sqrt(torch.tensor(lam)).item()
mu_beta = torch.randn(p, 1)
L_beta = torch.randn(p , p)
L_beta = torch.tril(L_beta)  # Lower triangular matrix
Sigma_beta = L_beta @ L_beta.T
Prec_beta = torch.linalg.inv(Sigma_beta)
L_beta = torch.linalg.cholesky(Prec_beta)  # Cholesky decomposition
# beta = torch.distributions.MultivariateNormal(mu_beta, sig**2*Sigma_beta/lam).sample().reshape(p, 1)
beta = torch.distributions.MultivariateNormal(mu_beta.squeeze(), sig**2*Sigma_beta/lam).sample().reshape(p, 1)
# Generate data
n = 100
X = torch.randn(n, p)
mu = X @ beta
eps = torch.randn(n, 1) * sig
y = mu + eps

ols = torch.linalg.solve(X.T @ X, X.T @ y)

# Bayesian linear regression posterior mean
A = X.T @ X  + lam * Prec_beta
b = X.T @ y + lam * Prec_beta @ mu_beta
posterior_mean = torch.linalg.solve(A, b)

# Bayesian linear regression, but as added data
Xt = torch.vstack([X, sqrtlam*L_beta.T])
yt = torch.vstack([y, sqrtlam*L_beta.T@mu_beta])
posterior_mean2 = torch.linalg.solve(Xt.T @ Xt, Xt.T @ yt)

# density ratio

p_mean = X @ posterior_mean
p_covar = torch.linalg.inv(A) * sig**2
p_var = sig**2 + (X @ p_covar @ X.T).diagonal()
p_density = torch.distributions.Normal(p_mean.squeeze(), p_var.sqrt()).log_prob(y.squeeze())

# p_mean = X @ mu_beta
# p_covar = Sigma_beta * sig**2 / lam
# p_var = sig**2 + (X @ p_covar @ X.T).diagonal()
# p_density = torch.distributions.Normal(p_mean.squeeze(), p_var.sqrt()).log_prob(y.squeeze())

q_mean = X @ ols
q_covar = torch.linalg.inv(X.T @ X) * sig**2 / lam
q_var = sig**2 + (X @ q_covar @ X.T).diagonal()
q_density = torch.distributions.Normal(q_mean.squeeze(), q_var.sqrt()).log_prob(y.squeeze())

density_ratio = (p_density - q_density).exp()
density_ratio = density_ratio / density_ratio.sum()

w2 = density_ratio / density_ratio.sum()  # Normalize weights
posterior_mean6 = torch.linalg.solve(X.T @ torch.diag(w2) @ X, X.T @ torch.diag(w2) @ y)



# Just try to find weights that match the posterior
# logw2 = torch.zeros(n) - math.log(n)
logw2 = w2.detach().log()
logw = torch.nn.parameter.Parameter(logw2)
lr = 0.01
optimizer = torch.optim.Adam([logw], lr=lr)
for i in range(1000):
    optimizer.zero_grad()
    w = logw.exp()
    w = w / w.sum()  # Normalize weights
    pm = torch.linalg.solve(X.T @ torch.diag(w) @ X, X.T @ torch.diag(w) @ y)
    loss = torch.mean((pm - posterior_mean) ** 2)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.item()}")
w = logw.data.exp()
w = w / w.sum()  # Normalize weights
posterior_mean5 = torch.linalg.solve(X.T @ torch.diag(w) @ X, X.T @ torch.diag(w) @ y)

# plot w vs w2
maxval = max(w.max(), w2.max())
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.plot(w.detach().numpy(), w2.detach().numpy(), 'o', alpha=0.5)
plt.xlabel('Weights from posterior matching')
plt.ylabel('Weights from density ratio')
plt.title('Comparison of Weights')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlim(0, maxval)
plt.ylim(0, maxval)
plt.grid()
plt.show()

torch.vstack([w, w2, w/w2, p_var, q_var, eps.abs().squeeze()]).T