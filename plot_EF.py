from matplotlib import pyplot as plt
import numpy as np

def plot_efficient_frontier(std, ret, expected_returns, cov_matrix, n_assets, n_samples):
    
    fig = plt.figure(figsize = (20,15))
    plt.scatter(std, ret, marker="*", s=500, c="r", label="Portfolio")

    n_samples = n_samples
    w = np.random.dirichlet(np.ones(n_assets), n_samples)
    rets_s = w.dot(expected_returns)
    stds_s = np.sqrt(np.diag(w @ cov_matrix @ w.T))
    
    sharpes = rets_s / stds_s
    plt.scatter(stds_s, rets_s, marker=".",s = 150, c=sharpes, cmap="viridis_r")
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.colorbar(label = 'Sharpe ratio')
    plt.title("Fronteira Eficiente", fontsize = 40)
    plt.legend()
    plt.tight_layout()
    plt.clf()
    return fig


def data_efficient_frontier(expected_returns, cov_matrix, n_assets, n_samples):

    n_samples = n_samples
    w = np.random.dirichlet(np.ones(n_assets), n_samples)
    rets_s = w.dot(expected_returns)
    stds_s = np.sqrt(np.diag(w @ cov_matrix @ w.T))
    
    sharpes = rets_s / stds_s
    
    return stds_s, rets_s, sharpes	