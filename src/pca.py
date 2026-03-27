# pca.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA


def pc1_vec_fun(X, gene_set, seed_sort, n=10, align_sign=False):
    """
    Compute PC1 values for all samples using PCA on the top n seed samples.

    Equivalent to R's PC1VecFun.

    Parameters
    ----------
    X         : np.ndarray  expression matrix (genes x samples)
    gene_set  : np.ndarray  gene indices to use (pruned genes)
    seed_sort : np.ndarray  ranked sample indices from SampleSort
    n         : int         number of top samples to fit PCA on (default 10)

    Returns
    -------
    pc1_vec : np.ndarray  PC1 value for every sample in seed_sort order
    """
    I = np.asarray(gene_set,  dtype=int)
    S = np.asarray(seed_sort, dtype=int)

    top_gem    = X[I][:, S]       # genes x all_ranked_samples
    pca_matrix = top_gem[:, :n]   # genes x n seed samples

    # PCA on top n samples
    pca = PCA(n_components=1)
    pca.fit(pca_matrix.T)         # fit on (n x genes)
    loadings = pca.components_.T  # genes x 1

   
    X_design = np.hstack([np.ones((len(I), 1)), loadings])   # genes x 2
    beta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ top_gem
    pc1_vec = beta[1, :]   # slope coefficients
    
    # align sign: positive arm should dominate
    if align_sign and pc1_vec.mean() < 0:
        pc1_vec = -pc1_vec
        print("Sign flipped")

    return pc1_vec


def plot_fork(pc1_vec, n_seed=10, title="Fork Plot",
              outlier_pos=None, outlier_label=None):
    """
    Simple fork plot: PC1 value vs sample rank.

    Parameters
    ----------
    pc1_vec       : np.ndarray  PC1 values in sample rank order
    n_seed        : int         number of seed samples
    title         : str         plot title
    outlier_pos   : int         optional rank position to highlight
    outlier_label : str         optional label for highlighted point
    """
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(pc1_vec)), pc1_vec,
                s=3, alpha=0.6, color="steelblue")
    plt.axhline(0,      color="red",   lw=1,   linestyle="--")
    plt.axvline(n_seed, color="green", lw=1,   linestyle="--",
                label=f"End of seed ({n_seed} samples)")

    if outlier_pos is not None:
        label = outlier_label or f"rank {outlier_pos}"
        plt.scatter(outlier_pos, pc1_vec[outlier_pos],
                    color="red", s=60, zorder=5,
                    label=f"{label} (PC1={pc1_vec[outlier_pos]:.1f})")

    plt.xlabel("Sample rank")
    plt.ylabel("PC1 value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def pc1_summary_stats(pc1_vec, n_seed=10):
    """Print key PC1 statistics."""
    upper = int((pc1_vec > 0).sum())
    lower = int((pc1_vec < 0).sum())
    print(f"PC1 shape:            {pc1_vec.shape}")
    print(f"Min / Max / Mean:     {pc1_vec.min():.4f} / "
          f"{pc1_vec.max():.4f} / {pc1_vec.mean():.4f}")
    print(f"Samples with PC1 > 0: {upper}")
    print(f"Samples with PC1 < 0: {lower}")
    print(f"Seed PC1 values:      "
          f"{np.round(pc1_vec[:n_seed], 2).tolist()}")