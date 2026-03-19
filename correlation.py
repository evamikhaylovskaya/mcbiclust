import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def avg_abs_corr_rows(A, return_corr=False):
    """
    Compute MCbiclust alpha score:
    mean absolute correlation between all gene pairs.

    Parameters
    ----------
    A : np.ndarray
        Expression matrix (genes x samples)

    return_corr : bool
        If True, also return correlation matrix.

    Returns
    -------
    alpha : float
        Average absolute correlation
    C : np.ndarray (optional)
        Gene–gene correlation matrix
    """

    A = np.asarray(A, dtype=float)
    genes, samples = A.shape

    if genes < 2 or samples < 2:
        return 0.0

    # gene-gene correlation
    C = np.corrcoef(A)
    C = np.nan_to_num(C, nan=0.0)

    absC = np.abs(C)

    alpha = absC.sum() / (genes * genes)

    if return_corr:
        return float(alpha), C
    else:
        return float(alpha)

def plot_gene_corr_heatmap(X, genes=None, subset_size=100, seed=42):
    """
    Plot correlation heatmap for a random subset of genes.
    """
    rng        = np.random.default_rng(seed)
    subset_idx = np.sort(rng.choice(X.shape[0], subset_size, replace=False))
    X_subset   = X[subset_idx]
    C_subset   = np.corrcoef(X_subset)

    # labels: gene names if provided, else gene indices
    if genes is not None:
        labels = genes[subset_idx]
    else:
        labels = subset_idx.astype(str)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        C_subset,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        cbar_kws={"label": "Pearson r"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    plt.title(f"Gene-gene correlation (random {subset_size} genes)")
    ax.set_xlabel("Genes")
    ax.set_ylabel("Genes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=6)

    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
