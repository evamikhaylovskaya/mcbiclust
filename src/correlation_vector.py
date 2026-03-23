import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt

from correlation import avg_abs_corr_rows


def gene_vec_fun(X, gene_set, seed, splits=8):
    """
    Build the gene vector from the best hierarchical subgroup.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (genes x samples)

    gene_set : array-like
        Gene indices to consider

    seed : array-like
        Seed sample indices

    splits : int
        Maximum number of hierarchical clusters to try

    Returns
    -------
    gene_vec : np.ndarray
        Mean expression profile of the best gene subgroup across seed samples

    best_idx : np.ndarray
        Local indices within gene_set of the best subgroup
    """
    X = np.asarray(X, dtype=float)
    I = np.asarray(gene_set, dtype=int)
    J = np.asarray(seed, dtype=int)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(I) == 0:
        raise ValueError("gene_set must not be empty")
    if len(J) == 0:
        raise ValueError("seed must not be empty")
    if splits < 2:
        raise ValueError("splits must be >= 2")

    gem_seed = X[I][:, J]

    C = np.corrcoef(gem_seed)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    dist_vec = pdist(C, metric="euclidean")
    Z = linkage(dist_vec, method="complete")

    best_k_score = -np.inf
    best_k = 2

    for k in range(2, splits + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        scores = []

        for g in range(1, k + 1):
            idx = np.where(labels == g)[0]
            if len(idx) < 2:
                scores.append(0.0)
                continue

            alpha = avg_abs_corr_rows(gem_seed[idx])
            scores.append(alpha * np.sqrt(len(idx)))

        if max(scores) > best_k_score:
            best_k_score = max(scores)
            best_k = k

    labels = fcluster(Z, t=best_k, criterion="maxclust")

    scores = []
    for g in range(1, best_k + 1):
        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            scores.append(0.0)
            continue

        alpha = avg_abs_corr_rows(gem_seed[idx])
        scores.append(alpha * np.sqrt(len(idx)))

    best_group = np.argmax(scores) + 1
    best_idx = np.where(labels == best_group)[0]

    print(f"Best k = {best_k}")
    print(f"Best group = {best_group}")
    print(f"n_genes = {len(best_idx)}")
    print(f"alpha*sqrt(n) = {scores[best_group - 1]:.4f}")

    gene_vec = gem_seed[best_idx].mean(axis=0)

    return gene_vec, best_idx

def print_top_genes(corr_vec, gene_names=None, top_n=20):
    corr_vec = np.asarray(corr_vec)
    ranked_idx = np.argsort(np.abs(corr_vec))[::-1]

    print("\nRank  Gene ID  Gene Name                     Correlation")
    print("-" * 60)

    for rank, i in enumerate(ranked_idx[:top_n], start=1):
        gene_id = i
        gene_name = gene_names[i] if gene_names is not None else f"gene_{i}"
        corr = corr_vec[i]

        print(f"{rank:<5} {gene_id:<8} {gene_name:<30} {corr:+.4f}")
def cv_eval(X_part, X_all, seed, splits=8):
    """
    Compute correlation vector for all genes against the best gene vector.

    Parameters
    ----------
    X_part : np.ndarray
        Pruned / subset matrix used to build the gene vector

    X_all : np.ndarray
        Full expression matrix used to compute correlations for all genes

    seed : array-like
        Seed sample indices

    splits : int
        Maximum number of hierarchical clusters to try

    Returns
    -------
    corr_vec : np.ndarray
        Correlation vector for all genes in X_all

    best_idx : np.ndarray
        Local indices inside X_part used to build the gene vector
    """
    X_part = np.asarray(X_part, dtype=float)
    X_all = np.asarray(X_all, dtype=float)
    J = np.asarray(seed, dtype=int)

    if X_part.ndim != 2 or X_all.ndim != 2:
        raise ValueError("X_part and X_all must be 2D arrays")
    if len(J) == 0:
        raise ValueError("seed must not be empty")

    gene_vec, best_idx = gene_vec_fun(
        X_part,
        np.arange(X_part.shape[0]),
        J,
        splits=splits
    )

    X_seed = X_all[:, J]
    n = len(J)

    gv_mean = gene_vec.mean()
    gv_std = gene_vec.std()
    gv_norm = (gene_vec - gv_mean) / (gv_std + 1e-8)

    g_mean = X_seed.mean(axis=1, keepdims=True)
    g_std = X_seed.std(axis=1, keepdims=True) + 1e-8
    g_norm = (X_seed - g_mean) / g_std

    corr_vec = (g_norm @ gv_norm) / n

    print(f"Correlation vector shape: {corr_vec.shape}")
    print(f"Min: {corr_vec.min():.4f}")
    print(f"Max: {corr_vec.max():.4f}")
    print(f"Mean |corr|: {np.abs(corr_vec).mean():.4f}")
    print_top_genes(corr_vec, gene_names=None, top_n=20)

    return corr_vec, best_idx



def plot_correlation_vector_distribution(
    corr_vector,
    bins=100,
    save_path=None
):
    """
    Plot distribution of correlation vector values.

    Parameters
    ----------
    corr_vector : np.ndarray
        Correlation vector for all genes

    bins : int
        Histogram bins

    save_path : str or None
        Optional file path to save figure
    """
    corr_vector = np.asarray(corr_vector, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        corr_vector,
        bins=bins,
        edgecolor="none",
        color="steelblue"
    )
    axes[0].axvline(x=0, color="red", linestyle="--")
    axes[0].set_xlabel("Correlation")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of correlation vector values")

    axes[1].hist(
        np.abs(corr_vector),
        bins=bins,
        edgecolor="none",
        color="steelblue"
    )
    axes[1].set_xlabel("|Correlation|")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of |correlation| values")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_correlation_vector_ranked(
    corr_vector,
    ranked_gene_indices=None,
    save_path=None
):
    """
    Scatter plot of correlation vector ordered by |correlation|.

    Parameters
    ----------
    corr_vector : np.ndarray
        Correlation vector

    ranked_gene_indices : np.ndarray or None
        Precomputed ranking indices. If None, they will be computed.

    save_path : str or None
        Optional file path to save figure
    """
    corr_vector = np.asarray(corr_vector, dtype=float)

    if ranked_gene_indices is None:
        ranked_gene_indices = np.argsort(np.abs(corr_vector))[::-1]

    plt.figure(figsize=(10, 4))

    plt.scatter(
        range(len(corr_vector)),
        corr_vector[ranked_gene_indices],
        s=1,
        alpha=0.5,
        color="steelblue"
    )

    plt.axhline(y=0, color="red", linestyle="--")

    plt.xlabel("Gene rank (by |correlation|)")
    plt.ylabel("Correlation value")
    plt.title("Correlation vector: genes ranked by |correlation|")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()