import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from correlation import avg_abs_corr_rows


def prune_bicluster_genes(
    X,
    gene_set,
    sample_set,
    original_alpha=None,
    n_groups=8,
    random_state=None,
    plot_dendrogram=True,
    dendrogram_sample_size=100,
):
    """
    Prune genes in a bicluster by hierarchical clustering and alpha filtering.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (genes x samples)

    gene_set : array-like
        Gene indices defining the current bicluster gene set

    sample_set : array-like
        Seed sample indices used for pruning

    original_alpha : float or None
        Baseline alpha for the full gene set on the seed samples.
        If None, it is computed from the input gene_set and sample_set.

    n_groups : int
        Number of hierarchical clusters to cut

    random_state : int or None
        Random seed used only for dendrogram subsampling

    plot_dendrogram : bool
        Whether to plot a sample dendrogram

    dendrogram_sample_size : int
        Number of genes to sample for dendrogram plotting

    Returns
    -------
    kept_local_idx : np.ndarray
        Local indices within gene_set that were kept

    pruned_gene_set_original : np.ndarray
        Original X row indices of kept genes

    cluster_labels : np.ndarray
        Cluster labels for all genes in gene_set

    hi_cor_values : np.ndarray
        Alpha values for each hierarchical group

    group_alphas : dict
        Per-group metadata including alpha, size, local indices, original indices
    """
    X = np.asarray(X, dtype=float)
    gene_indices = np.asarray(gene_set, dtype=int)
    sample_indices = np.asarray(sample_set, dtype=int)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array (genes x samples)")
    if len(gene_indices) == 0:
        raise ValueError("gene_set must not be empty")
    if len(sample_indices) == 0:
        raise ValueError("sample_set must not be empty")
    if n_groups <= 0:
        raise ValueError("n_groups must be > 0")

    gem = X[gene_indices, :]
    gem_seed = gem[:, sample_indices]

    print("gem shape:", gem.shape)
    print("gem_seed shape:", gem_seed.shape)

    # Gene-gene correlation across seed samples
    gene_corr = np.corrcoef(gem_seed)
    gene_corr = np.nan_to_num(gene_corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Hierarchical clustering on rows of the correlation matrix
    dist_vec = pdist(gene_corr, metric="euclidean")
    linkage_matrix = linkage(dist_vec, method="complete")

    k = int(min(n_groups, len(gene_indices)))
    cluster_labels = fcluster(linkage_matrix, t=k, criterion="maxclust")

    if original_alpha is None:
        original_alpha = avg_abs_corr_rows(gem_seed)

    hi_cor_values = []
    group_alphas = {}
    kept_local_idx = []

    for group_id in range(1, k + 1):
        local_group_idx = np.where(cluster_labels == group_id)[0]
        n_genes_group = len(local_group_idx)

        if n_genes_group < 2:
            group_alpha = 0.0
        else:
            group_alpha = avg_abs_corr_rows(gem_seed[local_group_idx])

        hi_cor_values.append(group_alpha)

        group_alphas[group_id] = {
            "alpha": float(group_alpha),
            "n_genes": int(n_genes_group),
            "local_idx": local_group_idx,
            "original_genes": gene_indices[local_group_idx],
        }

        if group_alpha > original_alpha:
            kept_local_idx.extend(local_group_idx.tolist())

    kept_local_idx = np.sort(np.asarray(kept_local_idx, dtype=int))
    hi_cor_values = np.asarray(hi_cor_values)

    print(f"Alpha original: {original_alpha:.6f}")
    for group_id in range(1, k + 1):
        info = group_alphas[group_id]
        status = "KEPT" if info["alpha"] > original_alpha else "REMOVED"
        print(
            f"  Group {group_id} ({info['n_genes']:4d} genes): "
            f"alpha = {info['alpha']:.6f} ({status})"
        )
    print(f"Genes before: {len(gene_indices)} | after: {len(kept_local_idx)}")

    if plot_dendrogram and len(gene_indices) > 1:
        rng = np.random.default_rng(random_state)
        sample_size = int(min(dendrogram_sample_size, len(gene_indices)))
        sample_idx = rng.choice(len(gene_indices), sample_size, replace=False)

        sample_corr = np.corrcoef(gem_seed[sample_idx])
        sample_corr = np.nan_to_num(sample_corr, nan=0.0, posinf=0.0, neginf=0.0)
        sample_dist_vec = pdist(sample_corr, metric="euclidean")
        sample_linkage = linkage(sample_dist_vec, method="complete")

        plt.figure(figsize=(12, 5))
        dendrogram(sample_linkage, leaf_rotation=90, leaf_font_size=8)
        plt.title(f"Dendrogram (sample of {sample_size} genes)")
        plt.xlabel("Genes")
        plt.ylabel("Euclidean distance on correlation rows")
        plt.tight_layout()
        plt.show()

    pruned_gene_set_original = gene_indices[kept_local_idx]

    return (
        kept_local_idx,
        pruned_gene_set_original,
        cluster_labels,
        hi_cor_values,
        group_alphas,
    )