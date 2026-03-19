import numpy as np


def select_initial_seed_genes(
    X,
    n_genes=1000,
    biological_gene_set=None,
    random_state=None
):
    """
    Select the initial gene set used by MCbiclust.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (genes x samples)

    n_genes : int
        Number of genes to select

    biological_gene_set : array-like or None
        Optional list of gene indices defining a biological gene set.
        If provided, genes are sampled from this set.
        For example, MitoCarta gene indices for mitochondrial analysis.

    random_state : int or None
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Sorted array of selected gene indices (0-indexed)
    """
    rng = np.random.default_rng(random_state)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array (genes x samples)")

    n_total_genes = X.shape[0]

    if n_total_genes < n_genes:
        raise ValueError(
            f"Dataset contains only {n_total_genes} genes, "
            f"but {n_genes} genes were requested."
        )

    if biological_gene_set is not None:
        biological_set = np.asarray(biological_gene_set, dtype=int)

        if len(biological_set) > n_genes:
            gene_indices = rng.choice(biological_set, size=n_genes, replace=False)
            print(f"Sampled {n_genes} genes from biological set of {len(biological_set)}")
        else:
            gene_indices = biological_set.copy()
            print(f"Using all {len(biological_set)} genes from biological set")
    else:
        gene_indices = rng.choice(X.shape[0], size=n_genes, replace=False)
        print(f"Randomly selected {n_genes} genes from {X.shape[0]} total")

    return np.sort(gene_indices)