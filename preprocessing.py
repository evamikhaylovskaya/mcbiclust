import numpy as np
from data_loading import ExpressionDataset


def log2_transform(dataset: ExpressionDataset, pseudocount: float = 1.0) -> ExpressionDataset:
    """
    Apply log2(x + pseudocount) to the expression matrix.
    Useful for RNA-seq style data such as CCLE.
    """
    if pseudocount <= 0:
        raise ValueError("pseudocount must be > 0")

    X_new = np.log2(dataset.X + pseudocount)

    return ExpressionDataset(
        X=X_new,
        genes=dataset.genes.copy(),
        samples=dataset.samples.copy(),
        name=f"{dataset.name} | log2"
    )


def filter_low_expression_genes(dataset: ExpressionDataset, min_mean: float = 1.0) -> ExpressionDataset:
    """
    Keep genes whose mean expression across samples is at least min_mean.
    Useful for RNA-seq data with many near-zero genes.
    """
    keep = dataset.X.mean(axis=1) >= min_mean
    kept = int(np.sum(keep))

    print(f"Keeping {kept} / {dataset.n_genes} genes with mean >= {min_mean}")

    return ExpressionDataset(
        X=dataset.X[keep],
        genes=dataset.genes[keep],
        samples=dataset.samples.copy(),
        name=f"{dataset.name} | mean>={min_mean}"
    )


def filter_top_variable_genes(dataset: ExpressionDataset, top_n: int) -> ExpressionDataset:
    """
    Keep the top_n genes with highest standard deviation across samples.
    """
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    top_n = min(top_n, dataset.n_genes)
    gene_std = dataset.X.std(axis=1)
    top_idx = np.argsort(gene_std)[::-1][:top_n]

    return ExpressionDataset(
        X=dataset.X[top_idx],
        genes=dataset.genes[top_idx],
        samples=dataset.samples.copy(),
        name=f"{dataset.name} | top_{top_n}_variable"
    )


def remove_samples(dataset: ExpressionDataset, sample_names: list) -> ExpressionDataset:
    """
    Remove specific samples by name.
    """
    mask = ~np.isin(dataset.samples, sample_names)
    removed = int(np.sum(~mask))

    print(f"Removed {removed} samples: {sample_names}")

    return ExpressionDataset(
        X=dataset.X[:, mask],
        genes=dataset.genes.copy(),
        samples=dataset.samples[mask],
        name=f"{dataset.name} | -{removed}_samples"
    )


def select_genes(dataset: ExpressionDataset, gene_names: list) -> ExpressionDataset:
    """
    Keep only genes in gene_names list.
    """
    mask = np.isin(dataset.genes, gene_names)
    selected = int(np.sum(mask))

    print(f"Selected {selected} / {dataset.n_genes} genes")

    return ExpressionDataset(
        X=dataset.X[mask],
        genes=dataset.genes[mask],
        samples=dataset.samples.copy(),
        name=f"{dataset.name} | {selected}_genes"
    )


def row_zscore_dataset(dataset: ExpressionDataset) -> ExpressionDataset:
    """
    Row-wise z-score normalisation:
    each gene is centered and scaled across samples.
    Useful for visualisation, but not a default preprocessing step for E.coli.
    """
    X = dataset.X
    X_new = (X - X.mean(axis=1, keepdims=True)) / (
        X.std(axis=1, keepdims=True) + 1e-8
    )

    return ExpressionDataset(
        X=X_new,
        genes=dataset.genes.copy(),
        samples=dataset.samples.copy(),
        name=f"{dataset.name} | row_zscore"
    )

def remove_genes(dataset: ExpressionDataset,
                 gene_names: list) -> ExpressionDataset:
    """Remove specific genes by name."""
    mask    = ~np.isin(dataset.genes, gene_names)
    removed = int(np.sum(~mask))
    print(f"Removed {removed} genes")

    return ExpressionDataset(
        X       = dataset.X[mask],
        genes   = dataset.genes[mask],
        samples = dataset.samples.copy(),
        name    = f"{dataset.name} | -{removed}_genes"
    )