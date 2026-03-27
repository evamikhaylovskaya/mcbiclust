import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_alpha_vs_iterations(steps, alpha_history, title="FindSeed: Alpha Improvement"):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, alpha_history, marker="o", lw=2)
    plt.axhline(y=max(alpha_history), color="red", linestyle="--")
    plt.xlabel("Iterations")
    plt.ylabel("Alpha")
    plt.title(title)
    plt.legend([f"Final alpha = {max(alpha_history):.4f}"])
    plt.tight_layout()
    plt.show()


def plot_sample_corr_heatmap(X, subset_size=100, seed=42, cluster=False, vmin=0.0, vmax=1.0):
    rng = np.random.default_rng(seed)
    subset_size = min(subset_size, X.shape[1])

    sample_idx = np.sort(rng.choice(X.shape[1], subset_size, replace=False))
    X_sample_sub = X[:, sample_idx]
    C_sample_sub = np.corrcoef(X_sample_sub, rowvar=False)

    if cluster:
        g = sns.clustermap(
            C_sample_sub,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            figsize=(7, 6),
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "Pearson r"},
        )
        g.fig.suptitle(
            f"Sample-Sample Correlation (random {subset_size} samples)",
            y=1.02,
        )
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            C_sample_sub,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            cbar_kws={"label": "Pearson r"},
            ax=ax,
        )
        ax.set_title(f"Sample-Sample Correlation (random {subset_size} samples)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Samples")

        ticks = np.arange(0, subset_size, max(1, subset_size // 25))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
        ax.invert_yaxis()

        plt.tight_layout()
        plt.show()
        
        
def plot_gene_corr_heatmap(
    X,
    sample_set,
    gene_names=None,
    title="Gene-gene correlation",
    max_genes=None,
    random_state=42,
    method="average",
    metric="euclidean",
    cmap="hot",
    vmin=-1,
    vmax=1,
    figsize=(16, 10),
):
    """
    Approximate R heatmap.2(cor(t(X[, sample_set])))

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (genes x samples)
    sample_set : array-like
        Selected sample indices
    gene_names : array-like or None
        Gene labels for rows/columns
    max_genes : int or None
        Optional subsample of genes for faster plotting
    """
    X = np.asarray(X, dtype=float)
    sample_set = np.asarray(sample_set, dtype=int)

    # genes x selected samples
    X_sub = X[:, sample_set]

    n_genes = X_sub.shape[0]

    if gene_names is None:
        gene_names = np.array([f"G{i}" for i in range(n_genes)])
    else:
        gene_names = np.asarray(gene_names)

    # optional gene subsampling for speed
    if max_genes is not None and n_genes > max_genes:
        rng = np.random.default_rng(random_state)
        keep = np.sort(rng.choice(n_genes, size=max_genes, replace=False))
        X_sub = X_sub[keep]
        gene_names = gene_names[keep]

    # R: cor(t(X[, sample_set])) -> gene-gene correlation
    C = np.corrcoef(X_sub)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    df = pd.DataFrame(C, index=gene_names, columns=gene_names)

    g = sns.clustermap(
        df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        method=method,
        metric=metric,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.22, 0.22),
        cbar_pos=(0.02, 0.82, 0.20, 0.05),   # top-left like R-ish key
        cbar_kws={"orientation": "horizontal", "label": "Value"},
    )

    g.ax_heatmap.set_title(title, fontsize=22, weight="bold", pad=28)
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    # label styling
    g.ax_heatmap.tick_params(axis="x", labelsize=8, rotation=90)
    g.ax_heatmap.tick_params(axis="y", labelsize=8, rotation=0)

    # make dendrogram lines black-ish like R
    for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
        for coll in ax.collections:
            coll.set_color("k")
            coll.set_linewidth(1.0)

    plt.show()
    return g