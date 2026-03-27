import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def cv_plot(cor_vec_list, geneset_loc, geneset_name,
            alpha1=0.005, alpha2=0.1, cnames=None,
            figsize=None):
    """
    Plot correlation vectors against each other for multiple biclusters.

    Equivalent to R's CVPlot.

    Creates an n x n grid where:
      - Diagonal:        density histogram of correlation vector values
      - Lower triangle:  scatter plot of non-geneset genes
      - Upper triangle:  scatter plot of geneset genes

    Parameters
    ----------
    cor_vec_list : list        correlation vectors, one per bicluster
    geneset_loc  : array-like  boolean mask or indices of geneset genes
    geneset_name : str         name of the geneset (e.g. 'Mitochondrial')
    alpha1       : float       transparency for non-geneset scatter (default 0.005)
    alpha2       : float       transparency for geneset scatter (default 0.1)
    cnames       : list        optional names for each correlation vector
    figsize      : tuple       optional figure size
    """
    n_cv    = len(cor_vec_list)
    n_genes = len(cor_vec_list[0])

    if cnames is None or len(cnames) != n_cv:
        cnames = [f"CV{i+1}" for i in range(n_cv)]

    # geneset mask
    mask = np.zeros(n_genes, dtype=bool)
    mask[geneset_loc] = True
    non_mask = ~mask

    # colours matching R's hue_pal
    col_geneset    = "#F8766D"
    col_nongeneset = "#00BFC4"

    if figsize is None:
        figsize = (4 * n_cv, 4 * n_cv)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(n_cv, n_cv, figure=fig,
                            hspace=0.3, wspace=0.3)

    for row in range(n_cv):
        for col in range(n_cv):

            ax = fig.add_subplot(gs[row, col])

            # ── diagonal: density histogram ───────────────────────
            if row == col:
                cv = cor_vec_list[row]
                ax.hist(cv[non_mask], bins=60, density=True,
                        color=col_nongeneset, alpha=0.5,
                        label=f"Non {geneset_name}")
                ax.hist(cv[mask], bins=60, density=True,
                        color=col_geneset, alpha=0.5,
                        label=geneset_name)
                ax.set_xlabel(cnames[row], fontsize=9)
                ax.set_ylabel("Density", fontsize=8)
                ax.set_title(cnames[row], fontsize=10, fontweight="bold")
                if row == 0:
                    ax.legend(fontsize=7, loc="upper left")

            # ── lower triangle: non-geneset genes ─────────────────
            elif row > col:
                x = cor_vec_list[col]
                y = cor_vec_list[row]
                ax.scatter(x[non_mask], y[non_mask],
                           s=2, alpha=alpha1,
                           color=col_nongeneset, rasterized=True)
                ax.set_xlabel(cnames[col], fontsize=8)
                ax.set_ylabel(cnames[row], fontsize=8)
                ax.axhline(0, color="grey", lw=0.5, linestyle="--")
                ax.axvline(0, color="grey", lw=0.5, linestyle="--")
                r = np.corrcoef(x[non_mask], y[non_mask])[0, 1]
                ax.text(0.05, 0.92, f"r={r:.2f}",
                        transform=ax.transAxes, fontsize=8,
                        color=col_nongeneset)

            # ── upper triangle: geneset genes ─────────────────────
            else:
                x = cor_vec_list[col]
                y = cor_vec_list[row]
                ax.scatter(x[mask], y[mask],
                           s=4, alpha=alpha2,
                           color=col_geneset, rasterized=True)
                ax.set_xlabel(cnames[col], fontsize=8)
                ax.set_ylabel(cnames[row], fontsize=8)
                ax.axhline(0, color="grey", lw=0.5, linestyle="--")
                ax.axvline(0, color="grey", lw=0.5, linestyle="--")
                r = np.corrcoef(x[mask], y[mask])[0, 1]
                ax.text(0.05, 0.92, f"r={r:.2f}",
                        transform=ax.transAxes, fontsize=8,
                        color=col_geneset)

            ax.tick_params(labelsize=7)

    fig.suptitle("Correlation Vector Plot", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
    
    

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