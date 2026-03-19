import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from correlation import avg_abs_corr_rows


def plot_pruning_summary(
    X,
    initial_gene_set,
    pruned_gene_set,
    best_samples,
    local_idx_py,
    cluster_labels_py,
    group_alphas,
    best_alpha,
    dataset_name="Dataset",
):
    """
    4-panel verification plot for the gene pruning step.
    """
    X = np.asarray(X, dtype=float)
    seed = np.asarray(best_samples, dtype=int)
    I_full = np.asarray(initial_gene_set, dtype=int)
    I_prun = np.asarray(pruned_gene_set, dtype=int)

    def _corr(A):
        C = np.corrcoef(A)
        return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    alpha_pruned = float(avg_abs_corr_rows(X[I_prun][:, seed]))

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # Panel 1
    ax1 = fig.add_subplot(gs[0, 0])
    C_before = _corr(X[I_full][:, seed].T)
    im1 = ax1.imshow(C_before, cmap="RdBu_r", vmin=0.4, vmax=1, aspect="auto")
    ax1.set_title("Sample-Sample Correlation\n(seed samples, full gene set)", fontsize=11)
    ax1.set_xlabel("Seed samples")
    ax1.set_ylabel("Seed samples")
    plt.colorbar(im1, ax=ax1, label="Pearson r")

    # Panel 2
    ax2 = fig.add_subplot(gs[0, 1])
    C_after = _corr(X[I_prun][:, seed].T)
    im2 = ax2.imshow(C_after, cmap="RdBu_r", vmin=0.4, vmax=1, aspect="auto")
    ax2.set_title("Sample-Sample Correlation\n(seed samples, pruned gene set)", fontsize=11)
    ax2.set_xlabel("Seed samples")
    ax2.set_ylabel("Seed samples")
    plt.colorbar(im2, ax=ax2, label="Pearson r")

    # Panel 3
    ax3 = fig.add_subplot(gs[1, 0])
    group_ids = sorted(group_alphas.keys())
    alpha_vals = [group_alphas[g]["alpha"] for g in group_ids]
    sizes = [group_alphas[g]["n_genes"] for g in group_ids]
    colors = [
        "steelblue" if group_alphas[g]["alpha"] > best_alpha else "lightcoral"
        for g in group_ids
    ]

    bars = ax3.bar([str(g) for g in group_ids], alpha_vals, color=colors, width=0.6)
    ax3.axhline(best_alpha, color="red", linestyle="--", linewidth=1.5)

    for bar, size in zip(bars, sizes):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"n={size}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax3.set_xlabel("Cluster group")
    ax3.set_ylabel("Alpha")
    ax3.set_title("Gene Pruning: Alpha per Hierarchical Group", fontsize=11)
    ax3.set_ylim(0, max(alpha_vals) * 1.12)
    ax3.legend(
        handles=[
            Patch(color="steelblue", label="Kept"),
            Patch(color="lightcoral", label="Removed"),
            plt.Line2D([0], [0], color="red", linestyle="--",
                       label=f"α baseline = {best_alpha:.4f}")
        ],
        fontsize=9,
    )

    # Panel 4
    ax4 = fig.add_subplot(gs[1, 1])
    labels_kept = cluster_labels_py[local_idx_py]
    order = np.argsort(labels_kept)
    C_gg = _corr(X[I_prun[order]][:, seed])

    im4 = ax4.imshow(C_gg, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax4.set_title(
        f"Gene-Gene Correlation\n(pruned {len(I_prun)} genes, ordered by group)",
        fontsize=11,
    )
    ax4.set_xlabel("Genes")
    ax4.set_ylabel("Genes")
    ax4.set_xticks([])
    ax4.set_yticks([])
    plt.colorbar(im4, ax=ax4, label="Pearson r")

    boundaries = np.cumsum(np.bincount(labels_kept[order])[1:])[:-1]
    for b in boundaries:
        ax4.axvline(b - 0.5, color="black", linewidth=0.6)
        ax4.axhline(b - 0.5, color="black", linewidth=0.6)

    fig.suptitle(
        f"MCbiclust Seed Bicluster Summary | {dataset_name}\n"
        f"Genes: {len(I_full)} → {len(I_prun)} after pruning   "
        f"α: {best_alpha:.4f} → {alpha_pruned:.4f}",
        fontsize=13,
        fontweight="bold",
    )

    plt.show()


def plot_pruned_gene_correlation_heatmap(
    X,
    gene_set,
    sample_set,
    title="Pruned Gene Correlation Heatmap",
    max_genes=None,
    random_state=None,
    figsize=(14, 10),
    cmap="RdBu_r",
    linkage_method="complete",
    color_threshold=0.4,
):
    """
    Plot gene-gene correlation heatmap with dendrograms and correlation histogram.
    Useful as a pruning-stage diagnostic plot.
    """
    gene_indices = np.asarray(gene_set, dtype=int)
    sample_indices = np.asarray(sample_set, dtype=int)
    X = np.asarray(X, dtype=float)

    data = X[gene_indices][:, sample_indices]
    data_norm = (data - data.mean(axis=1, keepdims=True)) / (
        data.std(axis=1, keepdims=True) + 1e-8
    )

    corr_matrix = np.corrcoef(data_norm)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if max_genes is not None and len(gene_indices) > max_genes:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(gene_indices), max_genes, replace=False)
        corr_matrix = corr_matrix[np.ix_(sample_idx, sample_idx)]
        gene_indices = gene_indices[sample_idx]

    dist_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0.0)
    dist_vec = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(dist_vec, method=linkage_method)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        3, 3,
        hspace=0.05,
        wspace=0.05,
        height_ratios=[0.5, 0.1, 2],
        width_ratios=[0.5, 0.1, 2]
    )

    dendro_top_info = dendrogram(
        linkage_matrix,
        orientation="top",
        no_plot=True
    )
    reorder_idx = dendro_top_info["leaves"]
    corr_reordered = corr_matrix[np.ix_(reorder_idx, reorder_idx)]

    # top dendrogram
    ax_dendro_top = fig.add_subplot(gs[0, 2])
    dendrogram(
        linkage_matrix,
        ax=ax_dendro_top,
        orientation="top",
        no_labels=True,
        color_threshold=color_threshold,
        above_threshold_color="black"
    )
    ax_dendro_top.set_xticks([])
    ax_dendro_top.set_yticks([])

    # left dendrogram
    ax_dendro_left = fig.add_subplot(gs[2, 0])
    dendrogram(
        linkage_matrix,
        ax=ax_dendro_left,
        orientation="left",
        no_labels=True,
        color_threshold=color_threshold,
        above_threshold_color="black"
    )
    ax_dendro_left.set_xticks([])
    ax_dendro_left.set_yticks([])

    # heatmap
    ax_heatmap = fig.add_subplot(gs[2, 2])
    im = ax_heatmap.imshow(
        corr_reordered,
        aspect="auto",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation="none"
    )

    ax_cbar = fig.add_subplot(gs[2, 1])
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Pearson Correlation", rotation=270, labelpad=20)

    # histogram
    ax_hist = fig.add_subplot(gs[0, 0])
    corr_flat = corr_reordered[np.triu_indices_from(corr_reordered, k=1)]
    ax_hist.hist(corr_flat, bins=50, color="gray", alpha=0.7, edgecolor="black")
    ax_hist.set_xlabel("Correlation")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution")
    ax_hist.axvline(0, color="red", linestyle="--", linewidth=1)

    ax_heatmap.set_title(title, fontsize=14, pad=20)
    ax_heatmap.set_xlabel("Genes")
    ax_heatmap.set_ylabel("Genes")

    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95)
    plt.show()

    return corr_matrix, linkage_matrix