# pca.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA


def pc1_vec_fun(X, gene_set, seed_sort, n=10):
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
    pca_matrix = X[I][:, S[:n]]   # genes x n seed samples

    # PCA on top n samples
    pca      = PCA(n_components=1)
    pca.fit(pca_matrix.T)         # fit on (n x genes)
    loadings = pca.components_.T  # genes x 1

    # least squares: project all samples onto PC1
    ones       = np.ones((len(I), 1))
    L          = np.hstack([ones, loadings])        # genes x 2
    LtL_inv_Lt = np.linalg.inv(L.T @ L) @ L.T      # 2 x genes
    coeffs     = LtL_inv_Lt @ top_gem              # 2 x n_samples
    pc1_vec    = coeffs[1, :]                       # slope = PC1

    # align sign: positive arm should dominate
    if abs(pc1_vec.min()) > abs(pc1_vec.max()):
        pc1_vec = -pc1_vec
        print("Sign flipped")

    print(f"PC1 shape: {pc1_vec.shape}")
    print(f"Min: {pc1_vec.min():.4f} | Max: {pc1_vec.max():.4f} | "
          f"Mean: {pc1_vec.mean():.4f}")

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


def plot_pc1_summary(pc1_vec, sample_alphas, n_seed=10,
                     outlier_pos=None, outlier_label=None):
    """
    3-panel PC1 summary plot:
      Panel 1 (top): fork plot coloured by PC1 value
      Panel 2 (bottom left): PC1 distribution histogram
      Panel 3 (bottom right): alpha decay vs PC1 values

    Parameters
    ----------
    pc1_vec       : np.ndarray  PC1 values in sample rank order
    sample_alphas : list        alpha values from SampleSort
    n_seed        : int         number of seed samples
    outlier_pos   : int         optional rank to highlight
    outlier_label : str         optional label for highlighted point
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel 1: fork plot ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(np.arange(len(pc1_vec)), pc1_vec,
                c=pc1_vec, cmap="RdBu_r", s=8, alpha=0.7)
    ax1.axhline(0,      color="black", linewidth=0.8, linestyle="--")
    ax1.axvline(n_seed, color="red",   linewidth=1.2, linestyle="--",
                label=f"End of seed (n={n_seed})")

    if outlier_pos is not None:
        label = outlier_label or f"rank {outlier_pos}"
        ax1.scatter(outlier_pos, pc1_vec[outlier_pos],
                    color="red", s=60, zorder=5,
                    label=f"{label} (PC1={pc1_vec[outlier_pos]:.1f})")

    ax1.set_xlabel("Sample rank (by SampleSort)")
    ax1.set_ylabel("PC1 value")
    ax1.set_title("PC1 vs Sample Ranking — Fork Pattern", fontsize=12)
    ax1.legend(fontsize=9)

    # ── Panel 2: PC1 distribution ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(pc1_vec, bins=50, color="steelblue",
             alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("PC1 value")
    ax2.set_ylabel("Count")
    ax2.set_title("PC1 Distribution", fontsize=11)

    upper = int((pc1_vec > 0).sum())
    lower = int((pc1_vec < 0).sum())
    ax2.text(0.98, 0.95,
             f"Upper: {upper}\nLower: {lower}",
             transform=ax2.transAxes,
             ha="right", va="top", fontsize=9)

    # ── Panel 3: alpha decay vs PC1 ───────────────────────────
    ax3      = fig.add_subplot(gs[1, 1])
    ax3_twin = ax3.twinx()

    ax3.plot(np.arange(len(sample_alphas)), sample_alphas,
             color="steelblue", lw=1.5, label="Alpha")
    ax3_twin.scatter(np.arange(len(pc1_vec)), pc1_vec,
                     c="coral", s=4, alpha=0.5, label="PC1")

    ax3.set_xlabel("Sample rank")
    ax3.set_ylabel("Alpha",  color="steelblue")
    ax3_twin.set_ylabel("PC1", color="coral")
    ax3.set_title("Alpha decay vs PC1 values", fontsize=11)
    ax3.axvline(n_seed, color="red", linestyle="--", linewidth=0.8)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    fig.suptitle("PC1VecFun — E. coli Bicluster",
                 fontsize=13, fontweight="bold")
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