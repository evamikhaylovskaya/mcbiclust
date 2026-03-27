import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples


def silhouette_clust_groups(
    cor_vec_mat,
    max_clusters=10,
    plots=True,
    seed1=100,
    rand_vec=True,
    min_cluster_size=1,
):
    """
    Identify distinct biclusters from multiple MCbiclust runs
    using silhouette analysis on correlation vectors.

    Parameters
    ----------
    cor_vec_mat : np.ndarray
        genes x runs matrix of correlation vectors
    max_clusters : int
        maximum number of clusters to try
    plots : bool
        whether to show silhouette and heatmap plots
    seed1 : int
        random seed for noise vector
    rand_vec : bool
        whether to add a random noise vector as control
    min_cluster_size : int
        minimum number of runs required to keep a cluster

    Returns
    -------
    cluster_groups : list of np.ndarray
        Boolean arrays indicating which runs belong to each real cluster
    """
    n_runs = cor_vec_mat.shape[1]

    if n_runs < 2:
        raise ValueError("Need at least 2 runs for silhouette clustering.")

    if max_clusters >= n_runs:
        max_clusters = n_runs - 1

    # ── add random noise vector as control ────────────────────
    if rand_vec:
        rng = np.random.default_rng(seed1)
        noise = rng.normal(0, 1, size=(cor_vec_mat.shape[0], 1))
        mat = np.hstack([cor_vec_mat, noise])
        noise_col = n_runs
    else:
        mat = cor_vec_mat
        noise_col = None

    # ── distance matrix: 1 - |cor(runs)| ──────────────────────
    C = np.corrcoef(mat.T)
    C = np.nan_to_num(C, nan=0.0)
    dist_mat = 1 - np.abs(C)
    np.fill_diagonal(dist_mat, 0.0)

    # force symmetry
    dist_mat = (dist_mat + dist_mat.T) / 2
    dist_vec = squareform(dist_mat, checks=False)

    # ── hierarchical clustering ───────────────────────────────
    Z = linkage(dist_vec, method="complete")

    # ── silhouette for k=2..max_clusters ──────────────────────
    sil_values = []
    labels_by_k = {}

    for k in range(2, max_clusters + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        labels_by_k[k] = labels
        si = silhouette_samples(dist_mat, labels, metric="precomputed")
        sil_values.append(float(si.mean()))

    best_k = int(np.argmax(sil_values)) + 2
    labels_best = labels_by_k[best_k]

    print(f"Silhouette values (k=2..{max_clusters}):")
    for k, sv in enumerate(sil_values, start=2):
        marker = " ← best" if k == best_k else ""
        print(f"  k={k:2d}: {sv:.4f}{marker}")

    # ── raw clusters before noise removal ─────────────────────
    raw_groups = [labels_best == g for g in range(1, best_k + 1)]

    # ── remove noise cluster ──────────────────────────────────
    removed_noise_cluster = None
    if rand_vec and noise_col is not None:
        for i, grp in enumerate(raw_groups):
            if grp[noise_col]:
                removed_noise_cluster = i
                break

        if removed_noise_cluster is not None:
            print(f"Removing noise cluster: {removed_noise_cluster + 1}")
            raw_groups.pop(removed_noise_cluster)

    # ── trim off noise column and drop tiny clusters ──────────
    cluster_groups = []
    for grp in raw_groups:
        grp_runs = grp[:n_runs]  # remove noise column if present
        if int(grp_runs.sum()) >= min_cluster_size:
            cluster_groups.append(grp_runs)

    print(f"\nBest k before noise removal: {best_k}")
    print(f"Distinct biclusters found: {len(cluster_groups)}")

    if len(cluster_groups) == 0:
        print("No valid biclusters remained after removing noise/tiny clusters.")
        return []

    for i, grp in enumerate(cluster_groups):
        run_ids = np.where(grp)[0].tolist()
        print(f"  Cluster {i+1}: {int(grp.sum())} runs → {run_ids}")

    if plots:
        _plot_silhouette_and_heatmap(
            sil_values=sil_values,
            best_k=best_k,
            C=C,
            labels=labels_best,
            max_clusters=max_clusters,
        )

    return cluster_groups


def _plot_silhouette_and_heatmap(sil_values, best_k, C, labels, max_clusters):
    """Internal: silhouette score plot + run-run correlation heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        range(2, max_clusters + 1),
        sil_values,
        marker="o",
        color="steelblue",
        lw=2
    )
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Mean silhouette width")
    axes[0].set_title("Silhouette Analysis")
    axes[0].legend()

    order = np.argsort(labels)
    C_ordered = np.abs(C)[np.ix_(order, order)]

    im = axes[1].imshow(C_ordered, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("|Correlation| between runs\n(ordered by cluster)")
    axes[1].set_xlabel("Run")
    axes[1].set_ylabel("Run")
    plt.colorbar(im, ax=axes[1], label="|Pearson r|")

    plt.tight_layout()
    plt.show()


def plot_silhouette(cor_vec_mat, cluster_groups, title="Silhouette plot of biclusters"):
    """
    Paper-style silhouette plot for the identified biclusters.

    Parameters
    ----------
    cor_vec_mat : np.ndarray
        genes x runs matrix
    cluster_groups : list
        boolean arrays from silhouette_clust_groups
    title : str
        plot title
    """
    if len(cluster_groups) <= 1:
        print("Only 1 distinct bicluster found — silhouette plot not applicable.")
        print("All runs converged to the same bicluster.")
        return

    # recompute distance matrix on runs only (no noise vector)
    C = np.corrcoef(cor_vec_mat.T)
    C = np.nan_to_num(C, nan=0.0)
    dist_mat = 1 - np.abs(C)
    np.fill_diagonal(dist_mat, 0.0)
    dist_mat = (dist_mat + dist_mat.T) / 2

    dist_vec = squareform(dist_mat, checks=False)
    Z = linkage(dist_vec, method="complete")
    labels = fcluster(Z, t=len(cluster_groups), criterion="maxclust")

    n = len(labels)
    k = len(cluster_groups)
    si = silhouette_samples(dist_mat, labels, metric="precomputed")
    mean_sil = float(si.mean())

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("lightgrey")
    ax.set_facecolor("lightgrey")

    y_lower = 0
    gap = 2

    for i, grp in enumerate(cluster_groups):
        cluster_label = i + 1
        mask = labels == cluster_label
        si_cluster = np.sort(si[mask])[::-1]
        n_cluster = len(si_cluster)
        y_upper = y_lower + n_cluster

        ax.barh(
            y=np.arange(y_lower, y_upper),
            width=si_cluster,
            height=1.0,
            left=0,
            color="red",
            edgecolor="none"
        )

        mid = y_lower + n_cluster / 2
        avg_si = float(si_cluster.mean())

        ax.text(
            -0.05, mid, f"C{cluster_label}",
            ha="right", va="center",
            fontsize=11, fontweight="bold", color="white"
        )
        ax.text(
            1.02, mid,
            f"{cluster_label}: {n_cluster:3d} | {avg_si:.2f}",
            ha="left", va="center", fontsize=9
        )

        y_lower = y_upper + gap

    ax.axvline(mean_sil, color="white", linestyle="-", linewidth=1.5,
               label=f"Mean = {mean_sil:.2f}")
    ax.set_xlim(-0.5, 1.05)
    ax.set_ylim(-gap, y_lower)
    ax.set_xlabel("Silhouette width $s_i$", fontsize=11)
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    ax.text(
        0.98, 0.98,
        f"{k} clusters $C_j$\n"
        f"j : $n_j$ | ave$_{{i\\in C_j}}$ $s_i$",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )
    ax.text(
        0.02, 0.98, f"n = {n}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9
    )

    plt.tight_layout()
    plt.show()
    
    
def average_corvec_per_cluster(cor_vec_mat, cluster_groups):
    """
    Compute the average correlation vector for each cluster.

    Parameters
    ----------
    cor_vec_mat : np.ndarray
        genes x runs matrix
    cluster_groups : list of np.ndarray
        Boolean masks indicating which runs belong to each cluster

    Returns
    -------
    av_corvec : list of np.ndarray
        Average correlation vector for each cluster
    cluster_sizes : list of int
        Number of runs in each cluster
    """
    av_corvec = []
    cluster_sizes = []

    for grp in cluster_groups:
        grp = np.asarray(grp, dtype=bool)
        if grp.sum() == 0:
            continue

        cluster_mat = cor_vec_mat[:, grp]
        av_corvec.append(cluster_mat.mean(axis=1))
        cluster_sizes.append(int(grp.sum()))

    return av_corvec, cluster_sizes