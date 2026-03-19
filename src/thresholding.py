import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def threshold_bic(cor_vec, sort_order, pc1,
                  samp_sig=0, sample_names=None):
    """
    Define bicluster gene and sample sets from correlation vector and PC1.

    Equivalent to R's ThresholdBic.

    Gene thresholding  : k-means (k=2) on |correlation vector| values.
                         The cluster with higher mean |corr| is kept.
    Sample thresholding: bottom 10% of PC1 range defines the boundary.
                         samp_sig controls strictness (0=loose, 1=strict).

    Parameters
    ----------
    cor_vec      : np.ndarray  correlation vector (all genes)
    sort_order   : np.ndarray  ranked sample indices from SampleSort (0-indexed)
    pc1          : np.ndarray  PC1 values in sort_order sequence
    samp_sig     : float       strictness 0-1 (default 0, paper uses 0.8)
    sample_names : np.ndarray  optional sample name array for printing

    Returns
    -------
    bic_genes : np.ndarray  boolean mask of bicluster genes
    bic_samps : np.ndarray  sample indices in bicluster (or None)
    """
    # ── gene thresholding ─────────────────────────────────────
    abs_cv = np.abs(cor_vec).reshape(-1, 1)
    km     = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(abs_cv)
    labels = km.labels_

    mean0     = np.abs(cor_vec)[labels == 0].mean()
    mean1     = np.abs(cor_vec)[labels == 1].mean()
    bic_genes = (labels == 0) if mean0 > mean1 else (labels == 1)

    # ── sample thresholding ───────────────────────────────────
    n        = len(pc1)
    n_tail   = int(np.ceil(n / 10))
    pc1_tail = pc1[::-1][:n_tail]

    pc1_min = np.quantile(pc1_tail, 0 + samp_sig / 2)
    pc1_max = np.quantile(pc1_tail, 1 - samp_sig / 2)

    in_range      = np.where((pc1 > pc1_min) & (pc1 < pc1_max))[0]
    first_no_samp = int(in_range[0]) if len(in_range) > 0 else None
    bic_samps     = sort_order[:first_no_samp] \
                    if first_no_samp is not None else None

    # ── print summary ─────────────────────────────────────────
    print(f"Bicluster genes:   {bic_genes.sum()} / {len(cor_vec)}")
    print(f"pc1_min={pc1_min:.4f}, pc1_max={pc1_max:.4f}")
    print(f"First excluded rank: {first_no_samp}")
    print(f"Bicluster samples: "
          f"{len(bic_samps) if bic_samps is not None else 0}")

    # ── top / bottom 20 samples ───────────────────────────────
    def _print_table(ranks, label):
        print(f"\nPC1 values for {label}:")
        for rank, idx in ranks:
            s    = sort_order[rank]
            name = sample_names[s] if sample_names is not None else str(s)
            sign = "+" if pc1[rank] >= 0 else ""
            print(f"  Rank {rank+1:4d}: sample {s:4d} "
                  f"({name:35s}) | PC1 = {sign}{pc1[rank]:.4f}")

    top20    = [(i, sort_order[i]) for i in range(min(20, n))]
    bottom20 = [(n-1-i, sort_order[n-1-i])
                for i in range(min(20, n)-1, -1, -1)]

    _print_table(top20,    "top 20 samples")
    _print_table(bottom20, "bottom 20 samples")

    # ── bicluster boundary ────────────────────────────────────
    if first_no_samp is not None:
        print(f"\n--- Bicluster boundary at rank {first_no_samp} ---")
        for rank in range(max(0, first_no_samp - 3),
                          min(n, first_no_samp + 3)):
            s      = sort_order[rank]
            name   = sample_names[s] if sample_names is not None else str(s)
            marker = " ← boundary" if rank == first_no_samp else ""
            print(f"  Rank {rank+1:4d}: sample {s:4d} "
                  f"({name:35s}) | PC1 = {pc1[rank]:+.4f}{marker}")

    return bic_genes, bic_samps


def plot_threshold_summary(cor_vec, pc1, bic_genes,
                           bic_samps, sort_order,
                           sample_names=None):
    """
    Two-panel verification plot for the thresholding step.

    Panel 1: |correlation vector| distribution with gene threshold line.
    Panel 2: Fork plot with bicluster samples highlighted.

    Parameters
    ----------
    cor_vec      : np.ndarray  correlation vector (all genes)
    pc1          : np.ndarray  PC1 values in sort_order sequence
    bic_genes    : np.ndarray  boolean mask of bicluster genes
    bic_samps    : np.ndarray  bicluster sample indices
    sort_order   : np.ndarray  ranked sample indices
    sample_names : np.ndarray  optional sample name array
    """
    gene_threshold = np.abs(cor_vec)[bic_genes].min() \
                     if bic_genes.sum() > 0 else 0.0
    n_bic = len(bic_samps) if bic_samps is not None else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # panel 1: gene threshold
    axes[0].hist(np.abs(cor_vec), bins=100,
                 color="steelblue", edgecolor="none")
    axes[0].axvline(gene_threshold, color="red",
                    linestyle="--", lw=2,
                    label=f"Threshold = {gene_threshold:.4f}")
    axes[0].set_xlabel("|Correlation|")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Gene Thresholding\n"
                      f"{bic_genes.sum()} / {len(cor_vec)} genes kept")
    axes[0].legend()

    # panel 2: fork plot with bicluster highlighted
    bic_set = set(bic_samps.tolist()) if bic_samps is not None else set()
    colors  = ["red" if sort_order[i] in bic_set else "lightgray"
               for i in range(len(sort_order))]

    axes[1].scatter(range(len(pc1)), pc1,
                    c=colors, s=4, alpha=0.7)
    axes[1].axhline(0, color="gray", lw=0.8, linestyle="--")
    if n_bic > 0:
        axes[1].axvline(n_bic, color="green", lw=1.5,
                        linestyle="--",
                        label=f"Bicluster boundary (n={n_bic})")
    axes[1].set_xlabel("Sample rank")
    axes[1].set_ylabel("PC1 value")
    axes[1].set_title(f"Fork Plot — Bicluster Samples\n"
                      f"{n_bic} samples highlighted in red")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()