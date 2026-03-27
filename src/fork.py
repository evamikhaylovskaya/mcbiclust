# fork.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# pca.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA


def fork_classifier(pc1, samp_num):
    if samp_num < 2:
        print("Warning: samp_num < 2, cannot classify fork")
        return np.array(["None"] * len(pc1), dtype=object)

    pc1_bic = pc1[:samp_num].reshape(-1, 1)
    km      = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(pc1_bic)
    labels  = km.labels_

    g1 = (labels == 0)
    g2 = (labels == 1)

    if pc1[:samp_num][g1].mean() > pc1[:samp_num][g2].mean():
        upper, lower = g1, g2
    else:
        upper, lower = g2, g1

    fork_status = np.full(len(pc1), "None", dtype=object)

    for i in np.where(upper)[0]:
        fork_status[i] = "Upper"
    for i in np.where(lower)[0]:
        fork_status[i] = "Lower"

    print(f"Upper fork: {(fork_status == 'Upper').sum()} samples")
    print(f"Lower fork: {(fork_status == 'Lower').sum()} samples")
    print(f"None:       {(fork_status == 'None').sum()} samples")

    return fork_status

def print_fork_summary(pc1, fork_status, sort_order,
                       bic_samps, bic_genes, sample_names=None,
                       n_top=20):
    """
    Print bicluster summary and fork assignment for top samples.

    Parameters
    ----------
    pc1          : np.ndarray  aligned PC1 values
    fork_status  : np.ndarray  fork classification per sample
    sort_order   : np.ndarray  ranked sample indices
    bic_samps    : np.ndarray  bicluster sample indices
    bic_genes    : np.ndarray  boolean mask of bicluster genes
    sample_names : np.ndarray  optional sample name array
    n_top        : int         number of top samples to print
    """
    n_bic = len(bic_samps) if bic_samps is not None else 0

    print(f"\nBicluster summary:")
    print(f"  Total samples:     {len(sort_order)}")
    print(f"  Bicluster samples: {n_bic}")
    print(f"  Upper fork:        {(fork_status == 'Upper').sum()}")
    print(f"  Lower fork:        {(fork_status == 'Lower').sum()}")
    print(f"  Bicluster genes:   {bic_genes.sum()}")

    print(f"\nFork assignment for top {n_top} samples:")
    for i in range(min(n_top, len(sort_order))):
        s    = sort_order[i]
        name = sample_names[s] if sample_names is not None else str(s)
        print(f"  Rank {i+1:3d}: {name:40s} | "
              f"PC1={pc1[i]:+.2f} | {fork_status[i]}")


def plot_fork_classified(pc1, fork_status, sort_order,
                         bic_samps, n_seed=10):
    """
    Fork plot with upper/lower fork samples coloured.

    Parameters
    ----------
    pc1         : np.ndarray  aligned PC1 values
    fork_status : np.ndarray  fork classification per sample
    sort_order  : np.ndarray  ranked sample indices
    bic_samps   : np.ndarray  bicluster sample indices
    n_seed      : int         number of seed samples
    """
    colors = [
        "red"       if f == "Upper" else
        "blue"      if f == "Lower" else
        "lightgray" for f in fork_status
    ]

    n_bic = len(bic_samps) if bic_samps is not None else 0

    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(pc1)), pc1,
                c=colors, s=4, alpha=0.7)
    plt.axhline(0, color="gray", lw=0.8, linestyle="--")
    plt.axvline(n_seed, color="red", lw=1, linestyle="--",
                label=f"End of seed ({n_seed} samples)")
    if n_bic > 0:
        plt.axvline(n_bic, color="green", lw=1.5, linestyle="--",
                    label=f"Bicluster boundary (n={n_bic})")

    # legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color="red",       label=f"Upper fork ({(fork_status=='Upper').sum()})"),
        Patch(color="blue",      label=f"Lower fork ({(fork_status=='Lower').sum()})"),
        Patch(color="lightgray", label="Non-bicluster"),
    ]
    plt.legend(handles=handles, fontsize=8)

    plt.xlabel("Sample rank")
    plt.ylabel("PC1 value")
    plt.title(f"Fork Plot — Upper/Lower Classification\n"
              f"({n_bic} bicluster samples)")
    plt.tight_layout()
    plt.show()