import time
import numpy as np
from matplotlib import pyplot as plt

from correlation import avg_abs_corr_rows


def extend_bicluster_samples_fast(
    X,
    gene_set,
    initial_sample_set,
    progress_every=50,
    print_only_improvements=False,
    max_rank=None,
):
    """
    Extend and rank bicluster samples greedily by maximizing alpha.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (genes x samples)

    gene_set : array-like
        Gene indices used for evaluating alpha

    initial_sample_set : array-like
        Initial seed sample indices

    progress_every : int
        Print progress every N iterations

    print_only_improvements : bool
        If True, only print progress lines when alpha improves

    max_rank : int or None
        Maximum number of ranked samples to produce.
        If None, rank all samples.

    Returns
    -------
    ranked : np.ndarray
        Ranked sample indices, starting with the initial seed

    alphas : list[float]
        Alpha values after each added sample

    picked : list[tuple]
        Tuples of (sample_index, alpha, delta)
    """
    start_time = time.time()

    X = np.asarray(X, dtype=float)
    gene_indices = np.asarray(gene_set, dtype=int)
    initial_sample_set = np.asarray(initial_sample_set, dtype=int)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array (genes x samples)")
    if len(gene_indices) == 0:
        raise ValueError("gene_set must not be empty")
    if len(initial_sample_set) == 0:
        raise ValueError("initial_sample_set must not be empty")
    if len(np.unique(initial_sample_set)) != len(initial_sample_set):
        raise ValueError("initial_sample_set contains duplicates")
    if progress_every <= 0:
        raise ValueError("progress_every must be > 0")

    G = X[gene_indices]
    n_genes = len(gene_indices)
    n_total = X.shape[1]

    current = list(initial_sample_set)
    remaining = list(np.setdiff1d(np.arange(n_total), current))
    ranked = list(current)
    alphas = []
    picked = []

    total_steps = len(remaining)
    iteration = 0

    G_cur = G[:, current]
    initial_alpha = avg_abs_corr_rows(G_cur)

    print(f"Initial: {len(current)} samples, {n_genes} genes")
    print(f"Initial alpha: {initial_alpha:.6f}")
    print(f"Remaining samples to rank: {len(remaining)}")
    print()

    prev_alpha = initial_alpha

    while remaining:
        if max_rank is not None and len(ranked) >= max_rank:
            break

        iteration += 1
        n = len(current)

        G_cur = G[:, current]
        G_rem = G[:, remaining]

        cur_sum = G_cur.sum(axis=1)
        cur_sum_sq = (G_cur ** 2).sum(axis=1)

        cand_sum = cur_sum[:, None] + G_rem
        cand_sum_sq = cur_sum_sq[:, None] + G_rem**2

        total_n = n + 1
        cand_mean = cand_sum / total_n
        cand_var = (cand_sum_sq / total_n) - cand_mean**2
        cand_std = np.sqrt(np.maximum(cand_var, 1e-16)) + 1e-8

        best_alpha = -np.inf
        best_idx = None

        for i in range(len(remaining)):
            mu_i = cand_mean[:, i]
            sd_i = cand_std[:, i]

            G_full = np.hstack([G_cur, G_rem[:, i:i + 1]])
            Z_full = (G_full - mu_i[:, None]) / sd_i[:, None]

            C = (Z_full @ Z_full.T) / total_n
            C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

            alpha_i = np.abs(C).sum() / (n_genes ** 2)

            if alpha_i > best_alpha:
                best_alpha = alpha_i
                best_idx = i

        best_sample = remaining[best_idx]
        delta = best_alpha - prev_alpha
        prev_alpha = best_alpha

        current.append(best_sample)
        remaining.pop(best_idx)
        ranked.append(best_sample)
        alphas.append(float(best_alpha))
        picked.append((int(best_sample), float(best_alpha), float(delta)))

        if iteration % progress_every == 0 or len(remaining) == 0:
            elapsed = time.time() - start_time
            steps_left = total_steps - iteration
            avg_time = elapsed / iteration
            eta = avg_time * steps_left
            elapsed_min = elapsed / 60
            eta_min = eta / 60

            if (not print_only_improvements) or (delta > 0):
                print(
                    f"Added sample {best_sample:4d} | Alpha: {best_alpha:.6f} | "
                    f"Delta: {delta:+.6f} | "
                    f"Progress: {len(ranked)}/{n_total} | "
                    f"Elapsed: {elapsed_min:.1f} min | ETA: {eta_min:.1f} min"
                )

    total_elapsed = time.time() - start_time

    if len(alphas) > 0:
        print(f"\nFinal alpha: {alphas[-1]:.6f}")
        print(f"Alpha improvement: {alphas[-1] - initial_alpha:.6f}")
    else:
        print("\nNo additional samples were ranked.")

    print(f"Total time: {total_elapsed/60:.1f} minutes")

    return np.array(ranked), alphas, picked



def plot_sample_extension_summary(
    X,
    gene_set,
    ranked_samples,
    sample_alphas,
    sample_names=None,
    global_alpha=None,
    n_seed=10,
    top_n_list=(10, 20, 50, 100, 150, 200)
    ):
    """
    Summary plot and statistics for the sample extension step.

    Parameters
    ----------
    X             : np.ndarray  expression matrix (genes x samples)
    gene_set      : np.ndarray  pruned gene indices
    ranked_samples: np.ndarray  ranked sample indices from SampleSort
    sample_alphas : list        alpha values during ranking
    sample_names  : np.ndarray  optional sample name array
    global_alpha  : float       optional global alpha reference line
    n_seed        : int         number of seed samples (default 10)
    top_n_list    : tuple       sample cutoffs to evaluate alpha at
    """

    # ── alpha decay plot ──────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(sample_alphas, color="steelblue", lw=1.5)
    plt.axvline(x=n_seed, color="red", linestyle="--",
                label=f"End of seed ({n_seed} samples)")
    plt.axhline(y=sample_alphas[0], color="green", linestyle="--",
                label=f"Seed alpha = {sample_alphas[0]:.4f}")
    if global_alpha is not None:
        plt.axhline(y=global_alpha, color="orange", linestyle="--",
                    label=f"Global alpha = {global_alpha:.4f}")
    plt.xlabel("Number of samples ranked")
    plt.ylabel("Alpha")
    plt.title("Alpha decay during sample extension")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ── alpha at cutoffs ──────────────────────────────────────
    print("Alpha at different sample cutoffs:")
    for n in top_n_list:
        if n <= len(ranked_samples):
            a = avg_abs_corr_rows(X[gene_set][:, ranked_samples[:n]])
            print(f"  Top {n:4d} samples: alpha = {a:.6f}")

    # ── threshold crossings ───────────────────────────────────
    thresholds = [0.7, 0.6, 0.5, 0.4, 0.3, 0.25]
    print("\nAlpha threshold crossings:")
    for t in thresholds:
        below = np.where(np.array(sample_alphas) < t)[0]
        if len(below) > 0:
            print(f"  alpha < {t:.2f} first at rank {below[0] + 1}")

    # ── top and bottom sample names ───────────────────────────
    if sample_names is not None:
        print(f"\nTop 20 samples:")
        for i, s in enumerate(ranked_samples[:20]):
            print(f"  Rank {i+1:3d}: sample {s:4d} -> {sample_names[s]}")

        print(f"\nBottom 20 samples:")
        for i, s in enumerate(ranked_samples[-20:]):
            rank = len(ranked_samples) - 19 + i
            print(f"  Rank {rank:3d}: sample {s:4d} -> {sample_names[s]}")

    # ── summary stats ─────────────────────────────────────────
    print(f"\nSummary:")
    print(f"  Total samples ranked: {len(ranked_samples)}")
    print(f"  Seed alpha:           {sample_alphas[0]:.4f}")
    print(f"  Final alpha:          {sample_alphas[-1]:.4f}")
    print(f"  Alpha drop:           {sample_alphas[0] - sample_alphas[-1]:.4f}")

