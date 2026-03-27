import numpy as np

from correlation import avg_abs_corr_rows


def find_seed_bicluster(
    X,
    gene_set,
    n_samples=10,
    iterations=1000,
    random_state=None,
    initial_seed=None,
    log_improvements=True,
    print_every_improvement=False,
    print_every_iter=False,
    verbose=False,
):
    """
    Greedy stochastic search for a seed sample set that maximizes
    average absolute gene-gene correlation (alpha) over a fixed gene set.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix of shape (genes x samples)

    gene_set : array-like
        Gene indices used to evaluate alpha

    n_samples : int
        Number of samples in the seed

    iterations : int
        Number of greedy search iterations

    random_state : int or None
        Random seed for reproducibility

    initial_seed : array-like or None
        Optional starting sample indices

    log_improvements : bool
        Whether to store improvement records

    print_every_improvement : bool
        Print only accepted improvements

    print_every_iter : bool
        Print every candidate iteration

    Returns
    -------
    best_J : np.ndarray
        Best sample indices found

    best_alpha : float
        Best alpha score

    improvements : list[dict]
        Logged improvement steps

    history : list[float]
        Monotonic alpha history
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)

    n_total_samples = X.shape[1]
    I = np.asarray(gene_set, dtype=int)

    if len(I) == 0:
        raise ValueError("gene_set must not be empty")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if n_samples > n_total_samples:
        raise ValueError("n_samples cannot exceed total number of samples")
    if iterations < 0:
        raise ValueError("iterations must be >= 0")

    X_subset = X[I, :]

    if initial_seed is None:
        J = np.sort(rng.choice(n_total_samples, size=n_samples, replace=False))
    else:
        J = np.sort(np.asarray(initial_seed, dtype=int))

    if len(J) != n_samples:
        raise ValueError(f"Expected {n_samples} samples, got {len(J)}")
    if len(np.unique(J)) != len(J):
        raise ValueError("Duplicate samples in initial seed")
    if not np.all((J >= 0) & (J < n_total_samples)):
        raise ValueError("Initial seed contains out-of-range sample indices")

    def calculate_alpha(sample_idx):
        A = X_subset[:, sample_idx]
        return float(avg_abs_corr_rows(A))

    best_J = J.copy()
    best_alpha = calculate_alpha(best_J)
    initial_alpha = best_alpha
    history = [best_alpha]
    history_iterations = [0]
    improvements = []

    if verbose:
        print(f"Using {len(I)} genes, {n_samples} samples")
        print(f"Initial samples: {best_J.tolist()}")
        print(f"Initial alpha ({n_samples} samples, {len(I)} genes) = {best_alpha:.6f}")

    for iteration in range(1, iterations + 1):
        idx_to_replace = rng.integers(0, n_samples)
        all_samples = np.arange(n_total_samples)
        outside = np.setdiff1d(all_samples, J, assume_unique=True)

        if outside.size == 0:
            break

        new_sample = int(rng.choice(outside))

        J_candidate = J.copy()
        old_sample = int(J_candidate[idx_to_replace])
        J_candidate[idx_to_replace] = new_sample
        J_candidate.sort()

        if len(np.unique(J_candidate)) != len(J_candidate):
            raise ValueError(f"Duplicate samples in candidate at iteration {iteration}")
        if not np.all((J_candidate >= 0) & (J_candidate < n_total_samples)):
            raise ValueError(f"Out-of-range candidate sample at iteration {iteration}")

        candidate_alpha = calculate_alpha(J_candidate)

        if candidate_alpha > best_alpha:
            prev_alpha = best_alpha

            J = J_candidate
            best_J = J_candidate.copy()
            best_alpha = candidate_alpha
            history.append(best_alpha)
            history_iterations.append(iteration)

            if log_improvements:
                improvements.append(
                    {
                        "iteration": iteration,
                        "alpha": best_alpha,
                        "replaced_position": int(idx_to_replace),
                        "old_sample": old_sample,
                        "new_sample": new_sample,
                        "J": best_J.copy(),
                    }
                )

            if best_alpha <= prev_alpha:
                raise AssertionError(
                    f"Alpha did not strictly improve at iteration {iteration}: "
                    f"{prev_alpha} -> {best_alpha}"
                )

            if print_every_improvement:
                print(
                    f"[Improved @ {iteration:4d}] "
                    f"pos {idx_to_replace}: {old_sample} -> {new_sample} | "
                    f"alpha = {best_alpha:.6f} | J = {J.tolist()}"
                )

            status = "↑ IMPROVED"
        else:
            status = "rejected"

        if print_every_iter:
            print(
                f"[Iter {iteration:4d}] "
                f"replace pos {idx_to_replace} -> {new_sample:4d} | "
                f"candidate alpha = {candidate_alpha:.6f} ({status})"
            )
            
        if iteration % 100 == 0:
            print(f"Iterations: {iteration:4d} | Alpha: {best_alpha:.6f}")

    alphas = [imp["alpha"] for imp in improvements]
    if not all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1)):
        raise AssertionError("Alpha decreased across improvements")

    if not all(history[i] <= history[i + 1] for i in range(len(history) - 1)):
        raise AssertionError("History is not monotonic")

    if len(set(best_J)) != len(best_J):
        raise AssertionError("Duplicate samples in final seed")

    recomputed = float(avg_abs_corr_rows(X[I][:, best_J]))
    if not np.isclose(recomputed, best_alpha):
        raise AssertionError(
            f"Alpha mismatch: recomputed={recomputed} vs stored={best_alpha}"
        )

    if len(improvements) > 0 and not (improvements[0]["alpha"] > initial_alpha):
        raise AssertionError(
            f"First logged improvement ({improvements[0]['alpha']}) "
            f"is not greater than initial alpha ({initial_alpha})"
        )

    if verbose:
        print(f"Final alpha verified: {recomputed:.6f}")
        print("All debug checks passed.")

    return best_J, float(best_alpha), improvements, history, history_iterations


