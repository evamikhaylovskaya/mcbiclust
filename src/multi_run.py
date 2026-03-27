# multi_run.py
import numpy as np
from find_seed_sample import find_seed_bicluster
from pruning import prune_bicluster_genes
from correlation_vector import cv_eval
from correlation import avg_abs_corr_rows


def run_multiple(X, n_runs=20, n_genes=1000, n_samples=10,
                 iterations=500, splits=8, cv_splits=10, random_state_offset=0):
    """
    Run MCbiclust multiple times to build correlation vector matrix.

    For each run:
      1. Select random gene set
      2. FindSeed — greedy search for best 10 samples
      3. HclustGenesHiCor — prune gene set
      4. CVEval — compute correlation vector

    Parameters
    ----------
    X                  : np.ndarray  expression matrix (genes x samples)
    n_runs             : int         number of independent runs
    n_genes            : int         initial gene set size per run
    n_samples          : int         seed size (default 10)
    iterations         : int         FindSeed iterations per run
    splits             : int         number of hierarchical clusters
    random_state_offset: int         offset for random seeds

    Returns
    -------
    cor_vec_mat : np.ndarray  genes x runs correlation vector matrix
    seeds_list  : list        seed sample indices per run
    """
    seeds_list  = []
    corvec_list = []

    print(f"Running {n_runs} MCbiclust iterations...\n")

    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")
        rs = i + random_state_offset

        # random gene set
        rng_i      = np.random.default_rng(rs)
        gene_set_i = np.sort(rng_i.choice(X.shape[0], n_genes, replace=False))

        # FindSeed
        J_i, alpha_i, _, _, _ = find_seed_bicluster(
            X=X,
            gene_set=gene_set_i,
            n_samples=n_samples,
            iterations=iterations,
            random_state=rs,
            initial_seed=None,
            log_improvements=False,
            print_every_improvement=False,
            print_every_iter=False,
            verbose=False,
        )

        # HclustGenesHiCor
        _, pruned_i, _, _, _ = prune_bicluster_genes(
            X,
            gene_set   = gene_set_i,
            sample_set = J_i,
            n_groups   = splits,
            plot_dendrogram = False,
        )

        # CVEval
        corvec_i, _ = cv_eval(
            X_part = X[pruned_i],
            X_all  = X,
            seed   = J_i,
            splits = cv_splits
        )

        seeds_list.append(J_i)
        corvec_list.append(corvec_i)

        print(f"  Alpha: {alpha_i:.4f} | Pruned genes: {len(pruned_i)}\n")

    cor_vec_mat = np.column_stack(corvec_list)
    print(f"Correlation vector matrix: {cor_vec_mat.shape}")
    print(f"({cor_vec_mat.shape[0]} genes x {cor_vec_mat.shape[1]} runs)")

    return cor_vec_mat, seeds_list


def multi_sample_sort_prep(X, av_corvec, top_genes_num,
                            groups, initial_seeds):
    """
    Prepare gene matrices and seeds for SampleSort on each distinct bicluster.

    Equivalent to R's MultiSampleSortPrep.

    For each cluster:
      - Select top top_genes_num genes by |average correlation vector|
      - Find best seed from runs in that cluster (highest alpha on top genes)

    Parameters
    ----------
    X             : np.ndarray  expression matrix (genes x samples)
    av_corvec     : list        average correlation vector per cluster
    top_genes_num : int         number of top genes to use
    groups        : list        boolean arrays from SilhouetteClustGroups
    initial_seeds : list        seed arrays from all runs

    Returns
    -------
    top_mats  : list  gene expression submatrices per cluster
    top_seeds : list  best seed per cluster
    """
    n_clusters = len(av_corvec)
    top_genes  = []
    top_seeds  = []
    top_mats   = []

    for c in range(n_clusters):
        # top genes by |average corvec|
        idx     = np.argsort(np.abs(av_corvec[c]))[::-1][:top_genes_num]
        top_genes.append(idx)

        # best seed = run with highest alpha on top genes
        cluster_runs = np.where(groups[c])[0]
        scores = []
        for run_idx in cluster_runs:
            seed  = initial_seeds[run_idx]
            gem   = X[idx][:, seed]
            a     = avg_abs_corr_rows(gem)
            scores.append(a)

        best_run  = cluster_runs[int(np.argmax(scores))]
        best_seed = initial_seeds[best_run]
        top_seeds.append(best_seed)

        print(f"Cluster {c+1}:")
        print(f"  Runs in cluster: {cluster_runs.tolist()}")
        print(f"  Best run:        {best_run} "
              f"| alpha={max(scores):.4f}")
        print(f"  Best seed:       {best_seed.tolist()}")

        top_mats.append(X[idx])

    print(f"\nReturning {n_clusters} (gem, seed) pairs:")
    for c in range(n_clusters):
        print(f"  Cluster {c+1}: gem={top_mats[c].shape} | "
              f"seed={top_seeds[c].tolist()}")

    return top_mats, top_seeds


def average_corvec_per_cluster(cor_vec_mat, cluster_groups,
                                use_abs=True):
    """
    Compute average correlation vector per cluster.

    Parameters
    ----------
    cor_vec_mat    : np.ndarray  genes x runs matrix
    cluster_groups : list        boolean arrays from SilhouetteClustGroups
    use_abs        : bool        if True average |corvec| to avoid
                                 cancellation of opposite-arm runs

    Returns
    -------
    av_corvec : list of np.ndarray  average corvec per cluster
    """
    av_corvec = []
    av_corvec_signed = []

    for k, grp in enumerate(cluster_groups):
        runs_k = np.where(grp)[0]

        av_cv = np.abs(cor_vec_mat[:, runs_k]).mean(axis=1)
        av_cv_signed = cor_vec_mat[:, runs_k].mean(axis=1)
        av_corvec.append(av_cv)
        av_corvec_signed.append(av_cv_signed)
        print(f"Cluster {k+1}: {len(runs_k)} runs | "
              f"mean|corr|={av_cv.mean():.4f}")

    return av_corvec, av_corvec_signed