"""
test_synthetic.py
=================
Validates the MCbiclust Python package on synthetic data matching the
Bentham et al. 2017 (MCbiclust paper) benchmark setup.

Contains:
  - jaccard / f1_score / hungarian_match  — metric utilities
  - plot_confusion_matrix               — gene + sample confusion heatmaps
  - smoke_test_alpha                    — fast sanity check
  - run_pipeline                        — single-run pipeline (quick test)
  - run_multi_pipeline                  — full multi-run pipeline (paper eval)
  - score_multi_results                 — Jaccard + F1 scoring vs ground truth

Typical usage from a notebook
------------------------------
    import sys
    sys.path.insert(0, "path/to/src")

    from generate_synthetic import generate_paper_dataset
    from test_synthetic import (smoke_test_alpha,
                                run_multi_pipeline,
                                score_multi_results,
                                plot_confusion_matrix)

    dataset, true_genes, true_samples = generate_paper_dataset()
    smoke_test_alpha(dataset, true_genes, true_samples)

    results, cluster_groups, cor_vec_mat = run_multi_pipeline(
        dataset, true_genes, true_samples, n_runs=50
    )
    metrics = score_multi_results(results, true_genes, true_samples)
    plot_confusion_matrix(results, true_genes, true_samples)
"""

import sys
import os
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loading       import ExpressionDataset
from correlation        import avg_abs_corr_rows
from find_seed_sample   import find_seed_bicluster
from pruning            import prune_bicluster_genes
from sample_sort        import extend_bicluster_samples_fast
from correlation_vector import cv_eval
from pca                import pc1_vec_fun
from thresholding       import threshold_bic
from fork               import pc1_align, fork_classifier
from multi_run          import run_multiple, average_corvec_per_cluster
from silhouette         import silhouette_clust_groups


# ─────────────────────────────────────────────────────────────────────────────
# Metric utilities
# ─────────────────────────────────────────────────────────────────────────────

def jaccard(a, b) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    inter  = len(sa & sb)
    union  = len(sa | sb)
    return inter / union if union else 0.0


def f1_score(found, truth) -> float:
    sf, st = set(found.tolist()), set(truth.tolist())
    tp     = len(sf & st)
    prec   = tp / len(sf) if sf else 0.0
    rec    = tp / len(st) if st else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def hungarian_match(sim_matrix):
    """
    Optimal bipartite matching using the Hungarian (Munkres) algorithm.
    Matches the paper's use of the Munkres algorithm for bicluster assignment.

    sim_matrix : (n_planted x n_found)

    Returns
    -------
    pairs   : list of (planted_idx, found_idx, score)
              found_idx = -1 if planted bicluster was unmatched
    spurious: list of found_idx not matched to any planted bicluster
    """
    from scipy.optimize import linear_sum_assignment

    nT, nF = sim_matrix.shape

    # linear_sum_assignment minimises — negate for maximisation
    cost = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    pairs   = []
    matched_found = set()

    for ti, fi in zip(row_ind.tolist(), col_ind.tolist()):
        pairs.append((int(ti), int(fi), float(sim_matrix[ti, fi])))
        matched_found.add(int(fi))

    # planted biclusters not assigned (only possible if nT > nF)
    matched_planted = {ti for ti, fi, _ in pairs}
    for ti in range(nT):
        if ti not in matched_planted:
            pairs.append((ti, -1, 0.0))

    spurious = [j for j in range(nF) if j not in matched_found]
    return pairs, spurious


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(results, true_genes, true_samples,
                          title_prefix="", save_path=None):
    """
    Plot gene-Jaccard and sample-Jaccard confusion matrices
    (planted biclusters x found biclusters).

    Parameters
    ----------
    results      : list[dict]   output of run_multi_pipeline
    true_genes   : list         planted gene index arrays
    true_samples : list         planted sample index arrays
    title_prefix : str          optional prefix for plot titles
    save_path    : str or None  optional path to save figure
    """
    nT = len(true_genes)
    nF = len(results)

    jmat_g = np.zeros((nT, nF))
    jmat_s = np.zeros((nT, nF))

    for i, (tg, ts) in enumerate(zip(true_genes, true_samples)):
        for j, res in enumerate(results):
            jmat_g[i, j] = jaccard(res["found_genes"], tg)
            jmat_s[i, j] = jaccard(res["found_samps"], ts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mat, label in [
        (axes[0], jmat_g, "Gene Jaccard"),
        (axes[1], jmat_s, "Sample Jaccard"),
    ]:
        im = ax.imshow(mat, vmin=0, vmax=1,
                       cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label=label)

        # Annotate cells
        for i in range(nT):
            for j in range(nF):
                ax.text(j, i, f"{mat[i,j]:.2f}",
                        ha="center", va="center",
                        fontsize=8,
                        color="black" if mat[i,j] < 0.6 else "white")

        ax.set_xticks(range(nF))
        ax.set_xticklabels([f"F{j+1}" for j in range(nF)], fontsize=9)
        ax.set_yticks(range(nT))
        ax.set_yticklabels([f"T{i+1}" for i in range(nT)], fontsize=9)
        ax.set_xlabel("Found biclusters", fontsize=10)
        ax.set_ylabel("Planted biclusters", fontsize=10)
        ax.set_title(
            f"{title_prefix}{label} matrix\n"
            f"(planted rows x found cols)",
            fontsize=11
        )

        # Highlight diagonal-equivalent matches
        pairs, _ = hungarian_match(jmat_g)
        for ti, fi, _ in pairs:
            if fi >= 0:
                rect = plt.Rectangle(
                    (fi - 0.5, ti - 0.5), 1, 1,
                    fill=False, edgecolor="blue",
                    linewidth=2.5
                )
                ax.add_patch(rect)

    fig.suptitle(
        f"{title_prefix}Bicluster Recovery — Confusion Matrices\n"
        f"{nT} planted  |  {nF} found",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_recovery_summary(results, true_genes, true_samples,
                          dataset, save_path=None):
    """
    4-panel recovery summary:
      Panel 1: Gene Jaccard bar chart per matched pair
      Panel 2: Sample Jaccard bar chart per matched pair
      Panel 3: Gene F1 bar chart per matched pair
      Panel 4: Sample F1 bar chart per matched pair

    Parameters
    ----------
    results      : list[dict]
    true_genes   : list
    true_samples : list
    dataset      : ExpressionDataset
    save_path    : str or None
    """
    nT = len(true_genes)
    nF = len(results)

    jmat_g = np.zeros((nT, nF))
    jmat_s = np.zeros((nT, nF))
    for i, (tg, ts) in enumerate(zip(true_genes, true_samples)):
        for j, res in enumerate(results):
            jmat_g[i, j] = jaccard(res["found_genes"], tg)
            jmat_s[i, j] = jaccard(res["found_samps"], ts)

    pairs, spurious = greedy_match(jmat_g)

    labels, gj_vals, sj_vals, gf_vals, sf_vals = [], [], [], [], []
    for ti, fi, _ in pairs:
        label = f"T{ti+1}→F{fi+1}" if fi >= 0 else f"T{ti+1}→—"
        labels.append(label)
        if fi >= 0:
            tg, ts = true_genes[ti], true_samples[ti]
            fg     = results[fi]["found_genes"]
            fs     = results[fi]["found_samps"]
            gj_vals.append(jmat_g[ti, fi])
            sj_vals.append(jmat_s[ti, fi])
            gf_vals.append(f1_score(fg, tg))
            sf_vals.append(f1_score(fs, ts))
        else:
            gj_vals.append(0.0)
            sj_vals.append(0.0)
            gf_vals.append(0.0)
            sf_vals.append(0.0)

    x    = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    configs = [
        (axes[0, 0], gj_vals, "Gene Jaccard",   "steelblue"),
        (axes[0, 1], sj_vals, "Sample Jaccard",  "coral"),
        (axes[1, 0], gf_vals, "Gene F1",         "steelblue"),
        (axes[1, 1], sf_vals, "Sample F1",        "coral"),
    ]
    for ax, vals, title, colour in configs:
        bars = ax.bar(x, vals, color=colour, alpha=0.8, width=0.6)
        ax.axhline(0.2, color="red", linestyle="--",
                   linewidth=1, label="threshold 0.2")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    if spurious:
        fig.text(0.5, 0.01,
                 f"Spurious (unmatched found) clusters: "
                 f"{[f'F{j+1}' for j in spurious]}",
                 ha="center", fontsize=9, color="red")

    fig.suptitle("MCbiclust Recovery Summary", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test_alpha(dataset, true_genes, true_samples, min_ratio=2.0):
    """
    Alpha on each planted bicluster block must be >> alpha on a random
    block of equal size.  Raises AssertionError if any bicluster fails.
    """
    print("\n" + "-" * 60)
    print("Smoke test: alpha signal vs noise")
    X          = dataset.X.astype(float)
    rng        = np.random.default_rng(999)
    all_passed = True

    for i, (bg, bs) in enumerate(zip(true_genes, true_samples)):
        a_sig   = avg_abs_corr_rows(X[np.ix_(bg, bs)])
        rg      = rng.choice(dataset.n_genes,   len(bg), replace=False)
        rs      = rng.choice(dataset.n_samples, len(bs), replace=False)
        a_noise = avg_abs_corr_rows(X[np.ix_(rg, rs)])
        ratio   = a_sig / (a_noise + 1e-12)
        ok      = ratio >= min_ratio
        all_passed = all_passed and ok
        print(f"  BC{i+1}: alpha_signal={a_sig:.4f}  "
              f"alpha_noise={a_noise:.4f}  "
              f"ratio={ratio:.1f}x  {'PASS' if ok else 'FAIL'}")

    assert all_passed, "Smoke test failed — increase signal_strength"
    print("Smoke test passed\n")


# ─────────────────────────────────────────────────────────────────────────────
# Single-run pipeline  (quick sanity check, not the paper evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    dataset,
    true_genes,
    true_samples,
    target_bc=1,
    n_seed_genes=1000,
    n_seed_samples=10,
    iterations=500,
    splits=8,
    samp_sig=0.0,
    random_state=42,
):
    """
    Run the full single-run MCbiclust pipeline targeting one planted
    bicluster.  Uses the planted samples as the initial seed so FindSeed
    refines rather than searches blind.

    Returns a dict with found_genes, found_samps, cor_vec, pc1, fork.
    """
    print("=" * 60)
    print(f"Single-run pipeline  (target BC{target_bc})")
    print("=" * 60)

    X            = dataset.X
    rng          = np.random.default_rng(random_state)
    planted      = true_genes[target_bc - 1]
    planted_samp = true_samples[target_bc - 1]

    noise_pool = np.setdiff1d(np.arange(dataset.n_genes), planted)
    n_extra    = max(0, n_seed_genes - len(planted))
    extra      = rng.choice(noise_pool, size=n_extra, replace=False)
    gene_set   = np.sort(np.concatenate([planted, extra]))
    print(f"Gene set: {len(gene_set)} ({len(planted)} signal + {n_extra} noise)")

    print("[1/7] FindSeed")
    seed, seed_alpha, _, _ = find_seed_bicluster(
        X, gene_set,
        n_samples               = n_seed_samples,
        iterations              = iterations,
        random_state            = random_state,
        initial_seed            = planted_samp[:n_seed_samples],
        log_improvements        = False,
        print_every_improvement = False,
        print_every_iter        = False,
    )
    print(f"  alpha = {seed_alpha:.4f}")

    print("[2/7] Gene pruning")
    _, pruned, _, _, _ = prune_bicluster_genes(
        X, gene_set=gene_set, sample_set=seed,
        n_groups=splits, plot_dendrogram=False,
    )
    print(f"  {len(gene_set)} -> {len(pruned)} genes")

    print("[3/7] SampleSort")
    ranked, sample_alphas, _ = extend_bicluster_samples_fast(
        X, pruned, seed,
        max_rank=None, progress_every=200, print_only_improvements=False,
    )

    print("[4/7] CVEval")
    cor_vec, _ = cv_eval(
        X_part=X[pruned], X_all=X, seed=seed, splits=splits
    )

    print("[5/7] PC1VecFun")
    pc1 = pc1_vec_fun(
        X=X, gene_set=pruned, seed_sort=ranked, n=n_seed_samples
    )

    print("[6/7] ThresholdBic")
    bic_genes, bic_samps = threshold_bic(
        cor_vec=cor_vec, sort_order=ranked, pc1=pc1,
        samp_sig=samp_sig, sample_names=dataset.samples,
    )

    print("[7/7] PC1Align + ForkClassifier")
    pc1_aligned = pc1_align(
        X=X, pc1=pc1.copy(), cor_vec=cor_vec,
        sort_order=ranked, bic_samps=bic_samps,
    )
    n_bic = len(bic_samps) if bic_samps is not None else 0
    fork  = (fork_classifier(pc1_aligned, samp_num=n_bic)
             if n_bic >= 2
             else np.full(len(ranked), "None", dtype=object))

    print(f"\n  Bicluster: {int(bic_genes.sum())} genes, {n_bic} samples")
    print(f"  Upper: {int(np.sum(fork=='Upper'))}  "
          f"Lower: {int(np.sum(fork=='Lower'))}")

    return {
        "found_genes": np.where(bic_genes)[0],
        "found_samps": bic_samps if bic_samps is not None else np.array([]),
        "seed_alpha":  seed_alpha,
        "cor_vec":     cor_vec,
        "pc1":         pc1_aligned,
        "fork":        fork,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-run pipeline  (paper-faithful evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_pipeline(
    dataset,
    true_genes,
    true_samples,
    n_runs=50,
    n_genes=1000,
    n_samples_seed=10,
    iterations=500,
    splits=8,
    samp_sig=0.0,
    max_clusters=12,
    top_genes_num=1000,
    random_state_offset=0,
):
    """
    Full multi-run MCbiclust evaluation matching the paper protocol:

      1. n_runs independent runs (random gene sets, FindSeed+Prune+CVEval)
      2. Silhouette clustering of correlation vectors -> k distinct biclusters
      3. For each cluster:
            average corvec -> top genes -> best seed
            -> prune -> SampleSort -> CVEval -> PC1 -> ThresholdBic
      4. Return per-cluster results for scoring

    Parameters
    ----------
    dataset             : ExpressionDataset
    true_genes          : list[np.ndarray]   ground-truth gene indices
    true_samples        : list[np.ndarray]   ground-truth sample indices
    n_runs              : int
    n_genes             : int   gene set size per run
    n_samples_seed      : int   seed size for FindSeed
    iterations          : int   FindSeed iterations per run
    splits              : int   hierarchical groups for pruning
    samp_sig            : float ThresholdBic strictness (0=loose, 0.8=paper)
    max_clusters        : int   upper bound for silhouette search
    top_genes_num       : int   genes used in final per-cluster pipeline
    random_state_offset : int   offset for run random seeds

    Returns
    -------
    results        : list[dict]   found_genes, found_samps per cluster
    cluster_groups : list         boolean arrays from silhouette
    cor_vec_mat    : np.ndarray   genes x runs correlation matrix
    """
    X = dataset.X

    # ── Step 1: multiple runs ─────────────────────────────────
    print("=" * 60)
    print(f"Step 1 / 3  —  {n_runs} independent runs")
    print("=" * 60)
    cor_vec_mat, seeds_list = run_multiple(
        X,
        n_runs              = n_runs,
        n_genes             = n_genes,
        n_samples           = n_samples_seed,
        iterations          = iterations,
        splits              = splits,
        random_state_offset = random_state_offset,
    )
    print(f"Correlation matrix: {cor_vec_mat.shape}  "
          f"({cor_vec_mat.shape[0]} genes x {cor_vec_mat.shape[1]} runs)\n")

    # ── Step 2: silhouette clustering ─────────────────────────
    print("=" * 60)
    print("Step 2 / 3  —  Silhouette clustering")
    print("=" * 60)
    cluster_groups = silhouette_clust_groups(
        cor_vec_mat  = cor_vec_mat,
        max_clusters = max_clusters,
        plots        = False,
    )
    n_found   = len(cluster_groups)
    n_planted = len(true_genes)
    print(f"\nDistinct biclusters found: {n_found}  "
          f"(planted: {n_planted})\n")

    av_corvec, _ = average_corvec_per_cluster(cor_vec_mat, cluster_groups)

    # ── Step 3: final pipeline per cluster ───────────────────
    print("=" * 60)
    print("Step 3 / 3  —  Final pipeline per cluster")
    print("=" * 60)

    results = []
    for k, (grp, av_cv) in enumerate(zip(cluster_groups, av_corvec)):
        print(f"\n--- Cluster {k+1} / {n_found} ---")

        # Top genes by |average corvec|
        top_gene_idx = np.argsort(np.abs(av_cv))[::-1][:top_genes_num]

        # Best seed = run with highest alpha on top genes
        cluster_runs = np.where(grp)[0]
        scores       = [
            avg_abs_corr_rows(X[top_gene_idx][:, seeds_list[r]])
            for r in cluster_runs
        ]
        best_seed = seeds_list[cluster_runs[int(np.argmax(scores))]]
        print(f"  Runs in cluster: {len(cluster_runs)}  "
              f"Best seed alpha: {max(scores):.4f}")

        # Prune — required step, matches the paper pipeline
        _, pruned_k, _, _, _ = prune_bicluster_genes(
            X,
            gene_set        = top_gene_idx,
            sample_set      = best_seed,
            n_groups        = splits,
            plot_dendrogram = False,
        )
        print(f"  Pruned: {len(top_gene_idx)} -> {len(pruned_k)} genes")

        # SampleSort on pruned genes
        ranked, _, _ = extend_bicluster_samples_fast(
            X, pruned_k, best_seed,
            max_rank=None, progress_every=200,
            print_only_improvements=False,
        )

        # CVEval — use top_gene_idx submatrix as X_part (matches multi pipeline)
        cor_vec, _ = cv_eval(
            X_part = X[top_gene_idx],
            X_all  = X,
            seed   = best_seed,
            splits = splits,
        )

        # PC1
        pc1 = pc1_vec_fun(
            X         = X,
            gene_set  = top_gene_idx,
            seed_sort = ranked,
            n         = n_samples_seed,
        )

        # ThresholdBic
        bic_genes, bic_samps = threshold_bic(
            cor_vec      = cor_vec,
            sort_order   = ranked,
            pc1          = pc1,
            samp_sig     = samp_sig,
            sample_names = dataset.samples,
        )

        # PC1Align
        pc1_aligned = pc1_align(
            X          = X,
            pc1        = pc1.copy(),
            cor_vec    = cor_vec,
            sort_order = ranked,
            bic_samps  = bic_samps,
        )

        n_bic = len(bic_samps) if bic_samps is not None else 0
        print(f"  Bicluster: {int(bic_genes.sum())} genes, {n_bic} samples")

        results.append({
            "found_genes": np.where(bic_genes)[0],
            "found_samps": bic_samps if bic_samps is not None else np.array([]),
            "cor_vec":     cor_vec,
            "pc1":         pc1_aligned,
        })

    return results, cluster_groups, cor_vec_mat


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_multi_results(
    results,
    true_genes,
    true_samples,
    gene_j_thresh=0.20,
    samp_j_thresh=0.25,
):
    """
    Score multi-run results against planted biclusters.

    Builds a gene-Jaccard similarity matrix (planted x found), greedily
    matches each planted bicluster to its best found counterpart, then
    computes per-pair Jaccard + F1 and aggregate metrics including the
    consensus score from Hochreiter et al. (as used in Table 1).

    Spurious (unmatched found) clusters are reported separately.

    Parameters
    ----------
    results        : list[dict]   output of run_multi_pipeline
    true_genes     : list         planted gene index arrays
    true_samples   : list         planted sample index arrays
    gene_j_thresh  : float        minimum gene Jaccard to count as PASS
    samp_j_thresh  : float        minimum sample Jaccard to count as PASS

    Returns
    -------
    dict with avg_gene_j, avg_samp_j, avg_gene_f1, avg_samp_f1,
             consensus, matched, n_planted, n_found, spurious
    """
    nT, nF = len(true_genes), len(results)

    # Build Jaccard matrices
    jmat_g = np.zeros((nT, nF))
    jmat_s = np.zeros((nT, nF))
    for i, (tg, ts) in enumerate(zip(true_genes, true_samples)):
        for j, res in enumerate(results):
            jmat_g[i, j] = jaccard(res["found_genes"], tg)
            jmat_s[i, j] = jaccard(res["found_samps"], ts)

    pairs, spurious = hungarian_match(jmat_g)

    # Print Jaccard matrix
    print("\n" + "=" * 60)
    print("Gene Jaccard matrix  (planted rows x found cols)")
    print("=" * 60)
    header = "        " + "".join(f"  F{j+1:<3}" for j in range(nF))
    print(header)
    for i in range(nT):
        row = (f"  T{i+1:<4}  " +
               "".join(f"  {jmat_g[i,j]:.2f}" for j in range(nF)))
        print(row)

    print("\n" + "=" * 60)
    print("Sample Jaccard matrix  (planted rows x found cols)")
    print("=" * 60)
    print(header)
    for i in range(nT):
        row = (f"  T{i+1:<4}  " +
               "".join(f"  {jmat_s[i,j]:.2f}" for j in range(nF)))
        print(row)

    # Per-pair breakdown
    print(f"\n{'Per-bicluster breakdown':}")
    print(f"  {'Planted':<10}{'Found':<8}"
          f"{'Gene J':<10}{'Samp J':<10}"
          f"{'Gene F1':<10}{'Samp F1':<10}{'Status'}")
    print("  " + "-" * 62)

    gj_l, sj_l, gf_l, sf_l, matched = [], [], [], [], 0
    for ti, fi, _ in pairs:
        if fi < 0:
            print(f"  T{ti+1:<9}{'—':<8}" + "—         " * 4 + "UNMATCHED")
            continue
        tg, ts = true_genes[ti], true_samples[ti]
        fg     = results[fi]["found_genes"]
        fs     = results[fi]["found_samps"]
        gj     = jmat_g[ti, fi]
        sj     = jmat_s[ti, fi]
        gf1    = f1_score(fg, tg)
        sf1    = f1_score(fs, ts)
        ok     = gj >= gene_j_thresh and sj >= samp_j_thresh
        if ok:
            matched += 1
        gj_l.append(gj); sj_l.append(sj)
        gf_l.append(gf1); sf_l.append(sf1)
        print(f"  T{ti+1:<9}F{fi+1:<7}"
              f"{gj:<10.3f}{sj:<10.3f}"
              f"{gf1:<10.3f}{sf1:<10.3f}"
              f"{'PASS' if ok else 'FAIL'}")

    # Spurious clusters
    if spurious:
        print(f"\n  Spurious (unmatched) found clusters: "
              f"{[f'F{j+1}' for j in spurious]}")

    avg_gj  = float(np.mean(gj_l))  if gj_l  else 0.0
    avg_sj  = float(np.mean(sj_l))  if sj_l  else 0.0
    avg_gf1 = float(np.mean(gf_l))  if gf_l  else 0.0
    avg_sf1 = float(np.mean(sf_l))  if sf_l  else 0.0

    # Consensus score: penalises wrong number of clusters
    consensus = avg_gj * avg_sj * nT / max(nT, nF)

    print(f"\n{'=' * 60}")
    print("AGGREGATE METRICS")
    print("=" * 60)
    print(f"  Planted:            {nT}")
    print(f"  Found:              {nF}")
    print(f"  Spurious:           {len(spurious)}")
    print(f"  Matched (PASS):     {matched} / {nT}")
    print(f"  Avg gene  Jaccard:  {avg_gj:.3f}")
    print(f"  Avg sample Jaccard: {avg_sj:.3f}")
    print(f"  Avg gene  F1:       {avg_gf1:.3f}")
    print(f"  Avg sample F1:      {avg_sf1:.3f}")
    print(f"  Consensus score:    {consensus:.4f}")
    print()
    print("  Paper reference (Table 1) — Bentham et al. 2017:")
    print("    MCbiclust optimum:   consensus=0.4368  gene_F1=0.8145  samp_F1=0.6634")
    print("    MCbiclust threshold: consensus=0.3462  gene_F1=0.8043  samp_F1=0.5864")
    print("    (Your pipeline uses threshold mode — compare against threshold row)")
    print("    Note: paper found 6/8 biclusters — perfect recovery is not expected")

    # Verdict uses threshold-mode paper values as reference
    verdict = (avg_gf1 >= 0.50 and avg_sf1 >= 0.35 and matched >= nT // 2)
    print(f"\n  VERDICT: {'PASS' if verdict else 'FAIL'}")

    return {
        "avg_gene_j":  avg_gj,
        "avg_samp_j":  avg_sj,
        "avg_gene_f1": avg_gf1,
        "avg_samp_f1": avg_sf1,
        "consensus":   consensus,
        "matched":     matched,
        "n_planted":   nT,
        "n_found":     nF,
        "spurious":    spurious,
    }