"""
generate_synthetic.py
=====================
Generates the synthetic dataset from Bentham et al. 2017 (MCbiclust paper).

From the Methods:
  "Eight separate synthetic datasets, using the FABIA model. Each dataset
   contained only one bicluster, on average containing approximately 500
   genes and 130 samples, and each dataset was mean centered according to
   the genes before being combined. Eight biclusters were chosen so that
   the final combined synthetic dataset contained 1000 genes and 1059
   samples. Enforcing sample exclusiveness to a single bicluster."

The FABIA multiplicative model for one bicluster:
    X = lambda @ z^T + noise
where:
    lambda : genes x 1  — gene loading vector (sparse)
    z      : samples x 1 — sample factor vector (sparse)
    noise  : genes x samples ~ N(0, sigma^2)

Usage
-----
    PYTHONPATH=src python generate_synthetic.py
"""

import numpy as np
import pandas as pd
from data_loading import ExpressionDataset


def generate_fabia_bicluster(
    n_genes: int,
    n_samples: int,
    n_bic_genes: int,
    n_bic_samples: int,
    signal_strength: float = 4.0,
    noise_std: float = 1.0,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Generate one FABIA-style multiplicative bicluster.

    Model: X = lambda @ z^T + noise
      - lambda: sparse gene loading — N(0, signal_strength^2) for bicluster
                genes, 0 elsewhere
      - z:      sparse sample factor — N(0, 1) for bicluster samples,
                0 elsewhere
      - noise:  N(0, noise_std^2) everywhere

    Then mean-center each gene across all samples (as in the paper).

    Returns
    -------
    X         : np.ndarray  genes x samples, mean-centered by gene
    bic_genes : np.ndarray  indices of bicluster genes (local, 0-based)
    bic_samps : np.ndarray  indices of bicluster samples (local, 0-based)
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_bic_genes > n_genes:
        raise ValueError(
            f"n_bic_genes ({n_bic_genes}) cannot exceed n_genes ({n_genes}). "
            f"Set n_bic_genes <= n_genes_per_bic."
        )
    if n_bic_samples > n_samples:
        raise ValueError(
            f"n_bic_samples ({n_bic_samples}) cannot exceed n_samples ({n_samples})."
        )

    # Sparse gene loadings
    bic_genes      = np.sort(rng.choice(n_genes, size=n_bic_genes, replace=False))
    lam            = np.zeros(n_genes)
    lam[bic_genes] = rng.standard_normal(n_bic_genes) * signal_strength

    # Sparse sample factors
    bic_samps      = np.sort(rng.choice(n_samples, size=n_bic_samples, replace=False))
    z              = np.zeros(n_samples)
    z[bic_samps]   = rng.standard_normal(n_bic_samples)

    # Multiplicative signal + noise
    X  = np.outer(lam, z)
    X += rng.standard_normal((n_genes, n_samples)) * noise_std

    # Mean-center each gene across all samples (paper step)
    X -= X.mean(axis=1, keepdims=True)

    return X.astype(np.float32), bic_genes, bic_samps


def generate_paper_dataset(
    n_biclusters: int      = 8,
    n_genes_per_bic: int   = 125,    # 8 x 125 = 1000 total genes
    n_samples_per_bic: int = 132,    # 8 x 132 = 1056 + 3 padding = 1059
    n_bic_genes: int       = 100,    # active genes per bicluster
                                     # must be <= n_genes_per_bic
    n_bic_samples: int     = 130,    # active samples per bicluster
                                     # must be <= n_samples_per_bic
    signal_strength: float = 4.0,
    noise_std: float       = 1.0,
    random_state: int      = 0,
    save_path: str         = None,
) -> tuple:
    """
    Build the combined 1000-gene x 1059-sample synthetic matrix.

    Structure
    ---------
    Genes are partitioned: bicluster k owns rows [k*125 : (k+1)*125].
    Samples are partitioned: bicluster k owns cols [k*132 : (k+1)*132].
    Each bicluster's active genes are drawn from its own partition only,
    enforcing strict gene and sample non-overlap between biclusters.

    n_bic_genes must be <= n_genes_per_bic (hard constraint, no silent
    capping). For a clean recoverable signal use n_bic_genes ~ 80-100
    (roughly 64-80% of each partition).

    Returns
    -------
    dataset      : ExpressionDataset
    true_genes   : list[np.ndarray]  global gene indices per bicluster
    true_samples : list[np.ndarray]  global sample indices per bicluster
    """
    # Hard validation — no silent capping
    if n_bic_genes > n_genes_per_bic:
        raise ValueError(
            f"n_bic_genes ({n_bic_genes}) must be <= n_genes_per_bic "
            f"({n_genes_per_bic}). The paper's stated ~500 active genes "
            f"refers to the full pooled matrix, not each partition."
        )
    if n_bic_samples > n_samples_per_bic:
        raise ValueError(
            f"n_bic_samples ({n_bic_samples}) must be <= n_samples_per_bic "
            f"({n_samples_per_bic})."
        )

    rng = np.random.default_rng(random_state)

    # Total dimensions
    n_genes_total   = n_biclusters * n_genes_per_bic    # 1000
    n_samples_total = n_biclusters * n_samples_per_bic  # 1056

    # Pad samples to match paper's 1059
    n_extra_samples  = max(0, 1059 - n_samples_total)
    n_samples_padded = n_samples_total + n_extra_samples

    X_combined = np.zeros(
        (n_genes_total, n_samples_padded), dtype=np.float32
    )

    true_genes_global   = []
    true_samples_global = []

    print(f"Generating {n_biclusters} FABIA biclusters:")
    print(f"  Total matrix: {n_genes_total} genes x {n_samples_padded} samples")
    print(f"  Active genes per bicluster:   {n_bic_genes} / {n_genes_per_bic}")
    print(f"  Active samples per bicluster: {n_bic_samples} / {n_samples_per_bic}")
    print()

    for b in range(n_biclusters):
        g_start = b * n_genes_per_bic
        g_end   = g_start + n_genes_per_bic
        s_start = b * n_samples_per_bic
        s_end   = s_start + n_samples_per_bic

        X_local, local_bg, local_bs = generate_fabia_bicluster(
            n_genes         = n_genes_per_bic,
            n_samples       = n_samples_per_bic,
            n_bic_genes     = n_bic_genes,
            n_bic_samples   = n_bic_samples,
            signal_strength = signal_strength,
            noise_std       = noise_std,
            rng             = rng,
        )

        X_combined[g_start:g_end, s_start:s_end] = X_local

        true_genes_global.append(local_bg + g_start)
        true_samples_global.append(local_bs + s_start)

        print(f"  Bicluster {b+1}: "
              f"genes [{g_start}:{g_end}] ({n_bic_genes} active)  "
              f"samples [{s_start}:{s_end}] ({n_bic_samples} active)")

    # Padding columns: pure noise, mean-centered per gene
    if n_extra_samples > 0:
        noise_cols  = rng.standard_normal(
            (n_genes_total, n_extra_samples)
        ).astype(np.float32) * noise_std
        noise_cols -= noise_cols.mean(axis=1, keepdims=True)
        X_combined[:, n_samples_total:] = noise_cols
        print(f"\n  + {n_extra_samples} noise-only padding samples "
              f"(columns {n_samples_total}:{n_samples_padded})")

    # Final global mean-centering across the full combined matrix
    X_combined -= X_combined.mean(axis=1, keepdims=True)

    genes   = np.array([f"gene_{i}"   for i in range(n_genes_total)])
    samples = np.array([f"sample_{j}" for j in range(n_samples_padded)])

    dataset = ExpressionDataset(
        X_combined, genes, samples, name="FABIA_synthetic_8bic"
    )

    print()
    dataset.summary()

    if save_path is not None:
        df = pd.DataFrame(X_combined, index=genes, columns=samples)
        df.to_csv(save_path)
        print(f"\nSaved to {save_path}")

    return dataset, true_genes_global, true_samples_global


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(dataset, true_genes, true_samples):
    """
    Check that:
      1. Alpha on each planted block >> alpha on a random block (signal check)
      2. Gene sets are non-overlapping
      3. Sample sets are non-overlapping
    """
    from correlation import avg_abs_corr_rows

    print("\n" + "-" * 60)
    print("Validation")

    X   = dataset.X.astype(float)
    rng = np.random.default_rng(999)

    # ── Alpha signal vs noise ─────────────────────────────────
    print("\nAlpha signal vs noise:")
    all_ok = True
    for i, (bg, bs) in enumerate(zip(true_genes, true_samples)):
        a_sig   = avg_abs_corr_rows(X[np.ix_(bg, bs)])
        rg      = rng.choice(dataset.n_genes,   len(bg), replace=False)
        rs      = rng.choice(dataset.n_samples, len(bs), replace=False)
        a_noise = avg_abs_corr_rows(X[np.ix_(rg, rs)])
        ratio   = a_sig / (a_noise + 1e-12)
        ok      = ratio > 1.5
        all_ok  = all_ok and ok
        print(f"  BC{i+1}: alpha_signal={a_sig:.4f}  "
              f"alpha_noise={a_noise:.4f}  "
              f"ratio={ratio:.1f}x  {'OK' if ok else 'WEAK'}")

    # ── Gene exclusivity ──────────────────────────────────────
    print("\nGene exclusivity:")
    for i in range(len(true_genes)):
        for j in range(i + 1, len(true_genes)):
            overlap = len(set(true_genes[i].tolist()) &
                          set(true_genes[j].tolist()))
            status  = "OK" if overlap == 0 else "FAIL"
            print(f"  BC{i+1} & BC{j+1}: {overlap} overlapping genes  {status}")

    # ── Sample exclusivity ────────────────────────────────────
    print("\nSample exclusivity:")
    for i in range(len(true_samples)):
        for j in range(i + 1, len(true_samples)):
            overlap = len(set(true_samples[i].tolist()) &
                          set(true_samples[j].tolist()))
            status  = "OK" if overlap == 0 else "FAIL"
            print(f"  BC{i+1} & BC{j+1}: {overlap} overlapping samples  {status}")

    print(f"\nAll alpha checks passed: {all_ok}")
    return all_ok


if __name__ == "__main__":
    dataset, true_genes, true_samples = generate_paper_dataset(
        n_biclusters      = 8,
        n_genes_per_bic   = 125,
        n_samples_per_bic = 132,
        n_bic_genes       = 100,
        n_bic_samples     = 130,
        signal_strength   = 5.0,
        noise_std         = 1.0,
        random_state      = 0,
        save_path         = "synthetic_8bic_combined.csv",
    )

    validate_dataset(dataset, true_genes, true_samples)