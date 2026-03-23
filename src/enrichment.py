import pandas as pd
import numpy as np
from scipy import stats
import urllib.request
import json


def load_gaf(gaf_path):
    """
    Load EcoCyc GAF annotation file.

    Parameters
    ----------
    gaf_path : str  path to .gaf file

    Returns
    -------
    gaf_bp : pd.DataFrame  biological process annotations only
    """
    gaf_cols = [
        "db", "db_object_id", "db_object_symbol", "qualifier", "go_id",
        "db_reference", "evidence_code", "with_from", "aspect",
        "db_object_name", "db_object_synonym", "db_object_type",
        "taxon", "date", "assigned_by", "annotation_extension",
        "gene_product_form_id"
    ]

    gaf = pd.read_csv(
        gaf_path,
        sep        = "\t",
        comment    = "!",
        header     = None,
        names      = gaf_cols,
        low_memory = False
    )

    print(f"GAF shape:       {gaf.shape}")
    print(f"Unique GO terms: {gaf['go_id'].nunique()}")
    print(f"Unique genes:    {gaf['db_object_symbol'].nunique()}")

    gaf["gene_lower"] = gaf["db_object_symbol"].str.lower()
    gaf_bp = gaf[gaf["aspect"] == "P"]

    return gaf_bp


def build_go_probe_mapping(gaf_bp, gene_names, min_genes=5):
    """
    Build GO term to probe indices mapping.

    Parameters
    ----------
    gaf_bp     : pd.DataFrame  biological process GAF annotations
    gene_names : np.ndarray    gene/probe names from expression dataset
    min_genes  : int           minimum number of probes per GO term

    Returns
    -------
    go_to_probes : dict  GO term -> probe indices
    """
    probe_series     = pd.Series(gene_names)
    probe_gene_names = probe_series.str.split("_").str[0].str.lower()

    go_to_probes = {}
    for go_id, group in gaf_bp.groupby("go_id"):
        genes_in_term = set(group["gene_lower"])
        probe_indices = np.where(probe_gene_names.isin(genes_in_term))[0]
        if len(probe_indices) >= min_genes:
            go_to_probes[go_id] = probe_indices

    print(f"GO terms with >= {min_genes} probes: {len(go_to_probes)}")

    return go_to_probes


def run_go_enrichment(cor_vec, go_to_probes, alternative="greater"):
    """
    Run Mann-Whitney GO enrichment on correlation vector.

    Parameters
    ----------
    cor_vec      : np.ndarray  correlation vector (all genes)
    go_to_probes : dict        GO term -> probe indices
    alternative  : str         'greater', 'less', or 'two-sided'

    Returns
    -------
    results_df : pd.DataFrame  results sorted by p-value
    """
    abs_cor_vec = np.abs(cor_vec)
    results     = []

    for go_id, probe_idx in go_to_probes.items():
        in_set = abs_cor_vec[probe_idx]

        mask           = np.ones(len(abs_cor_vec), dtype=bool)
        mask[probe_idx] = False
        out_set        = abs_cor_vec[mask]

        stat, pval = stats.mannwhitneyu(
            in_set, out_set,
            alternative = alternative,
            method      = "asymptotic"
        )
        results.append({
            "go_id"  : go_id,
            "n_genes": len(probe_idx),
            "pval"   : pval
        })

    results_df             = pd.DataFrame(results)
    results_df["pval_adj"] = results_df["pval"] * len(results_df)
    results_df             = results_df.sort_values("pval").reset_index(drop=True)

    print(f"Total GO terms tested: {len(results_df)}")

    return results_df


def fetch_go_term_names(go_ids, timeout=10):
    """
    Fetch GO term names from QuickGO API.

    Parameters
    ----------
    go_ids  : list  list of GO term IDs
    timeout : int   request timeout in seconds

    Returns
    -------
    go_term_names : dict  GO term ID -> name
    """
    go_term_names = {}

    for i, go_id in enumerate(go_ids):
        try:
            url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"
            req = urllib.request.Request(
                url, headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read())
                go_term_names[go_id] = data["results"][0]["name"]
            print(f"[{i+1}/{len(go_ids)}] {go_id} → {go_term_names[go_id]}")
        except Exception as e:
            go_term_names[go_id] = "unknown"
            print(f"[{i+1}/{len(go_ids)}] {go_id} → failed: {e}")

    return go_term_names


def go_enrichment_pipeline(cor_vec, gene_names, gaf_path,
                            top_n=20, save_path=None,
                            min_genes=5, alternative="greater"):
    """
    Full GO enrichment pipeline.

    Parameters
    ----------
    cor_vec    : np.ndarray  correlation vector
    gene_names : np.ndarray  gene names matching cor_vec
    gaf_path   : str         path to GAF file
    top_n      : int         number of top terms to fetch names for
    save_path  : str or None path to save results CSV
    min_genes  : int         minimum genes per GO term
    alternative: str         Mann-Whitney alternative

    Returns
    -------
    results_df : pd.DataFrame
    """
    # load GAF
    gaf_bp = load_gaf(gaf_path)

    # build mapping
    go_to_probes = build_go_probe_mapping(
        gaf_bp, gene_names, min_genes=min_genes
    )

    # run enrichment
    results_df = run_go_enrichment(
        cor_vec, go_to_probes, alternative=alternative
    )

    # fetch names
    top_go_ids            = results_df["go_id"].head(top_n).tolist()
    go_term_names         = fetch_go_term_names(top_go_ids)
    results_df["go_term_name"] = results_df["go_id"].map(go_term_names)

    # print top results
    print(f"\nTop {top_n} GO terms:")
    print(results_df[["go_id", "go_term_name", "n_genes",
                       "pval", "pval_adj"]].head(top_n).to_string(index=False))

    # save
    if save_path is not None:
        results_df.to_csv(save_path, index=False)
        print(f"\nSaved to {save_path}")

    return results_df