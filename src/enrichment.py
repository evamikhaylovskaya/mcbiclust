import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import urllib.request
import json



def load_gaf(gaf_path):
    """
    Load human GAF annotation file.
    Keep all GO aspects: P, C, F.
    Exclude NOT annotations.
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
        sep="\t",
        comment="!",
        header=None,
        names=gaf_cols,
        low_memory=False
    )

    gaf["qualifier"] = gaf["qualifier"].fillna("").astype(str)
    gaf = gaf[~gaf["qualifier"].str.contains("NOT", na=False)]

    print(f"GAF shape:       {gaf.shape}")
    print(f"Unique GO terms: {gaf['go_id'].nunique()}")
    print(f"Unique genes:    {gaf['db_object_symbol'].nunique()}")

    gaf["gene_lower"] = gaf["db_object_symbol"].astype(str).str.lower()
    return gaf

def build_go_probe_mapping(gaf_df, gene_names, min_genes=5):
    """
    Build GO term mapping.

    Returns
    -------
    go_to_term_genes : dict
        GO term -> all annotated genes in that term
    go_to_probe_idx : dict
        GO term -> indices of genes from expression matrix present in that term
    go_to_aspect : dict
        GO term -> ontology aspect (P/C/F)
    """
    probe_series = pd.Series(gene_names).astype(str)
    probe_gene_names = probe_series.str.lower()

    go_to_term_genes = {}
    go_to_probe_idx = {}
    go_to_aspect = {}

    for go_id, group in gaf_df.groupby("go_id"):
        term_genes = sorted(set(group["gene_lower"]))
        probe_idx = np.where(probe_gene_names.isin(term_genes))[0]

        if len(probe_idx) >= min_genes:
            go_to_term_genes[go_id] = term_genes
            go_to_probe_idx[go_id] = probe_idx
            go_to_aspect[go_id] = group["aspect"].iloc[0]

    print(f"GO terms with >= {min_genes} probes: {len(go_to_probe_idx)}")

    return go_to_term_genes, go_to_probe_idx, go_to_aspect


def run_go_enrichment(cor_vec, gene_names, go_to_term_genes, go_to_probe_idx, alternative="greater"):
    """
    Closer equivalent of R GOEnrichmentAnalysis.
    Uses signed correlation vector.
    """
    cor_vec = np.asarray(cor_vec, dtype=float)
    results = []
    global_mean = float(cor_vec.mean())

    for go_id, probe_idx in go_to_probe_idx.items():
        in_set = cor_vec[probe_idx]

        mask = np.ones(len(cor_vec), dtype=bool)
        mask[probe_idx] = False
        out_set = cor_vec[mask]

        _, pval = stats.mannwhitneyu(
            in_set,
            out_set,
            alternative=alternative,
            method="auto"
        )

        term_genes = go_to_term_genes[go_id]
        num_genes = len(term_genes)
        g_in_genelist = len(probe_idx)
        cv_av_value = float(in_set.mean())
        phenotype = 1 if cv_av_value > global_mean else -1

        results.append({
            "go_id": go_id,
            "num.genes": num_genes,
            "g.in.genelist": g_in_genelist,
            "pval": pval,
            "CV.av.value": cv_av_value,
            "phenotype": phenotype,
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise ValueError("No GO terms produced any results.")

    _, adj_p, _, _ = multipletests(results_df["pval"], method="fdr_bh")
    results_df["adj.p.value"] = adj_p

    results_df = results_df.sort_values("adj.p.value").reset_index(drop=True)

    print(f"Total GO terms tested: {len(results_df)}")
    return results_df


def fetch_go_term_names(go_ids, timeout=10):
    go_term_names = {}

    for i, go_id in enumerate(go_ids):
        try:
            url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
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
    gaf_df = load_gaf(gaf_path)

    go_to_term_genes, go_to_probe_idx, go_to_aspect = build_go_probe_mapping(
        gaf_df, gene_names, min_genes=min_genes
    )

    results_df = run_go_enrichment(
        cor_vec=cor_vec,
        gene_names=gene_names,
        go_to_term_genes=go_to_term_genes,
        go_to_probe_idx=go_to_probe_idx,
        alternative=alternative
    )

    results_df["ONTOLOGY"] = results_df["go_id"].map(go_to_aspect)

    top_go_ids = results_df["go_id"].head(top_n).tolist()
    go_term_names = fetch_go_term_names(top_go_ids)
    results_df["TERM"] = results_df["go_id"].map(go_term_names)

    print(f"\nTop {top_n} GO terms:")
    print(results_df[[
        "go_id", "TERM", "ONTOLOGY", "num.genes", "g.in.genelist",
        "adj.p.value", "CV.av.value", "phenotype"
    ]].head(top_n).to_string(index=False))

    if save_path is not None:
        results_df.to_csv(save_path, index=False)
        print(f"\nSaved to {save_path}")

    return results_df