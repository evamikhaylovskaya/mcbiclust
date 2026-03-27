# MCbiclust Python Reimplementation for Dissertation

This repository contains a Python reimplementation and extension of the MCbiclust algorithm for biclustering gene expression data.  
The project is developed as part of a university dissertation and includes:

- core MCbiclust pipeline implementation;
- multi-run bicluster discovery workflow;
- synthetic benchmark evaluation utilities;
- application notebooks for real datasets (E. coli and CCLE);
- optional GPU acceleration for the most expensive steps.

## Dissertation Context

The goal of this project is to reproduce and evaluate MCbiclust-style bicluster recovery in Python, then apply it to biological expression datasets for exploratory analysis and interpretation (including GO enrichment and metadata association).

## Method Overview

The single-run MCbiclust pipeline in `src/mcbiclust.py` follows these stages:

1. **FindSeed** - greedy search for a high-correlation seed sample set.
2. **Gene Pruning** - hierarchical filtering of genes into high-correlation groups.
3. **SampleSort** - ranks all samples by alpha progression.
4. **CVEval** - computes a genome-wide correlation vector.
5. **PC1VecFun** - computes principal component trajectory.
6. **ThresholdBic** - determines bicluster genes and samples.
7. **PC1Align** - aligns the PC1 sign convention.
8. **ForkClassifier** - labels upper/lower fork structure.

The multi-run workflow (`MCbiclustMulti`) repeats runs, clusters correlation vectors via silhouette analysis, and reconstructs representative biclusters per discovered cluster.

## Repository Structure

```text
mcbiclust/
├── data/                         # Input datasets and annotation files
├── src/                          # Core Python implementation
│   ├── mcbiclust.py              # MCbiclust and MCbiclustMulti classes
│   ├── test_synthetic.py         # Synthetic benchmarking & scoring
│   ├── generate_synthetic.py     # Synthetic dataset generator
│   ├── data_loading.py           # Dataset loaders and container
│   ├── *_gpu.py                  # Optional GPU-accelerated components
│   └── ...                       # Supporting modules (pruning, thresholding, etc.)
├── mcbiclust.ipynb               # Main analysis notebook
├── ccle.ipynb                    # CCLE-focused analysis notebook
├── mcbiclust-synthetic.ipynb     # Synthetic experiments notebook
└── mcbiclust-synthetic-Copy1.ipynb
```

## Function Documentation (`src/`)

This section describes the main classes and functions by module.

### Core Pipeline

- `mcbiclust.py`
  - `MCbiclust`: single-run end-to-end pipeline (FindSeed -> Pruning -> SampleSort -> CVEval -> PC1 -> Threshold -> Fork).
  - `MCbiclustMulti`: multi-run wrapper that discovers multiple distinct biclusters using silhouette-based grouping.

- `find_seed_sample.py`
  - `find_seed_bicluster(...)`: searches for the best initial sample seed that maximizes average absolute gene correlation.

- `pruning.py`
  - `prune_bicluster_genes(...)`: performs hierarchical gene pruning and returns high-correlation gene groups.

- `sample_sort.py`
  - `extend_bicluster_samples_fast(...)`: incrementally adds and ranks samples by alpha improvement.
  - `plot_sample_extension_summary(...)`: visual summary of sample extension behavior.

- `correlation_vector.py`
  - `gene_vec_fun(...)`: computes correlation contributions across genes.
  - `cv_eval(...)`: computes the correlation vector used for bicluster definition.
  - `print_top_genes(...)`: prints top positively/negatively associated genes.

- `pca.py`
  - `pc1_vec_fun(...)`: computes PC1 trajectory over sample ordering.
  - `plot_fork(...)`: plots the fork structure in PC1 space.
  - `pc1_summary_stats(...)`: prints summary statistics for PC1.

- `thresholding.py`
  - `threshold_bic(...)`: selects bicluster genes and samples from correlation vector and PC1.
  - `pc1_align(...)`: aligns PC1 sign/orientation for consistent interpretation.
  - `plot_threshold_summary(...)`: visual diagnostics for threshold choices.

- `fork.py`
  - `fork_classifier(...)`: labels samples as upper/lower fork.
  - `print_fork_summary(...)`: tabular summary of fork assignment.
  - `plot_fork_classified(...)`: fork-labeled visualization.

### Multi-Run and Clustering Utilities

- `multi_run.py`
  - `run_multiple(...)`: runs multiple independent MCbiclust initializations.
  - `multi_sample_sort_prep(...)`: prepares top genes/seeds for per-cluster reconstruction.
  - `average_corvec_per_cluster(...)`: computes mean correlation vectors per discovered cluster.

- `silhouette.py`
  - `silhouette_clust_groups(...)`: picks cluster structure from correlation vectors via silhouette score.
  - `plot_silhouette(...)`: plots silhouette and clustered correlation heatmaps.
  - `average_corvec_per_cluster(...)`: cluster-level average correlation vectors.

### Data Loading and Preprocessing

- `data_loading.py`
  - `ExpressionDataset`: container for expression matrix, gene names, and sample names.
  - `load_ecoli(...)`: loads E. coli matrix.
  - `load_ccle_raw(...)`: loads raw CCLE `.gct` matrix.
  - `load_expression_matrix(...)`: generic expression matrix loader.
  - `load_ccle_mitocarta(...)`: loads CCLE + MitoCarta and returns matched mitochondrial subset.

- `preprocessing.py`
  - `log2_transform(...)`: log2 transform with pseudocount.
  - `filter_low_expression_genes(...)`: removes low-mean genes.
  - `filter_top_variable_genes(...)`: keeps top-variance genes.
  - `remove_samples(...)`: removes selected samples.
  - `select_genes(...)` / `remove_genes(...)`: subset by gene names.
  - `row_zscore_dataset(...)`: row-wise z-score normalization.

- `gene_seed.py`
  - `select_initial_seed_genes(...)`: chooses initial gene subset for seed search.

### Statistical/Interpretation Modules

- `metadata_analysis.py`
  - `load_ccle_sample_info(...)`: loads sample metadata.
  - `build_sample_results_dataframe(...)`: creates per-sample result table from rankings and fork labels.
  - `merge_sample_metadata(...)`: merges MCbiclust outputs with external metadata.
  - `plot_pc1_by_category(...)`: compares PC1 distribution across metadata categories.
  - `chi_square_fork_vs_category(...)`: tests fork/category association.

- `enrichment.py`
  - `load_gaf(...)`: parses GO annotation files.
  - `build_go_probe_mapping(...)`: builds GO-term to probe index mappings.
  - `run_go_enrichment(...)`: performs enrichment tests on ranked genes.
  - `fetch_go_term_names(...)`: retrieves GO term names.
  - `go_enrichment_pipeline(...)`: complete enrichment pipeline from correlation vector to results table.

- `fork_analysis.py`
  - `split_samples_by_fork(...)`: separates ranked samples by fork label.
  - `compute_fork_expression_difference(...)`: computes expression differences between forks.
  - `get_top_fork_genes(...)`: extracts most fork-differential genes.
  - `run_fork_go_enrichment(...)`: GO enrichment on fork-differential genes.

### Validation, Plotting, and Synthetic Benchmarking

- `correlation.py`
  - `avg_abs_corr_rows(...)`: computes mean absolute pairwise row correlation (alpha metric).

- `test_synthetic.py`
  - `jaccard(...)`, `f1_score(...)`: recovery metrics.
  - `hungarian_match(...)`: optimal planted-vs-found bicluster matching.
  - `smoke_test_alpha(...)`: fast planted-signal sanity check.
  - `run_pipeline(...)`: single-run benchmark pipeline.
  - `run_multi_pipeline(...)`: full paper-style multi-run benchmark.
  - `score_multi_results(...)`: aggregate scoring and benchmark summary.
  - `plot_confusion_matrix(...)`, `plot_recovery_summary(...)`: diagnostic recovery plots.

- `generate_synthetic.py`
  - `generate_fabia_bicluster(...)`: generates one synthetic bicluster block.
  - `generate_paper_dataset(...)`: builds full synthetic benchmark dataset.
  - `validate_dataset(...)`: checks generated dataset integrity.

- `cv_plot.py`
  - plotting functions for correlation vector interpretation and ranked visualization.

- `seed_plots.py`
  - plotting functions for seed optimization and gene/sample correlation heatmaps.

- `point_score.py`
  - `point_score_calc(...)`: computes point-wise comparison score between gene sets.

### Optional GPU Modules

- `correlation_gpu.py`: GPU/CPU fallback implementations of alpha and batch correlation helpers.
- `find_seed_sample_gpu.py`: GPU-accelerated seed search.
- `sample_sort_gpu.py`: GPU-accelerated sample extension.
- `multi_run_gpu.py`: GPU-enabled multi-run wrapper.
- `gpu_setup_and_usage.py`: utilities for checking GPU availability and smoke tests.

## Requirements

Recommended environment:

- Python 3.10+ (3.11 tested most commonly in practice)
- macOS/Linux/Windows
- Jupyter Notebook or JupyterLab (for notebook workflows)

Python packages used by the project:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `seaborn`
- `statsmodels`
- `openpyxl` (needed when reading `.xls/.xlsx` metadata files)

Optional GPU support:

- `cupy` (if CUDA-compatible hardware/drivers are available)

## Installation

From the `mcbiclust` folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn seaborn statsmodels openpyxl jupyter
```

If using GPU modules, install a compatible CuPy build for your CUDA version (see official CuPy installation docs).

## Data

The `data/` directory already contains large expression resources and annotation files, including:

- CCLE RNA-seq matrix (`.gct`)
- E. coli expression matrix (`.tab`)
- MitoCarta files (`.csv`, `.xls`)
- GO annotation files (`.gaf`)
- sample metadata (`sample_info.csv`)

Because these files are large, they are kept as local dissertation assets rather than downloaded automatically by scripts.

## Quick Start (Python API)

Run a basic single bicluster fit:

```python
import sys
sys.path.insert(0, "src")

from data_loading import load_ecoli
from mcbiclust import MCbiclust

dataset = load_ecoli("data/E_coli_v4_Build_6_chips907probes7459.tab")

model = MCbiclust(
    n_samples=10,
    iterations=1000,
    n_genes=1000,
    splits=8,
    samp_sig=0.05,
    random_state=42,
).fit(dataset)

model.summary()
```

Run multi-run discovery:

```python
from mcbiclust import MCbiclustMulti

multi = MCbiclustMulti(
    n_runs=20,
    n_genes=1000,
    n_samples=10,
    iterations=500,
    splits=8,
    samp_sig=0.8,
    max_clusters=10,
).fit(dataset)

multi.summary()
```

## Synthetic Benchmark Workflow

The synthetic benchmark logic is in `src/test_synthetic.py` and supports:

- smoke testing of planted signal (`smoke_test_alpha`);
- full multi-run benchmark (`run_multi_pipeline`);
- matching/scoring with Jaccard and F1 (`score_multi_results`);
- confusion matrix and recovery plots.

Typical flow:

1. generate synthetic data;
2. run multi-run pipeline;
3. score found biclusters against planted biclusters;
4. compare aggregate metrics with paper-level reference values.

## Notebooks

Primary notebooks:

- `mcbiclust.ipynb` - general end-to-end experiments and outputs.
- `ccle.ipynb` - CCLE-focused analysis and biological interpretation.
- `mcbiclust-synthetic.ipynb` - synthetic recovery evaluation.

Notebook outputs in this repository represent active dissertation experiments and may include intermediate artifacts/plots.

## Reproducibility Notes

- Set explicit random seeds (`random_state`) in all experiments.
- Keep the same preprocessing path when comparing runs.
- Use fixed parameter sets for dissertation tables/figures.
- Record dataset versions and annotation file versions alongside results.

## Outputs and Artifacts

Example generated artifacts include:

- `sample_order.csv`, `sample_extension.csv`
- `go_enrichment_results.csv`
- saved NumPy arrays (`*.npy`) and JSON summaries
- diagnostic and result plots in notebook cells

## Limitations

- Runtime can be high for large expression matrices, especially in multi-run mode.
- Some steps rely on threshold heuristics that may require dataset-specific tuning.
- GPU acceleration is optional and environment-dependent.

## Citation

If you use this code or methodology, please cite:

- the original MCbiclust publication (Bentham et al., 2017);
- this dissertation project repository/work.

## Author

Yevа Mykhailovska  
Final-year university dissertation project

