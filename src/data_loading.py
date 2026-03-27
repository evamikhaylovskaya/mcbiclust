import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def make_unique(names):
    """
    Make gene names unique by appending _1, _2, ...
    """
    counts = {}
    result = []

    for name in names:
        name = str(name)

        if name not in counts:
            counts[name] = 0
            result.append(name)
        else:
            counts[name] += 1
            result.append(f"{name}_{counts[name]}")

    return np.array(result, dtype=str)


# ─────────────────────────────────────────────────────────────
# Dataset container
# ─────────────────────────────────────────────────────────────

class ExpressionDataset:
    """
    Container for gene expression matrix.

    Attributes
    ----------
    X : np.ndarray
        Expression matrix (genes x samples)

    genes : np.ndarray
        Gene names

    samples : np.ndarray
        Sample names

    name : str
        Dataset name
    """

    def __init__(self, X, genes, samples, name="dataset"):

        self.X = np.asarray(X, dtype=np.float64)
        self.genes = np.asarray(genes)
        self.samples = np.asarray(samples)
        self.name = name

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D matrix")

        if self.X.shape[0] != len(self.genes):
            raise ValueError("Number of rows in X must match genes")

        if self.X.shape[1] != len(self.samples):
            raise ValueError("Number of columns in X must match samples")


    # ─────────────────────────────────────────────────────────
    # Basic info
    # ─────────────────────────────────────────────────────────

    @property
    def n_genes(self):
        return self.X.shape[0]

    @property
    def n_samples(self):
        return self.X.shape[1]


    def summary(self):
        print(f"Dataset:  {self.name}")
        print(f"Shape:    {self.n_genes} genes x {self.n_samples} samples")
        print(f"Min:      {self.X.min():.4f}")
        print(f"Max:      {self.X.max():.4f}")
        print(f"Mean:     {self.X.mean():.4f}")
        print(f"Std:      {self.X.std():.4f}")


    # ─────────────────────────────────────────────────────────
    # Visualisation
    # ─────────────────────────────────────────────────────────

    def plot_distribution(self, bins=100, title=None):

        plt.figure(figsize=(8,4))
        plt.hist(self.X.flatten(), bins=bins)

        plt.title(title or f"{self.name}: Expression distribution")
        plt.xlabel("Expression value")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.show()


    def row_zscore(self):

        return (self.X - self.X.mean(axis=1, keepdims=True)) / (
               self.X.std(axis=1, keepdims=True) + 1e-8)

    def plot_heatmap(self, figsize=(14, 10), cmap="RdBu_r",
                     max_genes=None, vmin=None, vmax=None):

        X_plot = self.row_zscore()

        if max_genes is not None and X_plot.shape[0] > max_genes:
            X_plot = X_plot[:max_genes]

        print("Clustering genes...")
        gene_order = leaves_list(
            linkage(pdist(X_plot, metric="euclidean"), method="average")
        )

        print("Clustering samples...")
        sample_order = leaves_list(
            linkage(pdist(X_plot.T, metric="euclidean"), method="average")
        )

        data_reordered = X_plot[gene_order][:, sample_order]

        plt.figure(figsize=figsize)

        plt.imshow(
            data_reordered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="none"
        )

        plt.colorbar(label="Expression (row-normalised)")

        plt.title(
            f"{self.name}: Expression Heatmap\n"
            f"({data_reordered.shape[0]} genes x {data_reordered.shape[1]} samples)"
        )

        plt.xlabel("Samples")
        plt.ylabel("Genes")

        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────
    # Conversion
    # ─────────────────────────────────────────────────────────

    def to_dataframe(self):

        return pd.DataFrame(
            self.X,
            index=self.genes,
            columns=self.samples
        )


# ─────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────

def load_ecoli(path):
    """
    Load E.coli M3D expression matrix.
    """

    df = pd.read_csv(path, sep="\t", index_col=0)

    df = df.apply(pd.to_numeric, errors="coerce")

    X = df.to_numpy(dtype=np.float64)

    X = np.nan_to_num(X, nan=0.0)

    genes = df.index.to_numpy()
    samples = df.columns.to_numpy()

    return ExpressionDataset(
        X,
        genes,
        samples,
        name="E.coli M3D"
    )


def load_ccle_raw(path):
    """
    Load raw CCLE RNAseq dataset (.gct format).

    No preprocessing is applied here.
    """

    df = pd.read_csv(path, sep="\t", skiprows=2)

    genes = make_unique(
        df["Description"].astype(str).values
    )

    samples = df.columns[2:].astype(str).values

    X = df.iloc[:,2:].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=np.float64)

    X = np.nan_to_num(X, nan=0.0)

    return ExpressionDataset(
        X,
        genes,
        samples,
        name="CCLE raw"
    )


def load_expression_matrix(path, sep="\t", index_col=0, name="dataset"):
    """
    Generic loader for any gene expression matrix.

    Parameters
    ----------
    path : str
        Path to the file
    sep : str
        Delimiter (default: tab)
    index_col : int
        Column to use as gene index (default: 0)
    name : str
        Dataset name

    Returns
    -------
    ExpressionDataset
    """
    df = pd.read_csv(path, sep=sep, index_col=index_col)
    df = df.apply(pd.to_numeric, errors="coerce")

    X = df.to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)

    genes = df.index.to_numpy()
    samples = df.columns.to_numpy()

    return ExpressionDataset(X, genes, samples, name=name)


def load_ccle_mitocarta(
    ccle_path: str,
    mitocarta_path: str,
    pseudocount: float = 1.0,
    mitocarta_sheet: str = "A Human MitoCarta3.0",
    symbol_col: str = "Symbol",
    clean_ccle_gene_names: bool = True,
):
    """
    Load CCLE, apply log2 transform, load MitoCarta, and match mitochondrial genes.

    Parameters
    ----------
    ccle_path : str
        Path to CCLE GCT file.

    mitocarta_path : str
        Path to MitoCarta .xls or .csv file.

    pseudocount : float
        Pseudocount for log2 transform.

    mitocarta_sheet : str
        Excel sheet name for MitoCarta, used only for .xls/.xlsx files.

    symbol_col : str
        Column name containing gene symbols in MitoCarta.

    clean_ccle_gene_names : bool
        If True, strip suffixes like 'TP53 (7157)' -> 'TP53' before matching.

    Returns
    -------
    ccle_log : ExpressionDataset
        Full log-transformed CCLE dataset.

    gene_set : np.ndarray
        Indices of matched mitochondrial genes in ccle_log.

    ccle_mito : ExpressionDataset
        Mitochondrial-only subset dataset.

    mitocarta_genes : np.ndarray
        Raw gene symbol list from MitoCarta.
    """
    # 1. Load CCLE
    ccle = load_ccle_raw(ccle_path)
    if pseudocount <= 0:
        raise ValueError("pseudocount must be > 0")

    ccle_log = ExpressionDataset(
        X=np.log2(ccle.X + pseudocount),
        genes=ccle.genes.copy(),
        samples=ccle.samples.copy(),
        name=f"{ccle.name} | log2"
    )

    # 2. Load MitoCarta
    if mitocarta_path.lower().endswith(".csv"):
        mito_df = pd.read_csv(mitocarta_path)
    elif mitocarta_path.lower().endswith((".xls", ".xlsx")):
        mito_df = pd.read_excel(mitocarta_path, sheet_name=mitocarta_sheet)
    else:
        raise ValueError("mitocarta_path must be .csv, .xls, or .xlsx")

    mito_df.columns = mito_df.columns.astype(str).str.strip()

    if symbol_col not in mito_df.columns:
        raise KeyError(
            f"Column '{symbol_col}' not found in MitoCarta file. "
            f"Available columns: {mito_df.columns.tolist()}"
        )

    mitocarta_genes = (
        mito_df[symbol_col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    # 3. Match genes
    if clean_ccle_gene_names:
        ccle_gene_names = (
            pd.Series(ccle_log.genes)
            .astype(str)
            .str.split(r"\s*\(")
            .str[0]
            .str.strip()
            .to_numpy()
        )
    else:
        ccle_gene_names = np.asarray(ccle_log.genes).astype(str)

    gene_set = np.where(np.isin(ccle_gene_names, mitocarta_genes))[0]

    # 4. Build mito-only subset
    ccle_mito = ExpressionDataset(
        X=ccle_log.X[gene_set],
        genes=ccle_log.genes[gene_set],
        samples=ccle_log.samples.copy(),
        name=f"{ccle_log.name} | {len(gene_set)}_mitocarta_genes"
    )

    # 5. Print summary
    print(f"MitoCarta genes in file: {len(mitocarta_genes)}")
    print(f"Matched mitochondrial genes in CCLE: {len(gene_set)}")

    if len(gene_set) == 0:
        print("Warning: no genes matched.")
    else:
        print("First matched genes:", ccle_log.genes[gene_set[:10]])

    return ccle_log, gene_set, ccle_mito, mitocarta_genes