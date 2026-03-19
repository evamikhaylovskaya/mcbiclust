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

        self.X = np.asarray(X, dtype=np.float32)
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

    X = df.to_numpy(dtype=np.float32)

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
    ).to_numpy(dtype=np.float32)

    X = np.nan_to_num(X, nan=0.0)

    return ExpressionDataset(
        X,
        genes,
        samples,
        name="CCLE raw"
    )