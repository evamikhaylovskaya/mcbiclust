import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def avg_abs_corr_rows(A, return_corr=False):
    """
    Compute MCbiclust alpha score:
    mean absolute correlation between all gene pairs.

    Parameters
    ----------
    A : np.ndarray
        Expression matrix (genes x samples)

    return_corr : bool
        If True, also return correlation matrix.

    Returns
    -------
    alpha : float
        Average absolute correlation
    C : np.ndarray (optional)
        Gene–gene correlation matrix
    """

    A = np.asarray(A, dtype=np.float64)
    genes, samples = A.shape

    if genes < 2 or samples < 2:
        return 0.0

    # gene-gene correlation
    C = np.corrcoef(A)
    C = np.nan_to_num(C, nan=0.0)

    absC = np.abs(C)

    alpha = absC.sum() / (genes * genes)

    if return_corr:
        return float(alpha), C
    else:
        return float(alpha)

