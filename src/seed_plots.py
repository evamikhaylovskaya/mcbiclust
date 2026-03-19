import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_alpha_vs_iterations(steps, alpha_history, title="FindSeed: Alpha Improvement"):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, alpha_history, marker="o", lw=2)
    plt.axhline(y=max(alpha_history), color="red", linestyle="--")
    plt.xlabel("Iterations")
    plt.ylabel("Alpha")
    plt.title(title)
    plt.legend([f"Final alpha = {max(alpha_history):.4f}"])
    plt.tight_layout()
    plt.show()


def plot_sample_corr_heatmap(X, subset_size=100, seed=42, cluster=False, vmin=0.0, vmax=1.0):
    rng = np.random.default_rng(seed)
    subset_size = min(subset_size, X.shape[1])

    sample_idx = np.sort(rng.choice(X.shape[1], subset_size, replace=False))
    X_sample_sub = X[:, sample_idx]
    C_sample_sub = np.corrcoef(X_sample_sub, rowvar=False)

    if cluster:
        g = sns.clustermap(
            C_sample_sub,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            figsize=(7, 6),
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "Pearson r"},
        )
        g.fig.suptitle(
            f"Sample-Sample Correlation (random {subset_size} samples)",
            y=1.02,
        )
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            C_sample_sub,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            cbar_kws={"label": "Pearson r"},
            ax=ax,
        )
        ax.set_title(f"Sample-Sample Correlation (random {subset_size} samples)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Samples")

        ticks = np.arange(0, subset_size, max(1, subset_size // 25))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
        ax.invert_yaxis()

        plt.tight_layout()
        plt.show()