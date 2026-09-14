"""Plot the actual observations and stored posterior draws in the Quickstart."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from emcee.autocorr import function_1d


def plot_data(data, output_dir):
    """Save the mock velocity observations, including measurement errors."""
    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    ax.errorbar(data["R_pc"], data["vlos_kms"], yerr=data["e_vlos_kms"],
                fmt="o", ms=4, color="#23558b", ecolor=".6")
    ax.set(xscale="log", xlabel="Projected radius [pc]", ylabel="LOS velocity [km/s]")
    fig.savefig(Path(output_dir) / "observations.png", dpi=150)
    plt.close(fig)


# plots-start
def plot_posterior(chain, names, truth, output_dir):
    # chain is (chain or walker, draw, parameter), after discarding warmup.
    # emcee walkers interact; these panels do not treat them as independent chains.
    flat = chain.reshape(-1, len(names))
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 1.8*len(names)),
                             layout="constrained")
    for k, name in enumerate(names):
        axes[k, 0].hist(flat[:, k], bins=30, density=True, color="#23558b", alpha=.8)
        axes[k, 0].axvline(truth[k], color="#bb4430", ls="--")
        axes[k, 0].set(ylabel=name)
        axes[k, 1].plot(chain[:, :, k].T, alpha=.45, lw=.6)
        axes[k, 1].axhline(truth[k], color="#bb4430", ls="--")
    axes[-1, 0].set_xlabel("Posterior value")
    axes[-1, 1].set_xlabel("Stored draw after warmup")
    fig.savefig(Path(output_dir) / "trace.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(names), len(names), figsize=(9, 9),
                             layout="constrained")
    shown = flat[::max(1, len(flat)//1500)]
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(flat[:, i], bins=30, density=True, color="#23558b", alpha=.8)
                low, median, high = np.quantile(flat[:, i], [.16, .5, .84])
                ax.axvspan(low, high, alpha=.2, color="gray")
                ax.axvline(median, color="black", lw=1)
            else:
                ax.scatter(shown[:, j], shown[:, i], s=2, alpha=.12, color="#23558b")
                ax.axhline(truth[i], color="#bb4430", ls="--", lw=1)
            ax.axvline(truth[j], color="#bb4430", ls="--", lw=1)
            if i == len(names)-1:
                ax.set_xlabel(name_j)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(name_i)
            else:
                ax.tick_params(labelleft=False)
    fig.savefig(Path(output_dir) / "posterior.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for k, name in enumerate(names):
        acf = np.mean([function_1d(c[:, k]) for c in chain], axis=0)
        ax.plot(acf[:min(100, len(acf))], label=name)
    ax.set(xlabel="Lag [stored draws]", ylabel="Mean autocorrelation")
    ax.legend(fontsize="small")
    fig.savefig(Path(output_dir) / "autocorrelation.png", dpi=150)
    plt.close(fig)
# plots-end
