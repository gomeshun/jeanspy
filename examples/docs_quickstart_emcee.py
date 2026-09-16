"""Run a small spherical inference example with NumPy/SciPy and emcee."""
# imports-start
import numpy as np
import emcee
from jeanspy.model import DSphModel, PlummerModel, ZhaoModel, ConstantAnisotropyModel

from pathlib import Path
import argparse
from docs_quickstart_data import data
from docs_quickstart_plots import plot_data, plot_posterior
R_pc, vlos_kms, e_vlos_kms = (data[k] for k in ("R_pc", "vlos_kms", "e_vlos_kms"))

# imports-end

# model-start
model = DSphModel(vmem_kms=0., submodels={
    "StellarModel": PlummerModel(re_pc=200.),
    "DMModel": ZhaoModel(rs_pc=500., rhos_Msunpc3=.1,
                         a=1., b=3., g=1., r_t_pc=np.inf),
    "AnisotropyModel": ConstantAnisotropyModel(beta_ani=0.),
})
sigma_kms = np.sqrt(model.sigmalos2(R_pc, n=128, n_kernel=32))
# model-end

# inference-start
# Sample halo radius, density, inner slope, anisotropy and systemic velocity.
# These illustrative priors are uniform in the named coordinates.
names = ["log10_rs_pc", "log10_rhos_Msunpc3", "g", "beta_ani", "vmem_kms"]
lower = np.array([1.5, -3., 0., -1., -50.])
upper = np.array([4., 1., 2., .75, 50.])


def log_probability(theta):
    if not np.all((lower < theta) & (theta < upper)):
        return -np.inf
    log_rs, log_rho, g, beta, mean = theta
    model.update(rs_pc=10**log_rs, rhos_Msunpc3=10**log_rho,
                 g=g, beta_ani=beta, vmem_kms=mean)
    variance = model.sigmalos2(R_pc, n=128, n_kernel=32) + e_vlos_kms**2
    return -.5 * np.sum(np.log(2*np.pi*variance) + (vlos_kms-mean)**2/variance)


def run(output_dir, num_steps=256, resume_steps=128):
    # A new directory prevents an accidental reset of an existing analysis.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savetxt(output_dir / "observations.csv", np.column_stack(list(data.values())),
               delimiter=",", header=",".join(data), comments="")
    plot_data(data, output_dir)
    rng = np.random.default_rng(42)
    initial = np.array([np.log10(500.), -1., 1., 0., 0.])
    walkers = initial + rng.normal(size=(16, len(names))) * [.05, .05, .05, .05, .5]
    backend = emcee.backends.HDFBackend(str(output_dir / "chain.h5"))
    sampler = emcee.EnsembleSampler(16, len(names), log_probability, backend=backend)
    sampler.random_state = np.random.RandomState(42).get_state()
    sampler.run_mcmc(walkers, num_steps, progress=False)
    first = backend.get_chain().copy()
    # resume-start
    # Reopen the same target and data. None loads the stored walker/random state.
    restored_backend = emcee.backends.HDFBackend(str(output_dir / "chain.h5"))
    resumed = emcee.EnsembleSampler(16, len(names), log_probability, backend=restored_backend)
    resumed.run_mcmc(None, resume_steps, progress=False)
    np.testing.assert_array_equal(restored_backend.get_chain()[:num_steps], first)
    # resume-end
    chain = restored_backend.get_chain(discard=num_steps//2)
    samples = chain.reshape(-1, len(names))
    print("Stored chain shape:", restored_backend.get_chain().shape)
    print("Warmup discarded:", num_steps//2)
    print("Original stored draws preserved:", True)
    print("Mean acceptance fraction:", round(float(resumed.acceptance_fraction.mean()), 3))
    print("Parameter                 16%      50%      84%")
    for name, row in zip(names, np.quantile(samples, [.16, .5, .84], axis=0).T):
        print(f"{name:24s} {row[0]:8.3f} {row[1]:8.3f} {row[2]:8.3f}")
    print("Short-run autocorrelation-time estimates:")
    print(np.round(emcee.autocorr.integrated_time(chain, tol=0), 1))
    print("Walkers interact: no R-hat from this single ensemble.")
    plot_posterior(chain.transpose(1, 0, 2), names,
                   [np.log10(500.), -1., 1., 0., 0.], output_dir)
    assert np.all(np.isfinite(restored_backend.get_log_prob()))
    return restored_backend
# inference-end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args().output_dir)
