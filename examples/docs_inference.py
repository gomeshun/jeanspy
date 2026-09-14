"""Short classical save/restart workflow; these draws are not converged inference."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from jeanspy.model import get_default_estimation_model
from jeanspy.sampler import Sampler

# classical-inference-start
data = pd.DataFrame({"R_pc": [30., 100., 300., 500.],
                     "vlos_kms": [-4., 2., 8., -6.], "e_vlos_kms": [2.] * 4})
prior = pd.DataFrame(
    {"lower": [-30., 2., 2.5, -3., 3.5, -.3],
     "upper": [30., 2.6, 3.5, -1., 4.5, .3]},
    index=["vmem_kms", "log10_re_pc", "log10_rs_pc",
           "log10_rhos_Msunpc3", "log10_r_t_pc", "bfunc_beta_ani"],
)
model = get_default_estimation_model(data, 2.3, .1, config=prior)
center = np.array([0., 2.3, 3., -2., 4., 0.])

def initial_state(n):
    if n is None:  # Sampler also checks the conversion of one parameter vector.
        return center
    return center + np.random.default_rng(42).normal(0., .01, (n, len(center)))

# Use a persistent output directory for your own analysis. Here it is temporary.
with TemporaryDirectory() as directory:
    sampler = Sampler(model, initial_state, nwalkers=16, prefix=directory + "/")
    sampler.run_mcmc(4, 1, enable_convergence_check=False)
    first = sampler.get_chain().copy()
    resumed = Sampler(model, initial_state, nwalkers=16, prefix=directory + "/")
    resumed.run_mcmc(2, 1, enable_convergence_check=False)
    chain = resumed.get_chain()
    np.testing.assert_array_equal(chain[:4], first)
    assert chain.shape == (6, 16, 6) and Path(resumed.filename).is_file()
# Six steps only check this workflow. Do not report parameter estimates from it.
# classical-inference-end
