"""Adversarial checkpoint audit: change target while retaining output location.

All files are written to new temporary directories. No repository data are used.
"""
import json
import logging
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpyro import distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value
from jeanspy.sampler_numpyro import NumPyroSampler
from jeanspy.sampler import Sampler

def normal_model(center):
    def model():
        numpyro.sample('x', dist.Normal(center, 1.))
    return model

def mcmc(center):
    return MCMC(NUTS(normal_model(center), init_strategy=init_to_value(values={'x': jnp.array(center)})),
                num_warmup=40, num_samples=12, num_chains=1, progress_bar=False)

results = {}
with tempfile.TemporaryDirectory(prefix='jeanspy-review-resume-') as tmp:
    with NumPyroSampler(mcmc(0.), output_dir=tmp, storage_backend='h5netcdf', async_writes=False) as first:
        first.run(jax.random.PRNGKey(0))
        results['numpyro_first_mean'] = float(first.get_samples()['x'].mean())
        z = float(first.last_state.z['x'])
        results['numpyro_saved_z'] = z
        results['numpyro_saved_potential'] = float(first.last_state.potential_energy)
        results['numpyro_actual_potential_for_new_target'] = .5*(z-100.)**2+.5*np.log(2*np.pi)
    with NumPyroSampler(mcmc(100.), output_dir=tmp, storage_backend='h5netcdf', async_writes=False) as second:
        run = second.run(jax.random.PRNGKey(1))
        results['numpyro_changed_target_auto_resumed'] = run.resumed
        results['numpyro_second_draws'] = np.asarray(second.get_samples()['x']).tolist()
        results['numpyro_combined_draws'] = int(second.load_samples()['posterior'].sizes['draw'])
    fresh = mcmc(100.)
    fresh.run(jax.random.PRNGKey(1))
    results['numpyro_fresh_target100_mean'] = float(fresh.get_samples()['x'].mean())

class ToyModel:
    ndim = 1
    name = 'ReviewNormal'
    prior_names = ['prior']
    def __init__(self, center): self.center = center
    def convert_params(self, p): return pd.Series(p, index=['x'])
    def lnposterior(self, p):
        lnl = -.5*np.sum((np.asarray(p)-self.center)**2)
        return lnl, lnl, 0.
    lnposterior_wbic = lnposterior

def initial(n):
    if n is None: return np.array([0.])
    return np.linspace(-1., 1., n).reshape(-1, 1)

with tempfile.TemporaryDirectory(prefix='jeanspy-review-emcee-') as tmp:
    np.random.seed(42)
    first = Sampler(ToyModel(0.), initial, nwalkers=8, prefix=tmp+'/')
    first.run_mcmc(5, 1, enable_convergence_check=False)
    first_final = first.get_last_sample().coords.copy()
    second = Sampler(ToyModel(100.), initial, nwalkers=8, prefix=tmp+'/')
    second.run_mcmc(3, 1, enable_convergence_check=False)
    results['emcee_total_steps'] = int(second.backend.iteration)
    results['emcee_all_new_steps_equal_old_final'] = bool(np.all(second.get_chain()[-3:] == first_final))
    actual = np.array([second.model.lnposterior(p)[0] for p in second.get_last_sample().coords])
    results['emcee_max_cached_log_prob_error'] = float(np.max(np.abs(second.get_last_sample().log_prob-actual)))

print(json.dumps(results, indent=2))
