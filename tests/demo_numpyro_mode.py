from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro import optim


def make_rng(seed: int = 0) -> jax.Array:
    return jax.random.PRNGKey(seed)


def linreg_model(X, y=None):
    w = numpyro.sample("w", dist.Normal(0.0, 10.0))
    b = numpyro.sample("b", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))
    mu = w * X + b
    numpyro.deterministic("mu", mu)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def make_synth_data(key, n=200, w_true=2.0, b_true=-1.0, sigma_true=0.5):
    key_x, key_eps = jax.random.split(key)
    X = jax.random.uniform(key_x, shape=(n,), minval=-2.0, maxval=2.0)
    eps = sigma_true * jax.random.normal(key_eps, shape=(n,))
    y = w_true * X + b_true + eps
    truth = {"w": float(w_true), "b": float(b_true), "sigma": float(sigma_true)}
    return X, y, truth


def run_nuts(key, X, y, *, warmup=500, samples=500):
    nuts = NUTS(linreg_model)
    mcmc = MCMC(nuts, num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=False)
    mcmc.run(key, X=X, y=y)
    return mcmc


def run_map(key, X, y, *, steps=2000, lr=2e-2):
    guide = AutoDelta(linreg_model)
    svi = SVI(linreg_model, guide, optim.Adam(lr), loss=Trace_ELBO())
    result = svi.run(key, steps, X=X, y=y)
    params_map = guide.median(result.params)
    return params_map, np.array(result.losses)


def run_demo(seed: int = 0, n: int = 200):
    numpyro.set_host_device_count(1)
    key = make_rng(seed)
    key, subkey = jax.random.split(key)
    X, y, truth = make_synth_data(subkey, n=n)

    key, subkey = jax.random.split(key)
    mcmc = run_nuts(subkey, X, y)
    post = mcmc.get_samples()
    post_mean = {k: float(jnp.mean(v)) for k, v in post.items() if k in ("w", "b", "sigma")}

    key, subkey = jax.random.split(key)
    params_map, losses = run_map(subkey, X, y)
    params_map = {k: float(v) for k, v in params_map.items() if k in ("w", "b", "sigma")}

    print("truth:", truth)
    print("posterior mean:", post_mean)
    print("MAP(mode):", params_map)
    print("SVI loss start/end:", float(losses[0]), float(losses[-1]))

    return {
        "truth": truth,
        "posterior_mean": post_mean,
        "map": params_map,
        "losses": losses,
    }


if __name__ == "__main__":
    run_demo()
