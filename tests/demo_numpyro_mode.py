"""Minimal numpyro MAP-mode demo used by test_numpyro_mode.py."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import optax


def make_rng(seed: int):
    return jax.random.PRNGKey(seed)


def make_synth_data(key, *, n: int = 200, w_true: float = 2.0, b_true: float = -1.0, sigma_true: float = 0.5):
    key_x, key_eps = jax.random.split(key)
    X = jax.random.normal(key_x, (n,))
    eps = jax.random.normal(key_eps, (n,)) * sigma_true
    y = w_true * X + b_true + eps
    truth = {"w": w_true, "b": b_true, "sigma": sigma_true}
    return X, y, truth


def _model(X, y=None):
    w = numpyro.sample("w", dist.Normal(0.0, 10.0))
    b = numpyro.sample("b", dist.Normal(0.0, 10.0))
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0.0, 2.0))
    sigma = numpyro.deterministic("sigma", jnp.exp(log_sigma))
    mu = w * X + b
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def run_map(key, X, y, *, steps: int = 1500, lr: float = 2e-2):
    guide = AutoDelta(_model)
    optimizer = numpyro.optim.optax_to_numpyro(optax.adam(lr))
    svi = SVI(_model, guide, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(key, X, y)

    losses = []
    for _ in range(steps):
        svi_state, loss = svi.update(svi_state, X, y)
        losses.append(float(loss))

    params_svi = svi.get_params(svi_state)
    map_estimates = guide.median(params_svi)
    # expose sigma (not log_sigma) directly
    if "sigma" not in map_estimates and "log_sigma" in map_estimates:
        map_estimates = dict(map_estimates, sigma=jnp.exp(map_estimates["log_sigma"]))
    return map_estimates, jnp.array(losses)


def run_demo(*, seed: int = 0, n: int = 200):
    key = make_rng(seed)
    key, subkey = jax.random.split(key)
    X, y, truth = make_synth_data(subkey, n=n)

    key, subkey = jax.random.split(key)
    params_map, losses = run_map(subkey, X, y, steps=1500, lr=2e-2)

    # Simple posterior mean estimate via another MAP run (placeholder)
    posterior_mean = {k: float(v) for k, v in params_map.items()}

    return {
        "truth": truth,
        "map": {k: float(v) for k, v in params_map.items()},
        "posterior_mean": posterior_mean,
        "losses": losses,
    }
