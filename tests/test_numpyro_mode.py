import jax
import jax.numpy as jnp
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_numpyro_mode as d


def test_run_map_finite_and_reasonable():
    key = d.make_rng(0)
    key, subkey = jax.random.split(key)
    X, y, truth = d.make_synth_data(subkey, n=200, w_true=2.0, b_true=-1.0, sigma_true=0.5)

    key, subkey = jax.random.split(key)
    params_map, losses = d.run_map(subkey, X, y, steps=1500, lr=2e-2)

    # finite
    for name in ("w", "b", "sigma"):
        assert jnp.isfinite(params_map[name])

    # sigma positive
    assert float(params_map["sigma"]) > 0.0

    # loss decreases (not necessarily monotonic)
    assert float(losses[-1]) < float(losses[0])

    # rough closeness (wide tolerance; keep this stable)
    assert abs(float(params_map["w"]) - truth["w"]) < 1.0
    assert abs(float(params_map["b"]) - truth["b"]) < 1.0


@pytest.mark.mcmc
def test_run_demo_returns_expected_keys():
    out = d.run_demo(seed=0, n=100)
    assert set(out.keys()) >= {"truth", "posterior_mean", "map", "losses"}
