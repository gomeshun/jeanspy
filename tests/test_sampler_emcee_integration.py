"""Integration coverage for the supported emcee-based Sampler API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from jeanspy.sampler import Sampler


class _ToyPosteriorModel:
    """Minimal model contract exercised by :class:`jeanspy.sampler.Sampler`."""

    ndim = 2
    prior_names = ["lnprior"]
    name = "ToyPosteriorModel"
    dsph_name = "toy"

    @staticmethod
    def convert_params(p):
        return pd.Series(np.asarray(p, dtype=float), index=["x", "y"])

    @staticmethod
    def lnposterior(p):
        p = np.asarray(p, dtype=float)
        lnprior = -0.5 * np.sum(p**2)
        lnl = -0.5 * np.sum((p - np.array([0.25, -0.15])) ** 2 / 0.5**2)
        return lnprior + lnl, lnl, lnprior

    lnposterior_wbic = lnposterior


def _initial_state(nwalkers):
    if nwalkers is None:
        return np.array([0.1, -0.1])
    rng = np.random.default_rng(12345)
    return rng.normal(loc=[0.1, -0.1], scale=[0.08, 0.08], size=(nwalkers, 2))


def test_sampler_runs_and_persists_emcee_chain(tmp_path: Path):
    model = _ToyPosteriorModel()
    sampler = Sampler(
        model,
        _initial_state,
        nwalkers=8,
        prefix=f"{tmp_path}/",
        reset=True,
    )

    sampler.run_mcmc(
        iterations=24,
        loops=1,
        p0_generator=_initial_state,
        enable_convergence_check=False,
    )

    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    blobs = sampler.get_blobs()

    assert chain.shape == (24, 8, 2)
    assert log_prob.shape == (24, 8)
    assert blobs.shape == (24, 8)
    assert blobs.dtype.names == ("lnl", "lnprior")
    assert np.isfinite(chain).all()
    assert np.isfinite(log_prob).all()
    assert np.isfinite(blobs["lnl"]).all()
    assert np.isfinite(blobs["lnprior"]).all()
    assert Path(sampler.filename).is_file()

    last = sampler.get_last_sample()
    assert last.coords.shape == (8, 2)
    assert np.isfinite(last.log_prob).all()
