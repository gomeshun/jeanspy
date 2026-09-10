"""A changed target must never reuse cached energy or append to an old chain."""

from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pandas as pd
import pytest

from jeanspy._sampling_identity import fingerprint
from jeanspy.model import get_default_estimation_model
from jeanspy.model_numpyro import DSphModel, PlummerModel, NFWModel, ConstantAnisotropyModel
from jeanspy.sampler import Sampler
from jeanspy.sampler_numpyro import NumPyroSampler, JeansLikelihoodModel, ParameterSpec


def normal_model(center):
    def model(y=None):
        x = numpyro.sample('x', dist.Normal(center, 1.))
        numpyro.sample('y', dist.Normal(x, 1.), obs=y)
    return model


def make_mcmc(center=0.):
    return MCMC(NUTS(normal_model(center)), num_warmup=5, num_samples=4,
                num_chains=1, progress_bar=False)


def snapshot(path):
    return {p.relative_to(path): p.read_bytes() for p in Path(path).rglob('*') if p.is_file()}


def test_numpyro_prior_and_data_mismatch_preserves_output(tmp_path):
    observed = jnp.array([0., .1])
    with NumPyroSampler(make_mcmc(), output_dir=tmp_path, async_writes=False) as first:
        first.run(jax.random.PRNGKey(0), y=observed, save_samples=False)
        first.run(jax.random.PRNGKey(1), y=observed, save_samples=False)
        before = snapshot(tmp_path)
        state = first.mcmc.post_warmup_state
        for resume in [True, False, 'auto']:
            with pytest.raises(ValueError, match='identity mismatch'):
                first.run(jax.random.PRNGKey(2), y=observed+100., resume=resume, save_samples=False)
            assert snapshot(tmp_path) == before
            assert first.mcmc.post_warmup_state is state
        first.mcmc.sampler._model = normal_model(100.)
        with pytest.raises(ValueError, match='identity mismatch'):
            first.run(jax.random.PRNGKey(2), y=observed, save_samples=False)
        assert snapshot(tmp_path) == before
    with NumPyroSampler(make_mcmc(100.), output_dir=tmp_path, async_writes=False) as changed:
        with pytest.raises(ValueError, match='identity mismatch'):
            changed.load_checkpoint()
        with pytest.raises(ValueError, match='identity mismatch'):
            changed.run(jax.random.PRNGKey(3), y=observed, save_samples=False)
        assert changed.mcmc.post_warmup_state is None
        assert snapshot(tmp_path) == before
    with NumPyroSampler(make_mcmc(), output_dir=tmp_path, async_writes=False) as same:
        same.load_checkpoint()
        result = same.run(jax.random.PRNGKey(4), y=observed, save_samples=False)
        assert result.resumed


def test_numpyro_legacy_output_cannot_be_silently_adopted(tmp_path):
    with NumPyroSampler(make_mcmc(), output_dir=tmp_path, async_writes=False) as first:
        first.run(jax.random.PRNGKey(0), save_samples=False)
        metadata = json.loads(first.metadata_path.read_text())
        del metadata['analysis_identity']
        first.metadata_path.write_text(json.dumps(metadata))
    before = snapshot(tmp_path)
    with NumPyroSampler(make_mcmc(), output_dir=tmp_path, async_writes=False) as sampler:
        with pytest.raises(ValueError, match='no analysis identity'):
            sampler.run(jax.random.PRNGKey(1), save_samples=False)
    assert snapshot(tmp_path) == before


def test_numpyro_unverified_external_state_cannot_be_resumed(tmp_path):
    raw = make_mcmc()
    raw.run(jax.random.PRNGKey(0))
    with NumPyroSampler(raw, output_dir=tmp_path, async_writes=False) as sampler:
        with pytest.raises(ValueError, match='Unverified in-memory'):
            sampler.run(jax.random.PRNGKey(1), save_samples=False)


def test_jeans_target_hash_tracks_prior_schema_and_solver_but_not_jit_cache(tmp_path):
    dsph = DSphModel(submodels={'StellarModel': PlummerModel(), 'DMModel': NFWModel(),
                               'AnisotropyModel': ConstantAnisotropyModel()})
    specs = [ParameterSpec('a', dist.Normal(0., 1.)), ParameterSpec('b', dist.Normal(0., 1.))]
    model = JeansLikelihoodModel(dsph, specs, sigmalos2_kwargs={'n_u': 32})
    with NumPyroSampler(MCMC(NUTS(model), num_warmup=5, num_samples=4), output_dir=tmp_path) as sampler:
        original = sampler._target_fingerprint()
        dsph._jit_cache['compiled'] = object()
        assert sampler._target_fingerprint() == original
        model.sigmalos2_kwargs['n_u'] = 64
        assert sampler._target_fingerprint() != original
        model.sigmalos2_kwargs['n_u'] = 32
        model.parameter_specs = tuple(reversed(specs))
        assert sampler._target_fingerprint() != original
        model.parameter_specs = tuple(specs)
        specs[0].distribution.loc = 100.
        assert sampler._target_fingerprint() != original


class ToyModel:
    name = 'identity-toy'
    ndim = 1
    prior_names = ['prior']

    def __init__(self, center=0.):
        self.center = center

    def convert_params(self, p):
        return pd.Series(p, index=['x'])

    def lnposterior(self, p):
        logp = -.5*(p[0]-self.center)**2
        return logp, logp, 0.


def initial(n):
    return np.linspace(-.4, .4, n).reshape(n, 1) if n is not None else np.array([0.])


def test_emcee_changed_target_rejected_on_reopen_and_in_memory(tmp_path):
    original = ToyModel()
    prefix = str(tmp_path) + '/'
    sampler = Sampler(original, initial, nwalkers=6, prefix=prefix)
    sampler.run_mcmc(6, 1, enable_convergence_check=False)
    before = snapshot(tmp_path)
    original.center = 100.
    with pytest.raises(ValueError, match='identity mismatch'):
        sampler.run_mcmc(3, 1, enable_convergence_check=False)
    assert snapshot(tmp_path) == before
    with pytest.raises(ValueError, match='identity mismatch'):
        Sampler(ToyModel(100.), initial, nwalkers=6, prefix=prefix)
    assert snapshot(tmp_path) == before
    restored = Sampler(ToyModel(), initial, nwalkers=6, prefix=prefix)
    old = restored.get_chain().copy()
    restored.run_mcmc(3, 1, enable_convergence_check=False)
    np.testing.assert_array_equal(restored.get_chain()[:6], old)
    assert restored.get_chain().shape == (9, 6, 1)


def test_classical_identity_tracks_data_prior_and_not_sampled_coordinates(classical_prior_config):
    data = pd.DataFrame(dict(R_pc=[10., 20.], vlos_kms=[0., 1.], e_vlos_kms=[1., 1.]))
    model = get_default_estimation_model(data, 2.3, .1, config=classical_prior_config)
    before = fingerprint(model)
    model.update(model.convert_params(np.array([0., 2.3, 3., -2., 4., 0.])))
    assert fingerprint(model) == before
    model['PhotometryPriorModel'].reset_prior(2.4, .1)
    assert fingerprint(model) != before
    model['PhotometryPriorModel'].reset_prior(2.3, .1)
    assert fingerprint(model) == before
    model.reset_data(data.assign(vlos_kms=[10., 11.]))
    assert fingerprint(model) != before


def test_emcee_burn_in_continues_final_ensemble_without_reweighting(tmp_path, monkeypatch):
    sampler = Sampler(ToyModel(), initial, nwalkers=6, prefix=str(tmp_path)+'/')
    def forbidden(*args, **kwargs):
        raise AssertionError('posterior draws must not be resampled with posterior weights')
    monkeypatch.setattr(np.random, 'choice', forbidden)
    state = sampler.burn_in(5, initial)
    assert sampler.backend.iteration == 5
    np.testing.assert_array_equal(state.coords, sampler.get_chain()[-1])
    sampler.run_mcmc(3, 1, enable_convergence_check=False)
    assert sampler.get_chain(discard=5).shape == (3, 6, 1)


def test_wbic_rejects_single_observation(classical_prior_config):
    data = pd.DataFrame(dict(R_pc=[10.], vlos_kms=[0.], e_vlos_kms=[1.]))
    model = get_default_estimation_model(data, 2.3, .1, config=classical_prior_config)
    with pytest.raises(ValueError, match='at least two'):
        _ = model.inverse_temparature


def test_opaque_target_state_requires_explicit_identity():
    with pytest.raises(TypeError, match='sampling_identity'):
        fingerprint(object())


def test_callable_class_captures_changed_global_state():
    global CALLABLE_TARGET_CENTER
    CALLABLE_TARGET_CENTER = 0.
    class CallableModel:
        def __call__(self):
            return CALLABLE_TARGET_CENTER
    model = CallableModel()
    old = fingerprint(model)
    CALLABLE_TARGET_CENTER = 100.
    assert fingerprint(model) != old


def test_jitted_user_transforms_keep_their_captured_parameters():
    def make_transform(offset):
        return jax.jit(lambda value: value + offset)
    assert fingerprint(make_transform(1.)) == fingerprint(make_transform(1.))
    assert fingerprint(make_transform(1.)) != fingerprint(make_transform(2.))


def test_series_identity_includes_labels():
    a = pd.Series([1., 2.], index=['star1', 'star2'])
    b = pd.Series([1., 2.], index=['star2', 'star1'])
    assert fingerprint(a) != fingerprint(b)


def test_backend_precision_is_part_of_numpyro_identity(tmp_path, monkeypatch):
    with NumPyroSampler(make_mcmc(), output_dir=tmp_path) as sampler:
        original = sampler._target_fingerprint()
        precision = 'high' if jax.config.jax_default_matmul_precision == 'highest' else 'highest'
        with jax.default_matmul_precision(precision):
            assert sampler._target_fingerprint() != original
        assert sampler._target_fingerprint() == original
        monkeypatch.setattr(jax, 'default_backend', lambda: 'different-platform')
        assert sampler._target_fingerprint() != original


def test_explicit_identity_can_describe_opaque_state():
    class CustomModel:
        opaque = object()
        def __init__(self):
            self.center = 1.
        def __call__(self):
            return self.center
        def sampling_identity(self):
            return {'center': self.center, 'implementation_version': 1}
    model = CustomModel()
    previous = fingerprint(model)
    model.opaque = object()
    assert fingerprint(model) == previous
    model.center = 2.
    assert fingerprint(model) != previous


def test_function_identity_provider_can_describe_external_state():
    state = {'center': 1., 'opaque': object()}
    def model():
        return state['center']
    model.sampling_identity = lambda: {'center': state['center']}
    previous = fingerprint(model)
    state['opaque'] = object()
    assert fingerprint(model) == previous
    state['center'] = 2.
    assert fingerprint(model) != previous
