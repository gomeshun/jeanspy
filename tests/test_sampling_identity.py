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
from jeanspy.model import plummer_nfw_constant_anisotropy_model
from jeanspy.model_jax import DSphModel, PlummerModel, NFWModel, ConstantAnisotropyModel
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


@pytest.mark.parametrize('has_previous_run', [
    False, pytest.param(True, marks=pytest.mark.mcmc),
])
def test_numpyro_missing_metadata_stops_before_sampling(tmp_path, monkeypatch, has_previous_run):
    with NumPyroSampler(make_mcmc(), output_dir=tmp_path, async_writes=False) as sampler:
        if has_previous_run:
            sampler.run(jax.random.PRNGKey(0), save_samples=False)
        sampler.metadata_path.unlink()
        before = snapshot(tmp_path)
        identity = sampler._analysis_identity
        state = sampler.mcmc.post_warmup_state
        last_state = sampler.mcmc.last_state

        def forbidden(*args, **kwargs):
            pytest.fail('MCMC must not run without verifiable metadata')

        monkeypatch.setattr(sampler.mcmc, 'run', forbidden)
        for resume in [True, False, 'auto']:
            with pytest.raises(ValueError, match='metadata is missing.*analysis identity'):
                sampler.run(jax.random.PRNGKey(1), resume=resume)
            assert snapshot(tmp_path) == before
            assert sampler._analysis_identity is identity
            assert sampler.mcmc.post_warmup_state is state
            assert sampler.mcmc.last_state is last_state


@pytest.mark.mcmc
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


@pytest.mark.mcmc
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


@pytest.mark.mcmc
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


@pytest.mark.mcmc
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
    model = plummer_nfw_constant_anisotropy_model(data, 2.3, .1, config=classical_prior_config)
    before = fingerprint(model)
    model.update(model.convert_params(np.array([0., 2.3, 3., -2., 4., 0.])))
    assert fingerprint(model) == before
    model['PhotometryPriorModel'].reset_prior(2.4, .1)
    assert fingerprint(model) != before
    model['PhotometryPriorModel'].reset_prior(2.3, .1)
    assert fingerprint(model) == before
    model.reset_data(data.assign(vlos_kms=[10., 11.]))
    assert fingerprint(model) != before


@pytest.mark.mcmc
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
    model = plummer_nfw_constant_anisotropy_model(data, 2.3, .1, config=classical_prior_config)
    with pytest.raises(ValueError, match='at least two'):
        _ = model.inverse_temperature


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


@pytest.fixture
def editable_source(tmp_path, monkeypatch):
    """Use an isolated package tree without editing the running test package."""
    from jeanspy import _sampling_identity as identity
    root = tmp_path / 'package'
    root.mkdir()
    source = root / '__init__.py'
    source.write_text('"""Original documentation."""\nVALUE = 2\n')
    (root / 'data').mkdir()
    (root / 'data' / 'table.csv').write_text('value\n1\n')
    monkeypatch.setattr(identity, '__file__', str(source))
    return source


def test_software_identity_separates_documentation_from_computation(editable_source, monkeypatch):
    from jeanspy import _sampling_identity as identity
    before = identity.software_identity('emcee')
    provenance = identity.source_provenance()
    editable_source.write_text('"""Revised documentation.\nMore explanation."""\n# comment\nVALUE  = (2)\n')
    assert identity.software_identity('emcee') == before
    assert identity.source_provenance() != provenance
    editable_source.write_text('"""Revised documentation."""\nVALUE = 3\n')
    assert identity.software_identity('emcee') != before
    editable_source.write_text('VALUE = 2\n')
    assert identity.software_identity('emcee') == before
    data = editable_source.parent / 'data/table.csv'
    data.write_text('value\n2\n')
    assert identity.software_identity('emcee') != before
    data.write_text('value\n1\n')
    added = editable_source.parent / 'new_module.py'
    added.write_text('"""New module."""\n')
    assert identity.software_identity('emcee') != before
    added.unlink()
    real_version = identity.importlib.metadata.version
    monkeypatch.setattr(identity.importlib.metadata, 'version',
                        lambda name: 'changed' if name == 'numpy' else real_version(name))
    assert identity.software_identity('emcee') != before


@pytest.mark.parametrize('declared', [False, True])
def test_custom_class_documentation_is_not_target_state(tmp_path, monkeypatch, declared):
    import linecache
    import sys
    import types
    module = types.ModuleType('identity_external_model')
    module.__file__ = str(tmp_path / 'custom.py')
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def load(doc, expression='x * 2', default='1'):
        source = (f'class Target:\n    """{doc}"""\n'
                  f'    def __call__(self, x={default}):\n        """{doc}"""\n'
                  f'        return {expression}\n')
        if declared:
            source += '    def sampling_identity(self):\n        return {"state": 0}\n'
        Path(module.__file__).write_text(source)
        linecache.clearcache()
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
        return module.Target()

    before = fingerprint(load('Original'))
    assert fingerprint(load('Rewritten\n    documentation')) == before
    assert fingerprint(load('Rewritten', expression='x * 3')) != before
    assert fingerprint(load('Rewritten', default='2')) != before


def test_function_documentation_and_executable_strings_are_distinct():
    def load(doc, expression='x * 2'):
        namespace = {'__name__': 'identity_test'}
        exec(f'def target(x=1):\n    """{doc}"""\n    return {expression}\n', namespace)
        return namespace['target']
    before = fingerprint(load('Original'))
    assert fingerprint(load('Rewritten')) == before
    assert fingerprint(load('Rewritten', expression='x * 3')) != before
    assert fingerprint(load('same', expression='"same"')) != fingerprint(load('different', expression='"different"'))


@pytest.mark.mcmc
def test_emcee_documentation_change_resumes_with_source_history(tmp_path, editable_source):
    (tmp_path / 'chain').mkdir()
    prefix = str(tmp_path / 'chain') + '/'
    sampler = Sampler(ToyModel(), initial, nwalkers=6, prefix=prefix)
    sampler.run_mcmc(4, 1, enable_convergence_check=False)
    chain = sampler.get_chain().copy()
    with sampler.backend.open('r') as handle:
        history = [json.loads(v) for v in handle[sampler.backend_name]['jeanspy_source_provenance'].asstr()]
    editable_source.write_text('"""Updated documentation."""\nVALUE = 2\n')
    resumed = Sampler(ToyModel(), initial, nwalkers=6, prefix=prefix)
    resumed.run_mcmc(3, 1, enable_convergence_check=False)
    np.testing.assert_array_equal(resumed.get_chain()[:4], chain)
    assert resumed.get_chain().shape == (7, 6, 1)
    with resumed.backend.open('r') as handle:
        updated = [json.loads(v) for v in handle[resumed.backend_name]['jeanspy_source_provenance'].asstr()]
    assert updated[:1] == history
    assert len(updated) == 2 and updated[1]['iteration'] == 4
    assert updated[0]['source_sha256'] != updated[1]['source_sha256']
    editable_source.write_text('VALUE = 3\n')
    before = snapshot(tmp_path / 'chain')
    with pytest.raises(ValueError, match='identity mismatch'):
        Sampler(ToyModel(), initial, nwalkers=6, prefix=prefix)
    assert snapshot(tmp_path / 'chain') == before


@pytest.mark.mcmc
def test_numpyro_documentation_change_resumes_with_source_history(tmp_path, editable_source):
    output = tmp_path / 'chain'
    with NumPyroSampler(make_mcmc(), output_dir=output, async_writes=False) as first:
        first.run(jax.random.PRNGKey(0), save_samples=False)
    history = json.loads((output / 'metadata.json').read_text())['source_provenance']
    editable_source.write_text('"""Updated documentation."""\nVALUE = 2\n')
    with NumPyroSampler(make_mcmc(), output_dir=output, async_writes=False) as resumed:
        resumed.load_checkpoint()
        assert resumed.run(jax.random.PRNGKey(1), save_samples=False).resumed
    updated = json.loads((output / 'metadata.json').read_text())['source_provenance']
    assert updated[:1] == history
    assert len(updated) == 2
    assert updated[0]['source_sha256'] != updated[1]['source_sha256']
    editable_source.write_text('VALUE = 3\n')
    before = snapshot(output)
    with NumPyroSampler(make_mcmc(), output_dir=output, async_writes=False) as changed:
        with pytest.raises(ValueError, match='identity mismatch'):
            changed.load_checkpoint()
        with pytest.raises(ValueError, match='identity mismatch'):
            changed.run(jax.random.PRNGKey(2), save_samples=False)
    assert snapshot(output) == before


def test_emcee_rejects_old_identity_format_without_mutating_output(tmp_path):
    sampler = Sampler(ToyModel(), initial, nwalkers=6, prefix=str(tmp_path)+'/')
    with sampler.backend.open('a') as handle:
        handle[sampler.backend_name].attrs['jeanspy_identity_format'] = 1
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match='identity mismatch'):
        Sampler(ToyModel(), initial, nwalkers=6, prefix=str(tmp_path)+'/')
    assert snapshot(tmp_path) == before


@pytest.mark.mcmc
def test_numpyro_rejects_old_identity_format_without_mutating_output(tmp_path):
    import pickle
    with NumPyroSampler(make_mcmc(), output_dir=tmp_path, async_writes=False) as first:
        first.run(jax.random.PRNGKey(0), save_samples=False)
        metadata = json.loads(first.metadata_path.read_text())
        metadata['analysis_identity']['format'] = 1
        first.metadata_path.write_text(json.dumps(metadata))
        payload = pickle.loads(first.checkpoint_path.read_bytes())
        payload['analysis_identity']['format'] = 1
        first.checkpoint_path.write_bytes(pickle.dumps(payload))
    before = snapshot(tmp_path)
    with NumPyroSampler(make_mcmc(), output_dir=tmp_path, async_writes=False) as sampler:
        with pytest.raises(ValueError, match='identity mismatch'):
            sampler.load_checkpoint()
        with pytest.raises(ValueError, match='identity mismatch'):
            sampler.run(jax.random.PRNGKey(1), save_samples=False)
    assert snapshot(tmp_path) == before


@pytest.mark.mcmc
def test_emcee_multiple_loops_match_uninterrupted_transitions(tmp_path):
    old_random = np.random.get_state()
    chains = []
    try:
        for label, iterations, loops in [('split', 4, 2), ('uninterrupted', 8, 1)]:
            directory = tmp_path / label
            directory.mkdir()
            np.random.seed(314159)
            sampler = Sampler(ToyModel(), initial, nwalkers=6, prefix=str(directory)+'/')
            sampler.run_mcmc(iterations, loops, enable_convergence_check=False)
            chains.append(sampler.get_chain())
        np.testing.assert_array_equal(*chains)
    finally:
        np.random.set_state(old_random)


@pytest.mark.parametrize('body', ['pass', 'return None', 'return 2',
                                  'try:\n        return 1/0\n    except ZeroDivisionError:\n        return None'])
def test_adding_and_removing_function_docstrings_does_not_change_target(body):
    values = []
    for doc in ('', '    """Help."""\n', '    """More\n    help."""\n'):
        namespace = {'__name__': 'identity_test'}
        exec('def target():\n' + doc + '    ' + body + '\n', namespace)
        values.append(fingerprint(namespace['target']))
    assert len(set(values)) == 1


def test_documentation_used_as_data_can_be_declared_explicitly():
    def target():
        """1.0"""
        return float(target.__doc__)
    target.sampling_identity = lambda: {'documentation_as_data': target.__doc__}
    before = fingerprint(target)
    target.__doc__ = '2.0'
    assert fingerprint(target) != before


def test_documentation_presence_flag_is_not_computational_state(monkeypatch):
    import inspect
    from jeanspy._sampling_identity import _code
    def target():
        return None
    # Python 3.14 records docstring presence in a dedicated code flag.
    flag = getattr(inspect, 'CO_HAS_DOCSTRING', 1 << 26)
    monkeypatch.setattr(inspect, 'CO_HAS_DOCSTRING', flag, raising=False)
    without = target.__code__.replace(co_flags=target.__code__.co_flags & ~flag)
    with_doc = without.replace(co_flags=without.co_flags | flag)
    assert _code(without) == _code(with_doc)
