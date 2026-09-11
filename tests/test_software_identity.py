"""Software changes in a running process must invalidate saved-target identities."""

import pytest

from jeanspy import _sampling_identity


@pytest.mark.parametrize('changed_component', ['source', 'data', 'dependency'])
def test_software_identity_detects_changes_between_calls(tmp_path, monkeypatch, changed_component):
    source = tmp_path / 'model.py'
    source.write_text('coefficient = 1\n')
    data = tmp_path / 'data' / 'coefficients.csv'
    data.parent.mkdir()
    data.write_text('coefficient\n1\n')
    versions = dict.fromkeys(['numpy', 'scipy', 'pandas', 'emcee'], '1.0')
    monkeypatch.setattr(_sampling_identity, '__file__', str(tmp_path / '_sampling_identity.py'))
    monkeypatch.setattr(_sampling_identity.importlib.metadata, 'version', versions.__getitem__)

    original = _sampling_identity.software_identity('emcee')
    assert _sampling_identity.software_identity('emcee') == original
    if changed_component == 'source':
        source.write_text('coefficient = 2\n')
    elif changed_component == 'data':
        data.write_text('coefficient\n2\n')
    else:
        versions['emcee'] = '2.0'
    assert _sampling_identity.software_identity('emcee') != original
