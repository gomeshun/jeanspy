"""Base-runtime portability regressions; no optional JAX dependency."""

import importlib.util
from pathlib import Path

import numpy as np


def test_dequad_import_does_not_require_platform_float128(monkeypatch):
    from jeanspy import dequad
    monkeypatch.delattr(np, 'float128', raising=False)
    spec = importlib.util.spec_from_file_location('portable_dequad', Path(dequad.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.dequad)
