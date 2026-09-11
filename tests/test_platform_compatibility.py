"""Base-runtime portability regressions; no optional JAX dependency."""

import importlib.util
from pathlib import Path

import numpy as np

from jeanspy._sampling_identity import fingerprint


def test_dequad_import_does_not_require_platform_float128(monkeypatch):
    from jeanspy import dequad
    monkeypatch.delattr(np, 'float128', raising=False)
    spec = importlib.util.spec_from_file_location('portable_dequad', Path(dequad.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.dequad)


def test_attribute_name_does_not_capture_unrelated_global(monkeypatch):
    from threading import RLock
    def model(value):
        return value.logger
    previous = fingerprint(model)
    monkeypatch.setitem(model.__globals__, 'logger', RLock())
    assert fingerprint(model) == previous


def test_nested_expression_global_is_part_of_identity(monkeypatch):
    def model():
        return sum(offset for _ in range(3))
    monkeypatch.setitem(model.__globals__, 'offset', 1.)
    previous = fingerprint(model)
    monkeypatch.setitem(model.__globals__, 'offset', 2.)
    assert fingerprint(model) != previous
