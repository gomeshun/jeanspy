"""Conservative, deterministic identities for persisted inference targets.

This is an accidental-mismatch guard, not a sandbox for arbitrary Python code.
Custom models that read hidden state (files, services, mutable extension objects)
must expose that state through ``sampling_identity()``. Opaque objects fail
closed instead of being identified by their memory address or repr.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import dis
import functools
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd


def _name(value):
    return f"{value.__module__}.{value.__qualname__}"


def _code(code):
    return [code.co_code.hex(), code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags,
            [_code(v) if isinstance(v, types.CodeType) else v for v in code.co_consts]]


def _function_globals(function):
    # Older inspect.getclosurevars versions search co_names, which also
    # contains attribute names. self.logger must not capture a module-global
    # logger (and its locks). Include only actual global loads, including those
    # in nested comprehensions/generator code.
    def loaded_names(code):
        for instruction in dis.get_instructions(code):
            if instruction.opname in {'LOAD_GLOBAL', 'LOAD_NAME'}:
                yield instruction.argval
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                yield from loaded_names(constant)
    return {name: function.__globals__[name] for name in loaded_names(function.__code__)
            if name in function.__globals__}


def _declared_implementation(value):
    """Code identity without traversing state that an explicit provider owns."""
    if isinstance(value, types.FunctionType):
        return [_name(value), _code(value.__code__)]
    cls = value if isinstance(value, type) else type(value)
    implementation = []
    for base in cls.__mro__:
        if base is object:
            continue
        try:
            source = inspect.getsource(base)
        except (OSError, TypeError):
            source = None
        methods = {}
        for name, method in vars(base).items():
            if isinstance(method, (staticmethod, classmethod)):
                method = method.__func__
            if isinstance(method, types.FunctionType):
                methods[name] = _code(method.__code__)
        implementation.append([_name(base), source, methods])
    return implementation


class _Encoder:
    def __init__(self):
        self.active = set()

    def encode(self, value):
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, (float, complex)):
            return [type(value).__name__, repr(value)]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        if isinstance(value, Path):
            return ["path", str(value)]
        if isinstance(value, np.dtype):
            return ["dtype", str(value)]
        if isinstance(value, types.ModuleType):
            return ["module", value.__name__, getattr(value, '__version__', None)]
        if id(value) in self.active:
            return ["recursive", _name(type(value))]
        self.active.add(id(value))
        try:
            return self._encode(value)
        finally:
            self.active.remove(id(value))

    def _encode(self, value):
        provider = getattr(value, 'sampling_identity', None)
        if callable(provider):
            implementation = self.encode(_declared_implementation(value))
            if isinstance(value, type):
                return ["declared_type", implementation]
            return ["declared_state", implementation, self.encode(provider())]
        if isinstance(value, pd.DataFrame):
            return ["dataframe", self.encode(value.index.tolist()),
                    self.encode(value.columns.tolist()),
                    [self.encode(value[col].to_numpy()) for col in value.columns]]
        if isinstance(value, pd.Series):
            return ["series", self.encode(value.name), self.encode(value.index.tolist()),
                    self.encode(value.to_numpy())]
        if isinstance(value, (np.ndarray, np.generic)) or (
                hasattr(value, '__array__') and hasattr(value, 'dtype')):
            array = np.asarray(value)
            if array.dtype.hasobject:
                content = self.encode(array.tolist())
            else:
                content = hashlib.sha256(array.tobytes(order='C')).hexdigest()
            return ["array", str(array.dtype), array.shape, content]
        if isinstance(value, Mapping):
            entries = [[self.encode(k), self.encode(v)] for k, v in value.items()]
            return ["mapping", sorted(entries, key=lambda kv: json.dumps(kv[0], sort_keys=True))]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [self.encode(v) for v in value]]
        if isinstance(value, (set, frozenset)):
            return ["set", sorted((self.encode(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))]
        if isinstance(value, functools.partial):
            return ["partial", self.encode(value.func), self.encode(value.args), self.encode(value.keywords)]
        if isinstance(value, types.MethodType):
            return ["method", self.encode(value.__func__), self.encode(value.__self__)]
        if isinstance(value, types.FunctionType):
            closure = inspect.getclosurevars(value)
            # Library globals contain runtime caches; their installed version
            # and callable name identify the implementation. Still hash captured
            # arguments (e.g. a NumPyro move factory's tuning parameter).
            library = value.__module__.split('.')[0] in {'numpy', 'scipy', 'jax', 'jaxlib', 'numpyro'}
            return ["function", _name(value), self.encode(_code(value.__code__)),
                    self.encode(value.__defaults__), self.encode(value.__kwdefaults__),
                    self.encode(closure.nonlocals),
                    None if library else self.encode(_function_globals(value))]
        if isinstance(value, type):
            library = value.__module__.split('.')[0] in {'builtins', 'numpy', 'scipy', 'jax', 'jaxlib', 'numpyro'}
            if library:
                return ["type", _name(value)]
            if value.__module__.startswith('jeanspy.'):
                # The complete package source is included by software_identity.
                state = {k: v for k, v in vars(value).items()
                         if not k.startswith('__') and isinstance(v, (str, bool, int, float, list, tuple, dict, type))}
                return ["type", _name(value), self.encode(state)]
            members = {k: v for k, v in vars(value).items()
                       if (not k.startswith('__') or k == '__call__')
                       and not isinstance(v, (property, types.MemberDescriptorType))}
            try:
                source = inspect.getsource(value)
            except (OSError, TypeError):
                source = None
            return ["type", _name(value), source, self.encode(value.__bases__), self.encode(members)]
        if isinstance(value, (staticmethod, classmethod)):
            return self.encode(value.__func__)
        if isinstance(value, types.BuiltinFunctionType):
            return ["builtin", _name(value)]
        if isinstance(value, types.CodeType):
            return self.encode(_code(value))
        if value is Ellipsis:
            return ["ellipsis"]
        if is_dataclass(value):
            return [self.encode(type(value)), self.encode({f.name: getattr(value, f.name) for f in fields(value)})]
        module = type(value).__module__.split('.')[0]
        if module == 'scipy' and hasattr(value, 'name'):
            return ["scipy_distribution", _name(type(value)), value.name]
        if callable(value) and module in {'jax', 'jaxlib', 'numpy'} and hasattr(value, '__name__'):
            wrapped = getattr(value, '__wrapped__', None)
            origin = getattr(value, '__module__', '').split('.')[0]
            if wrapped is not None and origin not in {'jax', 'jaxlib', 'numpy'}:
                return ["wrapped_callable", self.encode(wrapped)]
            return ["library_callable", getattr(value, '__module__', module), value.__name__]
        if hasattr(value, '__dict__'):
            return [self.encode(type(value)), self.encode(vars(value))]
        raise TypeError(f"Cannot fingerprint {type(value).__name__}; expose its relevant state "
                        "with a sampling_identity() method before persisting inference")


def fingerprint(value):
    payload = json.dumps(_Encoder().encode(value), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def software_identity(*packages):
    """Include source contents: editable checkouts may share a version number."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*root.rglob('*.py'), *root.glob('data/*.csv')]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    versions = {}
    for package in ('numpy', 'scipy', 'pandas', *packages):
        versions[package] = importlib.metadata.version(package)
    return {'format': 1, 'jeanspy_source': digest.hexdigest(),
            'python': tuple(sys.version_info[:2]), 'dependencies': versions}
