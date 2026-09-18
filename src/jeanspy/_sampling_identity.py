"""Conservative, deterministic identities for persisted inference targets.

This is an accidental-mismatch guard, not a sandbox for arbitrary Python code.
Custom models that read hidden state (files, services, mutable extension objects)
must expose that state through ``sampling_identity()``. Opaque objects fail
closed instead of being identified by their memory address or repr.
"""

import ast
from bisect import bisect_left
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
import textwrap
import types

import numpy as np
import pandas as pd


def _name(value):
    return f"{value.__module__}.{value.__qualname__}"


IDENTITY_FORMAT = 2


class _WithoutDocstrings(ast.NodeTransformer):
    def _body(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
        return node

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _body


def _source_structure(source):
    """Canonical Python syntax, excluding comments, layout and docstrings only.

    Executable string constants, defaults, annotations, imports and assertions
    remain. Reflective models must declare any documentation they use as data
    in sampling_identity(), just as they must declare external file contents.
    """
    tree = _WithoutDocstrings().visit(ast.parse(textwrap.dedent(source)))
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _class_structure(cls):
    try:
        return _source_structure(inspect.getsource(cls))
    except (OSError, TypeError):
        return None


def _code(code):
    # Record loaded constants by value, not by their table index. Adding or
    # removing a docstring can renumber constants, even for ``return None``.
    # A string actually returned by code remains part of the identity, including
    # RETURN_CONST on Python 3.12. Never assume constant zero is documentation.
    instructions = [i for i in dis.get_instructions(code)
                    if i.opname not in {'NOP', 'EXTENDED_ARG'}]
    offsets = [i.offset for i in instructions]
    operations = []
    for instruction in instructions:
        if instruction.opcode in dis.hasconst:
            value = instruction.argval
            argument = _code(value) if isinstance(value, types.CodeType) else value
        elif instruction.opcode in dis.hasjabs or instruction.opcode in dis.hasjrel:
            argument = bisect_left(offsets, instruction.argval)
        else:
            argument = instruction.arg
        operations.append([instruction.opname, argument])
    exceptions = [[bisect_left(offsets, entry.start), bisect_left(offsets, entry.end),
                   bisect_left(offsets, entry.target), entry.depth, entry.lasti]
                  for entry in dis.Bytecode(code).exception_entries]
    return [operations, code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_flags & ~getattr(inspect, 'CO_HAS_DOCSTRING', 0),
            exceptions]


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
        return [_name(value), _code(value.__code__), value.__defaults__, value.__kwdefaults__]
    cls = value if isinstance(value, type) else type(value)
    implementation = []
    for base in cls.__mro__:
        if base is object:
            continue
        source = _class_structure(base)
        methods = {}
        for name, method in vars(base).items():
            if isinstance(method, (staticmethod, classmethod)):
                method = method.__func__
            if isinstance(method, types.FunctionType):
                methods[name] = [_code(method.__code__), method.__defaults__, method.__kwdefaults__]
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
                # Package computational syntax is included by software_identity.
                state = {k: v for k, v in vars(value).items()
                         if not k.startswith('__') and isinstance(v, (str, bool, int, float, list, tuple, dict, type))}
                return ["type", _name(value), self.encode(state)]
            members = {k: v for k, v in vars(value).items()
                       if (not k.startswith('__') or k == '__call__')
                       and not isinstance(v, (property, types.MemberDescriptorType))}
            source = _class_structure(value)
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


def _source_manifest(*, computational):
    root = Path(__file__).parent
    paths = {*root.rglob('*.py'), *(p for p in (root / 'data').rglob('*') if p.is_file())}
    manifest = {}
    for path in sorted(paths):
        content = path.read_bytes()
        if computational and path.suffix == '.py':
            content = _source_structure(content.decode('utf-8')).encode('utf-8')
        manifest[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
    return manifest


def source_provenance():
    """Record original source/data bytes separately from resume compatibility."""
    manifest = _source_manifest(computational=False)
    return {'source_sha256': fingerprint(manifest), 'files': manifest}


def software_identity(*packages):
    """Recheck computational syntax, data and dependencies at every boundary.

    Do not cache across calls: an editable checkout or installed dependency can
    change between sampler runs within the same process. Format 2 cannot resume
    format-1 chains; full source bytes are retained separately as provenance.
    """
    versions = {}
    for package in ('numpy', 'scipy', 'pandas', *packages):
        versions[package] = importlib.metadata.version(package)
    return {'format': IDENTITY_FORMAT,
            'jeanspy_computation': fingerprint(_source_manifest(computational=True)),
            'python': tuple(sys.version_info[:3]), 'dependencies': versions}
