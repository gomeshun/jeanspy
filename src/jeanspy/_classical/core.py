"""Shared building blocks for the classical NumPy/SciPy backend."""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import MutableMapping
from copy import deepcopy
import logging
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger("jeanspy.model")


class Parameters(MutableMapping):
    r"""Lightweight mapping used for stateful model parameters.

    The container preserves the small subset of the historical ``pandas.Series``
    surface used by JeansPy while supporting attribute access and predictable
    shallow/deep copy semantics.

    Notes
    -----
    **Inputs and units.** An optional mapping plus keyword values; keys are
    strings and units belong to the stored values.

    **Returns and shape.** Parameters.copy is shallow; deepcopy separates nested
    values. Parameters.index and .values are lists, not NumPy arrays or dict
    methods; ``to_series`` returns a pandas Series. DotDict follows dict
    operations; assignment to an existing key through an attribute changes the
    key.

    **Validity.** Container operations have no physical validation. DotDict new
    attributes need not become keys.

    **Errors.** Missing mapping keys raise KeyError; missing attributes raise
    AttributeError.

    **Backend.** Python host-side containers.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** Parameters({'``re_pc``': 300.}).``to_series``().
    """

    __slots__ = ("_data",)

    def __init__(
        self,
        data: Optional[Mapping[str, Any]] = None,
        **kw: Any,
    ) -> None:
        object.__setattr__(self, "_data", {})
        if data is not None:
            if isinstance(data, pd.Series):
                self._data.update(data.to_dict())
            elif isinstance(data, Parameters):
                self._data.update(data._data)
            else:
                self._data.update(data)
        if kw:
            self._data.update(kw)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        kv = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"Parameters({kv})"

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except (KeyError, AttributeError) as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            try:
                self._data[name] = value
            except AttributeError:
                object.__setattr__(self, "_data", {name: value})

    def __getstate__(self):
        return {"_data": self._data}

    def __setstate__(self, state):
        self._data = state["_data"]

    def update(
        self,
        other: Mapping[str, Any] | "Parameters" | pd.Series,
        **kw: Any,
    ) -> None:
        """Merge a mapping, Parameters or Series, then keyword replacements.

        Returns None. Values retain their units and are not deep-copied or
        physically validated; later entries replace existing keys.
        """
        if isinstance(other, pd.Series):
            self._data.update(other.to_dict())
        elif isinstance(other, Parameters):
            self._data.update(other._data)
        else:
            self._data.update(dict(other))
        if kw:
            self._data.update(kw)

    def to_series(self) -> pd.Series:
        r"""Convert parameter storage to a pandas Series.

        Notes
        -----
        **Inputs and units.** No arguments.

        **Returns and shape.** Series named params, preserving key order; values
        keep their original units.
        """
        return pd.Series(self._data, name="params")

    @property
    def index(self):
        """Return parameter names as an insertion-ordered list of strings."""
        return list(self._data.keys())

    @property
    def values(self):
        """Return stored values as an insertion-ordered list, retaining their units."""
        return list(self._data.values())

    def copy(self) -> "Parameters":
        r"""Make a shallow parameter copy.

        Notes
        -----
        **Inputs and units.** No arguments.

        **Returns and shape.** New Parameters mapping; nested mutable values remain
        shared. Use copy.deepcopy for them.
        """
        return Parameters(self._data)

    def __deepcopy__(self, memo):
        cls = type(self)
        copied = cls.__new__(cls)
        memo[id(self)] = copied
        object.__setattr__(copied, "_data", deepcopy(self._data, memo=memo))
        return copied


class Model(metaclass=ABCMeta):
    r"""Base class for stateful classical model components.

    Notes
    -----
    **Inputs and units.** ``show_init`` is a logging flag; submodels maps
    required role names to component instances; \*\*params sets scalar physical
    parameters. Subclasses declare required names and roles.

    **Returns and shape.** A mutable model; model[role] returns its submodel.
    update(``new_params``, \*\*kwargs) mutates the owning components and returns
    None. ``params_all`` is a flattened Parameters copy;
    ``params_all_with_model_name`` retains role-qualified names.

    **Validity.** This is a subclassing interface. Unspecified parameters are
    initialized to NaN; supply all physical values before numerical evaluation.
    target is retained but ignored by update.

    **Errors.** Missing subclass declarations raise AttributeError; mismatched
    roles or unknown parameters raise ValueError.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** See the spherical quickstart and composition guide.
    """

    def __init__(self, show_init=False, submodels=None, **params):
        self.name = self.__class__.__name__
        self.logger = logger.getChild(self.name)
        if submodels is None:
            submodels = {}

        if not hasattr(self, "required_param_names"):
            raise AttributeError(
                self.name + ' has no attribute "required_param_names"'
            )
        if not hasattr(self, "required_models"):
            raise AttributeError(self.name + ' has no attribute "required_models"')

        if set(self.required_models.keys()) != set(submodels.keys()):
            raise ValueError(
                self.name
                + " has the models: "
                + str(self.required_models.keys())
                + " but input is "
                + str(submodels.keys())
            )
        self.submodels = pd.Series(submodels, dtype=object)

        self.params = Parameters({p: np.nan for p in self.required_param_names})
        self._parammap: Dict[str, "Model"] = {}
        self._build_parammap()
        self.update(params, target="all")

        if len(self.submodels) > 0:
            self.name += "_" + "+".join(model.name for model in self.submodels.values)

        if self.params_all.index != self.required_param_names_combined:
            raise ValueError(
                "params_all and required_param_names_combined are inconsistent: "
                f"{self.params_all.index} vs {self.required_param_names_combined}"
            )

        if show_init:
            self.logger.info("initialized:\n%s", self)

    def _as_dataframe(self):
        tuples = []
        values = []
        for full_key, val in self.params_all_with_model_name.items():
            if ":" in full_key:
                path, param = full_key.split(":", 1)
            else:
                path, param = self.__class__.__name__, full_key
            tuples.append((path, param))
            values.append(val)
        idx = pd.MultiIndex.from_tuples(tuples, names=["model", "param"])
        return pd.DataFrame({"value": values}, index=idx)

    def __repr__(self):
        return self._as_dataframe().to_string()

    def __str__(self):
        return self._as_dataframe().to_string()

    def __getitem__(self, key):
        return self.submodels[key]

    def sampling_identity(self, sampled_names=()):
        """Configuration and fixed parameters, excluding changing MCMC coordinates."""
        state = {k: v for k, v in vars(self).items()
                 if k not in {"logger", "_parammap", "params", "submodels"}}
        state["params"] = {k: v for k, v in self.params.items() if k not in sampled_names}
        state["submodels"] = {k: (type(v), v.sampling_identity(sampled_names))
                              for k, v in self.submodels.items()}
        return state

    def _repr_html_(self):
        return self._as_dataframe().to_html()

    def _build_parammap(self):
        for param in self.required_param_names:
            self._parammap[param] = self
        for model in self.submodels.values:
            model._build_parammap()
            self._parammap.update(model._parammap)

    @property
    def params_all(self):
        """Return a flattened Parameters copy of this model and its submodels.

        Values retain their physical units. Later submodels overwrite duplicate
        names; use ``params_all_with_model_name`` to retain role-qualified names.
        """
        merged = Parameters(self.params)
        for model in self.submodels.values:
            merged.update(model.params_all)
        return merged

    @property
    def params_all_with_model_name(self):
        """Return a new Parameters mapping with submodel-role prefixes.

        Nested names use ``role:parameter`` notation. Values retain their physical
        units; this operation copies the mapping, not nested mutable values.
        """
        merged = Parameters()
        merged.update(self.params)
        for name, model in self.submodels.items():
            merged.update(
                Parameters(
                    {
                        f"{name}:{key}": value
                        for key, value in model.params_all_with_model_name.items()
                    }
                )
            )
        return merged

    @property
    def required_param_names_combined(self):
        """Return this model's and all nested submodels' required parameter names.

        The result is a list in traversal order; duplicate names are retained.
        """
        result = self.required_param_names[:]
        for model in self.submodels.values:
            result.extend(model.required_param_names_combined)
        return result

    def is_required_param_names(self, param_names_candidates):
        """Test a sequence of names against this component's required parameters.

        Returns a list of bool with the same length and order as
        ``param_names_candidates``. Submodel requirements are not included.
        """
        return [p in self.required_param_names for p in param_names_candidates]

    def update(self, new_params=None, target: str = "all", **kwargs):
        r"""Replace named parameters in the owning components.

        Notes
        -----
        **Inputs and units.** ``new_params`` is an optional
        mapping/Parameters/Series; keyword values are additional replacements. Names
        are physical names declared by this model and its components; target is
        ignored.

        **Returns and shape.** None; mutates component parameters. ``params_all``
        returns the resulting flattened copy.
        """
        del target  # retained for API compatibility
        merged: Dict[str, Any] = {}
        if new_params is not None:
            if isinstance(new_params, Parameters):
                merged.update(new_params._data)
            elif isinstance(new_params, pd.Series):
                merged.update(new_params.to_dict())
            else:
                merged.update(dict(new_params))
        merged.update(kwargs)

        for key, value in merged.items():
            try:
                owner = self._parammap[key]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown parameter '{key}' for model '{self.name}'."
                ) from exc
            owner.params[key] = value


__all__ = ["Model", "Parameters", "logger"]
