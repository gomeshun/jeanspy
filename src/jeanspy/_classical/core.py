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
    """Lightweight mapping used for stateful model parameters.

    The container preserves the small subset of the historical ``pandas.Series``
    surface used by JeansPy while supporting attribute access and predictable
    shallow/deep copy semantics.
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
        if isinstance(other, pd.Series):
            self._data.update(other.to_dict())
        elif isinstance(other, Parameters):
            self._data.update(other._data)
        else:
            self._data.update(dict(other))
        if kw:
            self._data.update(kw)

    def to_series(self) -> pd.Series:
        return pd.Series(self._data, name="params")

    @property
    def index(self):
        return list(self._data.keys())

    @property
    def values(self):
        return list(self._data.values())

    def copy(self) -> "Parameters":
        return Parameters(self._data)

    def __deepcopy__(self, memo):
        cls = type(self)
        copied = cls.__new__(cls)
        memo[id(self)] = copied
        object.__setattr__(copied, "_data", deepcopy(self._data, memo=memo))
        return copied


class Model(metaclass=ABCMeta):
    """Base class for stateful classical model components."""

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
        merged = Parameters(self.params)
        for model in self.submodels.values:
            merged.update(model.params_all)
        return merged

    @property
    def params_all_with_model_name(self):
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
        result = self.required_param_names[:]
        for model in self.submodels.values:
            result.extend(model.required_param_names_combined)
        return result

    def is_required_param_names(self, param_names_candidates):
        return [p in self.required_param_names for p in param_names_candidates]

    def update(self, new_params=None, target: str = "all", **kwargs):
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
