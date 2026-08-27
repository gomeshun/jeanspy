"""Shared parameter and model infrastructure for the classical backend."""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import MutableMapping
from copy import deepcopy
from logging import Formatter, StreamHandler, getLogger
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import pandas as pd


logger = getLogger("jeanspy.model")
_handler = StreamHandler()
_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
if not logger.hasHandlers():
    logger.addHandler(_handler)
logger.setLevel("INFO")


class Parameters(MutableMapping):
    """
    Lightweight substitute for `pd.Series` used in Model.params.
    * Dot access           : p.re_pc
    * Dict compatibility    : p['re_pc']
    * update method support : p.update({...} or pd.Series)
    * Conversion with pandas: p.to_series()
    """

    __slots__ = ("_data",)

    def __init__(self,
                 data: Optional[Mapping[str, Any]] = None,
                 **kw: Any) -> None:
        object.__setattr__(self, "_data", dict())
        if data is not None:
            if isinstance(data, pd.Series):
                object.__getattribute__(self, "_data").update(data.to_dict())
            elif isinstance(data, Parameters):
                object.__getattribute__(self, "_data").update(data._data)
            else:
                object.__getattribute__(self, "_data").update(data)
        if kw:
            object.__getattribute__(self, "_data").update(kw)

    def __getitem__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        return data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        data = object.__getattribute__(self, "_data")
        data[key] = value

    def __delitem__(self, key: str) -> None:
        data = object.__getattribute__(self, "_data")
        del data[key]

    def __iter__(self) -> Iterator[str]:
        data = object.__getattribute__(self, "_data")
        return iter(data)

    def __len__(self) -> int:
        data = object.__getattribute__(self, "_data")
        return len(data)

    def __repr__(self) -> str:
        data = object.__getattribute__(self, "_data")
        kv = ", ".join(f"{k}={v!r}" for k, v in data.items())
        return f"Parameters({kv})"

    def __getattr__(self, name: str) -> Any:
        try:
            # Use object.__getattribute__ to avoid triggering __getattr__ recursively
            data = object.__getattribute__(self, "_data")
            return data[name]
        except (KeyError, AttributeError) as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        # _data is in __slots__, so set it directly
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            # Use object.__getattribute__ to avoid triggering __getattr__
            try:
                data = object.__getattribute__(self, "_data")
                data[name] = value
            except AttributeError:
                # _data hasn't been initialized yet
                object.__setattr__(self, "_data", {name: value})

    def __getstate__(self):
        return {"_data": self._data}

    def __setstate__(self, state):
        self._data = state["_data"]

    def update(self, other: Mapping[str, Any] | "Parameters" | pd.Series,
               **kw: Any) -> None:
        """
        Accepts dict, Parameters, or pd.Series as input.
        Does not return a value (mimics pandas' surface API).
        """
        data = object.__getattribute__(self, "_data")
        if isinstance(other, pd.Series):
            data.update(other.to_dict())
        elif isinstance(other, Parameters):
            data.update(other._data)
        else:
            data.update(dict(other))
        if kw:
            data.update(kw)

    def to_series(self) -> pd.Series:
        data = object.__getattribute__(self, "_data")
        return pd.Series(data, name="params")

    @property
    def index(self):
        data = object.__getattribute__(self, "_data")
        return list(data.keys())

    @property
    def values(self):
        data = object.__getattribute__(self, "_data")
        return list(data.values())

    def copy(self) -> "Parameters":
        """
        Return a shallow copy of the Parameters object.
        """
        data = object.__getattribute__(self, "_data")
        return Parameters(data)

    def __deepcopy__(self, memo):
        """
        Return a deep copy of the Parameters object.
        """
        cls = type(self)
        copied = cls.__new__(cls)
        memo[id(self)] = copied
        data = object.__getattribute__(self, "_data")
        object.__setattr__(copied, "_data", deepcopy(data, memo=memo))
        return copied

class Model(metaclass=ABCMeta):
    '''base class of model objects.

    attributes:
        name: str, name of the model
        params: pd.Series, parameters of the model
        required_param_names: list of str, required parameters' names
        required_models: dict of {name: model_class}, required submodels' names and classes
        submodels: dict of {name: model_object}, submodels' names and objects


    methods:
        __init__(show_init=False, submodels=None, **params):
            Load parameters and check if all required parameters are given.
            Load submodels and check if all required models are given.
            Set model name as a combination submodels' names.
        __repr__():
            show model name and parameters.
        params_all():
            show all parameters as a pd.Series.
        required_param_names_combined():
            show required parameters' name recursively.
        is_required_param_names(param_names_candidates):
            check if param_names_candidates are in self.required_param_names
        update(new_params_dict=None,target='all',**kwargs):
            update model parameters recurrently.


    Note: "self.params", "self.required_models" and "self.required_param_names" are undefined.
        They must be defined in child class.

        self.required_models is a dict of {name: model_class}
        self.required_param_names is a list of str
    '''


    def __init__(self,show_init=False,submodels=None,**params):
        """
        Load parameters and check if all required parameters are given.
        Load submodels and check if all required models are given.
        Set model name as a combination submodels' names.

        Parameters
        ----------
        show_init: bool, if True, show parameters after initialization.
        submodels: dict of {name: model_object}, submodels' names and objects
        params: dict of {name: value}, parameters' names and values
        """
        self.name = self.__class__.__name__
        self.logger = logger.getChild(self.name)
        if submodels is None:
            submodels = {}

        # check if the model has "required_param_names" and "params" attributes.
        if not hasattr(self,'required_param_names'):
            raise AttributeError(self.name+' has no attribute "required_param_names"')

        # check if the model has "required_models" attribute.
        if not hasattr(self,'required_models'):
            raise AttributeError(self.name+' has no attribute "required_models"')
        # check if all required models are given.
        if set(self.required_models.keys()) != set(submodels.keys()):
            raise ValueError(self.name+' has the models: '+str(self.required_models.keys())+" but input is "+str(submodels.keys()))
        else:
            # load submodels
            self.submodels = pd.Series(submodels)

        # initialize parameters of this model
        self.params = Parameters({ p:np.nan for p in self.required_param_names})
        self._parammap: Dict[str, "Model"] = {}
        self._build_parammap()
        self.update(params,target="all")

        # set model name
        if len(self.submodels) > 0:
            self.name += "_" + '+'.join((model.name for model in self.submodels.values))

        # check the consistency of params_all and required_param_names_combined
        params_all_index = self.params_all.index  #[ pname.split(":")[-1] for pname in self.params_all.index]
        required_param_names_combined = self.required_param_names_combined
        if not np.all(params_all_index == required_param_names_combined):
            raise ValueError("params_all and required_param_names_combined are inconsistent: "+str(params_all_index)+" vs "+str(required_param_names_combined))

        if show_init:
            self.logger.info("initialized:\n%s", self)


    def _as_dataframe(self):                           # ← 上書き
        """ Convert the model parameters to a DataFrame for better readability.
        """
        # 1) (<path>, <param>) → value の辞書を作る --------------------
        tuples = []
        values = []
        for full_key, val in self.params_all_with_model_name.items():
            # full_key = "StellarModel:re_pc" など
            if ":" in full_key:
                path, param = full_key.split(":", 1)
            else:
                path, param = self.__class__.__name__, full_key
            tuples.append((path, param))
            values.append(val)

        # 2) MultiIndex DataFrame に変換 ------------------------------
        idx = pd.MultiIndex.from_tuples(tuples, names=["model", "param"])
        df = pd.DataFrame({"value": values}, index=idx)

        # 3) 見やすさ調整（Optional）----------------------------------
        with pd.option_context("display.max_rows", None,
                            "display.max_colwidth", 20,
                            "display.precision", 6):
            return df


    def __repr__(self):
        return self._as_dataframe().to_string()  # __str__() を上書きしているので、print() で表示される


    def __str__(self):
        return self._as_dataframe().to_string()


    def __getitem__(self,key):
        """ syntax sugar for self.submodels[key] """
        return self.submodels[key]


    def _repr_html_(self):
        return self._as_dataframe().to_html()  # __str__() を上書きしているので、print() で表示される


    def _build_parammap(self):
        """ build a map of parameters and models.
        """
        for p in self.required_param_names:
            self._parammap[p] = self
        for mdl in self.submodels.values:
            mdl._build_parammap()
            self._parammap.update(mdl._parammap)


    @property
    def params_all(self):
        """ show all parameters as a pd.Series.
        """
        merged = Parameters(self.params)
        for mdl in self.submodels.values:
            merged.update(mdl.params_all)
        return merged

    @property
    def params_all_with_model_name(self):
        """ show all parameters as a pd.Series.
        """
        merged = Parameters()
        merged.update(self.params)        # 自分はそのまま

        for name, mdl in self.submodels.items():
            tmp = Parameters({f"{name}:{k}": v for k, v in mdl.params_all_with_model_name.items()})
            merged.update(tmp)

        return merged



    @property
    def required_param_names_combined(self):
        """ show all required parameters' name recursively.
        """
        # load required parameters' name of this model
        ret = self.required_param_names[:] # need copy because we must keep self.required_param_names
        if len(self.submodels) > 0: # if there are submodels
            # add submodels' required parameters' name
            [ ret.extend(model.required_param_names_combined) for model in self.submodels.values ]
        return ret


    def is_required_param_names(self,param_names_candidates):
        return [ (p in self.required_param_names) for p in param_names_candidates ]


    def update(self,
            new_params=None,
            target: str = "all",  # No target means all, just for compatibility with other models
            **kwargs):
        merged: Dict[str, Any] = {}

        if new_params is not None:
            if isinstance(new_params, Parameters):
                merged.update(new_params._data)
            elif isinstance(new_params, pd.Series):
                merged.update(new_params.to_dict())
            else:
                merged.update(dict(new_params))
        if kwargs:
            merged.update(kwargs)
        for key, val in merged.items():
            try:
                owner = self._parammap[key]
            except KeyError:
                raise ValueError(f"Unknown parameter '{key}' for model '{self.name}'.")
            owner.params[key] = val
