"""Prior and inference utilities for the classical NumPy/SciPy backend."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import cached_property
from multiprocessing.shared_memory import SharedMemory
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from .core import Model, logger
from .profiles import ConstantAnisotropyModel, NFWModel, PlummerModel
from .solver import DSphModel


class FittableModel(Model, metaclass=ABCMeta):
    """Base class for stateful models that expose likelihood/prior methods."""

    def __init__(self, args_load_data=None, kwargs_load_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("Fittable Model: args_load_data: %r", args_load_data)
        if not isinstance(args_load_data, list):
            raise TypeError("args_load_data must be a list.")
        if kwargs_load_data is None:
            kwargs_load_data = {}
        self.logger.info("Fittable Model: kwargs_load_data: %r", kwargs_load_data)
        if not isinstance(kwargs_load_data, dict):
            raise TypeError("kwargs_load_data must be a dict.")
        self.load_data(*args_load_data, **kwargs_load_data)
        if not hasattr(self, "prior_names"):
            raise AttributeError("FittableModel must have the prior_names attribute.")

    @abstractmethod
    def convert_params(self, p):
        raise NotImplementedError

    @abstractmethod
    def load_data(self, *args, **kwargs):
        raise NotImplementedError

    @cached_property
    def inverse_temparature(self):
        """Return the WBIC inverse temperature ``1/log(N_data)``."""
        n_data = self.n_data if hasattr(self, "n_data") else len(self.data)
        return 1 / np.log(n_data)

    @abstractmethod
    def _lnlikelihoods(self, *args, **kwargs):
        raise NotImplementedError

    def lnlikelihoods(self, p, *args, **kwargs):
        params = self.convert_params(p)
        self.update(params)
        return self._lnlikelihoods(*args, **kwargs)

    def _lnlikelihood(self, *args, **kwargs):
        value = np.sum(self._lnlikelihoods(*args, **kwargs))
        return -np.inf if np.isnan(value) else value

    def lnlikelihood(self, p, *args, **kwargs):
        params = self.convert_params(p)
        self.update(params)
        return self._lnlikelihood(*args, **kwargs)

    @abstractmethod
    def _lnpriors(self, p, *args, **kwargs):
        raise NotImplementedError

    def lnpriors(self, p, *args, **kwargs):
        params = self.convert_params(p)
        self.update(params)
        return self._lnpriors(p, *args, **kwargs)

    @property
    def blobs_dtype(self):
        return [("lnl", float), *((name, float) for name in self.prior_names)]

    def lnposterior(self, p, *args, **kwargs):
        params = self.convert_params(p)
        self.update(params)
        lnl = -np.inf
        lnp_list = self._lnpriors(p, *args, **kwargs)
        if np.all([lnp > -np.inf for lnp in lnp_list]):
            lnl = self._lnlikelihood(*args, **kwargs)
        result = (lnl + np.sum(lnp_list), lnl, *lnp_list)
        if np.isnan(result[0]):
            self.logger.error("lnposterior is nan. lnl:%s, lnp_list:%s", lnl, lnp_list)
            self.logger.error("p:%s", p)
            self.logger.error("args:%s", args)
            self.logger.error("kwargs:%s", kwargs)
            self.logger.error("params:%s", params)
            raise ValueError(
                [
                    f"lnposterior is nan. lnl:{lnl}, lnp_list:{lnp_list}",
                    f"p:{p}",
                    f"args:{args}",
                    f"kwargs:{kwargs}",
                    f"params:{params}",
                ]
            )
        return result

    def lnposterior_wbic(self, p, *args, **kwargs):
        params = self.convert_params(p)
        self.update(params)
        lnl = -np.inf
        lnp_list = self._lnpriors(p, *args, **kwargs)
        if np.all([lnp > -np.inf for lnp in lnp_list]):
            lnl = self._lnlikelihood(*args, **kwargs) * self.inverse_temparature
        result = (lnl + np.sum(lnp_list), lnl, *lnp_list)
        if np.isnan(result[0]):
            raise ValueError(
                "lnposterior_wbic is nan. "
                f"lnl:{lnl}, lnp_list:{lnp_list}\np:{p}\n"
                f"args:{args}\nkwargs:{kwargs}\nparams:{params}"
            )
        return result

    @cached_property
    def ndim(self):
        return len(self.params_all)


class FlatPriorModel(Model):
    """Flat prior over the sampling-coordinate configuration."""

    required_param_names = []
    required_models = {}

    def __init__(self, config, show_init=False, submodels=None, **params):
        super().__init__(show_init, submodels or {}, **params)
        self.load_config(config)

    def load_config(self, config):
        if isinstance(config, str):
            try:
                self.fname_config = config
                self.data = pd.read_csv(config, index_col=0)
            except FileNotFoundError:
                logger.error("config file '%s' is not found.", config)
                raise
        else:
            self.data = config

        self.lower = self.data["lower"].values
        self.upper = self.data["upper"].values

    def get_index(self, param_name):
        return self.data.index.get_loc(param_name)

    def extract_value_by_name(self, params, name):
        assert len(params) == len(
            self.data
        ), f"len(param)={len(params)} != len(self.data)={len(self.data)}"
        return params[self.get_index(name)]

    def sample(self, size=None):
        size = (size,) if isinstance(size, int) else size
        size = size + (len(self.lower),) if isinstance(size, tuple) else size
        try:
            return np.random.uniform(self.lower, self.upper, size=size)
        except OverflowError as exc:
            message = f"OverflowError: lower:{self.lower}, upper:{self.upper}, size:{size}"
            exc.args = (message,) + exc.args
            raise

    def _lnprior(self, p):
        lower = self.data["lower"].values
        upper = self.data["upper"].values
        return 0.0 if np.all((lower <= p) & (p <= upper)) else -np.inf

    @staticmethod
    def generate_default_config_file(fname, param_names, lower=-np.inf, upper=np.inf):
        df = pd.DataFrame({"lower": lower, "upper": upper}, index=param_names)
        df.to_csv(fname)
        logger.info("generated %s.", fname)
        return df


class PhotometryPriorModel(Model):
    """Gaussian prior for ``log10(re_pc)``."""

    required_param_names = []
    required_models = {}

    def __init__(self, loc, scale, show_init=False, submodels=None, **params):
        super().__init__(show_init, submodels or {}, **params)
        self.logger.info(
            "%s:%r",
            self.__class__.__name__,
            {"log10_re_pc": loc, "e_log10_re_pc": scale},
        )
        self.reset_prior(loc, scale)

    def reset_prior(self, loc, scale):
        self._lnprior_func = norm(loc=loc, scale=scale).logpdf
        self._sample = norm(loc=loc, scale=scale).rvs

    def _lnprior(self, log10_re_pc):
        return self._lnprior_func(log10_re_pc)

    def sample(self, size):
        return self._sample(size=size)


class DotDict(dict):
    """Dictionary with attribute access retained for historical data access."""

    def __getattr__(self, key):
        if key in self:
            return self[key]
        return super().__getattr__(key)

    def __setattr__(self, key, value):
        if key in self:
            self[key] = value
        else:
            super().__setattr__(key, value)

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            super().__delattr__(key)


class SimpleDSphEstimationModel(FittableModel, Model):
    """Kinematics-only classical dwarf-spheroidal estimation model."""

    required_param_names = []
    required_models = {
        "DSphModel": DSphModel,
        "FlatPriorModel": FlatPriorModel,
        "PhotometryPriorModel": PhotometryPriorModel,
    }
    dtype = np.float32
    prior_names = ["flat_prior", "photometry_prior"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fname_config = self["FlatPriorModel"].fname_config
        self.logger.info(
            "%s: Please check the consistency of model parameters and config file: %s.",
            self.__class__,
            fname_config,
        )
        comparison = {
            "config": self.p_names_lnprob,
            "params": self.required_param_names_combined,
        }
        try:
            self.logger.info("%s", pd.DataFrame(comparison))
            consistencies = [
                param in p
                for p, param in zip(comparison["config"], comparison["params"])
            ]
            assert all(consistencies)
        except ValueError:
            self.logger.error("%r", comparison)
            raise
        except AssertionError:
            self.logger.error("ERROR: config and params are not consistent.")
            self.logger.error("config file: %s", fname_config)
            self.logger.error("%r", comparison)
            self.logger.error("%r", consistencies)
            raise

    @property
    def p_names_lnprob(self):
        return self["FlatPriorModel"].data.index.tolist()

    def convert_params(self, p):
        p_names = self.p_names_lnprob
        param_names = self.required_param_names_combined

        def convert_param(name, value):
            if "log10_" in name:
                return 10**value
            if "bfunc_" in name:
                return 1 - 10**value
            return value

        return pd.Series(
            {
                param_name: convert_param(p_name, value)
                for p_name, param_name, value in zip(p_names, param_names, p)
            }
        )

    def load_data(self, data, shared=False):
        """Load explicitly supplied observed kinematic data."""
        self.shared = shared
        self.reset_data(data.astype(self.dtype))

    def reset_data(self, data):
        self.data = data
        lower = self.data["vlos_kms"].min()
        upper = self.data["vlos_kms"].max()
        self["FlatPriorModel"].data.loc["vmem_kms", "lower"] = lower
        self["FlatPriorModel"].data.loc["vmem_kms", "upper"] = upper
        self.logger.info(
            "%s: set vmem_kms prior bounds from data: lower=%.6f upper=%.6f",
            self.__class__.__name__,
            float(lower),
            float(upper),
        )

    @property
    def shared_memory_basename(self):
        if not self.shared:
            return None
        return f"SimpleDSphEstimationModel_{id(self)}"

    @property
    def data(self):
        if not self.shared:
            return self._data

        try:
            return DotDict(
                {
                    "R_pc": np.ndarray(
                        self.shared_shape,
                        dtype=self.dtype,
                        buffer=self.shm_R_pc.buf,
                    ),
                    "vlos_kms": np.ndarray(
                        self.shared_shape,
                        dtype=self.dtype,
                        buffer=self.shm_vlos_kms.buf,
                    ),
                    "e_vlos_kms": np.ndarray(
                        self.shared_shape,
                        dtype=self.dtype,
                        buffer=self.shm_e_vlos_kms.buf,
                    ),
                }
            )
        except (FileNotFoundError, AttributeError):
            self.logger.error(
                "SharedMemory '%s' is not initialized yet.",
                self.shared_memory_basename,
            )
            raise

    @property
    def n_data(self):
        return self._n_data

    @data.setter
    def data(self, data: pd.DataFrame):
        self._n_data = len(data)
        if not self.shared:
            self._data = DotDict(
                {
                    "R_pc": data["R_pc"].values,
                    "vlos_kms": data["vlos_kms"].values,
                    "e_vlos_kms": data["e_vlos_kms"].values,
                }
            )
            return

        self.shared_shape = data["R_pc"].shape
        assert self.shared_shape == data["vlos_kms"].shape
        assert self.shared_shape == data["e_vlos_kms"].shape
        self.buffer_size = data["R_pc"].values.nbytes
        assert self.buffer_size == data["vlos_kms"].values.nbytes
        assert self.buffer_size == data["e_vlos_kms"].values.nbytes

        for field in ("R_pc", "vlos_kms", "e_vlos_kms"):
            shm_name = self.shared_memory_basename + "_" + field
            try:
                shm = SharedMemory(
                    name=shm_name,
                    create=True,
                    size=self.buffer_size,
                )
                array = np.ndarray(
                    self.shared_shape,
                    dtype=self.dtype,
                    buffer=shm.buf,
                )
                array[:] = data[field].values
            except FileExistsError:
                shm = SharedMemory(name=shm_name, create=False)
            setattr(self, f"shm_{field}", shm)

    def _release_shared_memory(self, suffix):
        if not self.shared:
            return
        name = self.shared_memory_basename + suffix
        if not hasattr(self, "shared_shape"):
            raise ValueError(
                f"{self.__class__.__name__}: try to release shared memory "
                f"{name} before initialization."
            )
        try:
            shm = getattr(self, f"shm{suffix}")
            shm.close()
            shm.unlink()
            self.logger.info("shared memory '%s' is released.", name)
        except FileNotFoundError:
            self.logger.info("shared memory '%s' is already released.", name)

    def release_shared_memory(self):
        self._release_shared_memory("_R_pc")
        self._release_shared_memory("_vlos_kms")
        self._release_shared_memory("_e_vlos_kms")

    def _lnlikelihoods(self):
        s2 = self["DSphModel"].sigmalos2_dequad(self.data.R_pc)
        err2 = self.data.e_vlos_kms**2
        vmem_kms = self["DSphModel"].params.vmem_kms
        return norm.logpdf(
            self.data.vlos_kms,
            loc=vmem_kms,
            scale=np.sqrt(s2 + err2),
        )

    def _lnpriors(self, p_before_conversion):
        idx_log10_re_pc = self["FlatPriorModel"].get_index("log10_re_pc")
        log10_re_pc = p_before_conversion[idx_log10_re_pc]
        return [
            self["FlatPriorModel"]._lnprior(p_before_conversion),
            self["PhotometryPriorModel"]._lnprior(log10_re_pc),
        ]

    def sample(self, size=None):
        p = self["FlatPriorModel"].sample(size)
        idx_log10_re_pc = self["FlatPriorModel"].get_index("log10_re_pc")
        p[..., idx_log10_re_pc] = self["PhotometryPriorModel"].sample(size)
        return p

    def sample_data(self, size=None):
        s2 = self["DSphModel"].sigmalos2_dequad(self.data.R_pc)
        err2 = self.data.e_vlos_kms**2
        vmem_kms = self["DSphModel"].params.vmem_kms
        return norm.rvs(
            loc=vmem_kms,
            scale=np.sqrt(s2 + err2),
            size=size,
        )


def get_default_estimation_model(
    data,
    photometry_prior_loc,
    photometry_prior_scale,
    config="priorconfig.csv",
):
    """Return the historical default classical estimation-model composition."""
    dsph_model = DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": ConstantAnisotropyModel(),
        }
    )

    if not os.path.exists(config):
        logger.warning("config file '%s' is not found.", config)
        logger.info("generate a default config file.")
        FlatPriorModel.generate_default_config_file(
            config,
            dsph_model.params_all.index,
        )

    return SimpleDSphEstimationModel(
        args_load_data=[data],
        submodels={
            "DSphModel": DSphModel(
                submodels={
                    "StellarModel": PlummerModel(),
                    "DMModel": NFWModel(),
                    "AnisotropyModel": ConstantAnisotropyModel(),
                }
            ),
            "FlatPriorModel": FlatPriorModel(config=config),
            "PhotometryPriorModel": PhotometryPriorModel(
                loc=photometry_prior_loc,
                scale=photometry_prior_scale,
            ),
        },
    )


__all__ = [
    "DotDict",
    "FittableModel",
    "FlatPriorModel",
    "PhotometryPriorModel",
    "SimpleDSphEstimationModel",
    "get_default_estimation_model",
]
