"""Prior and inference utilities for the classical NumPy/SciPy backend."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from functools import cached_property
from multiprocessing.shared_memory import SharedMemory
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

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
    """Finite uniform bounds in explicitly named sampling coordinates.

    The DataFrame is the single source of truth for evaluation and sampling.
    A generated template must be filled in before constructing this model.
    """

    required_param_names = []
    required_models = {}

    def __init__(self, config, show_init=False, submodels=None, **params):
        super().__init__(show_init, submodels or {}, **params)
        self.load_config(config)

    def load_config(self, config):
        self.fname_config = (
            os.fspath(config) if isinstance(config, (str, os.PathLike)) else None
        )
        if self.fname_config is not None:
            try:
                data = pd.read_csv(self.fname_config, index_col=0)
            except FileNotFoundError:
                logger.error("config file '%s' is not found.", config)
                raise
        else:
            data = config
        self.validate_config(data)
        self.data = data.copy(deep=True)

    @staticmethod
    def validate_config(data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Prior config must be a DataFrame or CSV path.")
        if data.empty or not data.index.is_unique or not data.columns.is_unique:
            raise ValueError("Prior config must have nonempty, unique parameter names and columns.")
        if any(not isinstance(name, str) or not name for name in data.index):
            raise ValueError("Prior parameter names must be nonempty strings.")
        if not {"lower", "upper"}.issubset(data.columns):
            raise ValueError("Prior config needs lower and upper columns.")
        bounds = data[["lower", "upper"]].to_numpy(dtype=float)
        valid = np.isfinite(bounds).all(axis=1) & (bounds[:, 0] < bounds[:, 1])
        if not valid.all():
            raise ValueError(
                "Supply explicit finite prior bounds with lower < upper for: "
                + ", ".join(data.index[~valid])
                + ". Fill in the prior template before inference."
            )

    @property
    def lower(self):
        return self.data["lower"].to_numpy(dtype=float, copy=True)

    @property
    def upper(self):
        return self.data["upper"].to_numpy(dtype=float, copy=True)

    def get_index(self, param_name):
        return self.data.index.get_loc(param_name)

    def extract_value_by_name(self, params, name):
        if np.shape(params) != (len(self.data),):
            raise ValueError(f"Parameters must have shape ({len(self.data)},).")
        return params[self.get_index(name)]

    def sample(self, size=None):
        self.validate_config(self.data)
        size = (size,) if isinstance(size, int) else size
        size = size + (len(self.lower),) if isinstance(size, tuple) else size
        try:
            return np.random.uniform(self.lower, self.upper, size=size)
        except OverflowError as exc:
            message = f"OverflowError: lower:{self.lower}, upper:{self.upper}, size:{size}"
            exc.args = (message,) + exc.args
            raise

    def _lnprior(self, p):
        self.validate_config(self.data)
        if np.shape(p) != (len(self.data),):
            raise ValueError(f"Parameters must have shape ({len(self.data)},).")
        lower = self.lower
        upper = self.upper
        return 0.0 if np.all((lower <= p) & (p <= upper)) else -np.inf

    @staticmethod
    def generate_default_config_file(fname, param_names, lower=np.nan, upper=np.nan):
        """Write a template; unspecified bounds deliberately cannot be sampled."""
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
        self.loc, self.scale = loc, scale
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
        raise AttributeError(key)

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

    def __init__(self, *args, vmem_prior_from_data=False, **kwargs):
        self.vmem_prior_from_data = vmem_prior_from_data
        super().__init__(*args, **kwargs)
        self._validate_prior_schema()

    def _validate_prior_schema(self):
        prior = self["FlatPriorModel"]
        prior.validate_config(prior.data)
        names = self.p_names_lnprob
        physical = [name[6:] if name.startswith(("log10_", "bfunc_")) else name for name in names]
        if physical != self.required_param_names_combined:
            raise ValueError(
                "Prior names/order must match model parameters exactly after removing "
                f"log10_ or bfunc_: expected {self.required_param_names_combined}, got {names}."
            )
        if "log10_re_pc" not in names:
            raise ValueError("The photometry prior requires the sampling coordinate log10_re_pc.")
        photometry = self["PhotometryPriorModel"]
        if not np.isfinite(photometry.loc) or not np.isfinite(photometry.scale) or photometry.scale <= 0:
            raise ValueError("Photometry prior needs a finite location and positive finite scale.")

    @property
    def p_names_lnprob(self):
        return self["FlatPriorModel"].data.index.tolist()

    def convert_params(self, p):
        self._validate_prior_schema()
        p_names = self.p_names_lnprob
        param_names = self.required_param_names_combined
        if np.shape(p) != (len(p_names),):
            raise ValueError(f"Parameters must have shape ({len(p_names)},) in prior config order.")

        def convert_param(name, value):
            if name.startswith("log10_"):
                return 10.0**value
            if name.startswith("bfunc_"):
                return 1 - 10.0**value
            return value

        return pd.Series(
            {
                param_name: convert_param(p_name, value)
                for p_name, param_name, value in zip(p_names, param_names, p)
            }
        )

    def load_data(self, data, shared=False):
        """Load explicitly supplied observed kinematic data."""
        # Reject schema mistakes before allocating shared observation buffers.
        self._validate_prior_schema()
        previous_shared = getattr(self, "shared", False)
        self.shared = shared
        try:
            self.reset_data(data)
        except Exception:
            self.shared = previous_shared
            raise

    def reset_data(self, data):
        """Replace observations while sampler workers are idle.

        Shared models keep their buffer shape so existing readers stay attached.
        Construct a new model to use a different number of observations.
        Explicit velocity-prior bounds are preserved unless the model was
        constructed with vmem_prior_from_data=True (an empirical-prior choice).
        """
        # Validate optional empirical bounds before committing a data update.
        prior = self["FlatPriorModel"]
        updated_prior = prior.data.copy(deep=True)
        if self.vmem_prior_from_data:
            if "vmem_kms" not in updated_prior.index:
                raise ValueError("Data-derived velocity bounds require the vmem_kms coordinate.")
            velocities = np.asarray(data["vlos_kms"], dtype=self.dtype)
            updated_prior.loc["vmem_kms", ["lower", "upper"]] = [
                velocities.min(), velocities.max()
            ]
            prior.validate_config(updated_prior)
        self.data = data
        if self.vmem_prior_from_data:
            prior.data = updated_prior
        self.__dict__.pop("inverse_temparature", None)

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
        fields = ("R_pc", "vlos_kms", "e_vlos_kms")
        shape = data["R_pc"].shape
        if self.shared and hasattr(self, "shared_shape"):
            if shape != self.shared_shape:
                raise ValueError(
                    "Cannot resize shared kinematic data; construct a new model "
                    "for a different number of observations."
                )
        data = data.astype(self.dtype)
        values = {field: data[field].values for field in fields}
        if any(array.shape != shape for array in values.values()):
            raise ValueError("Kinematic columns must have matching shapes.")
        if len(shape) != 1 or not len(data) or not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError("Kinematic data must contain nonempty finite one-dimensional columns.")
        if np.any(values["R_pc"] <= 0) or np.any(values["e_vlos_kms"] < 0):
            raise ValueError("Kinematic data require R_pc > 0 and e_vlos_kms >= 0.")
        if not self.shared:
            self._data = DotDict(values)
            self._n_data = len(data)
            return

        buffer_size = values["R_pc"].nbytes
        handles = {}
        arrays = {}
        opened = []
        created = []
        try:
            # Validate every segment before changing any observations or metadata.
            for field in fields:
                shm_name = self.shared_memory_basename + "_" + field
                shm = getattr(self, f"shm_{field}", None)
                if shm is None:
                    try:
                        shm = SharedMemory(name=shm_name, create=True, size=buffer_size)
                        created.append(shm)
                    except FileExistsError:
                        shm = SharedMemory(name=shm_name, create=False)
                    opened.append(shm)
                if shm.size != buffer_size:
                    raise ValueError(
                        f"Shared memory {shm.name!r} has size {shm.size} bytes; "
                        f"expected {buffer_size} bytes for {field}."
                    )
                handles[field] = shm
                arrays[field] = np.ndarray(shape, dtype=self.dtype, buffer=shm.buf)
        except Exception:
            arrays.clear()
            for shm in opened:
                shm.close()
            for shm in created:
                shm.unlink()
            raise

        for field in fields:
            arrays[field][:] = values[field]
            setattr(self, f"shm_{field}", handles[field])
        self._n_data = len(data)
        self.shared_shape = shape
        self.buffer_size = buffer_size

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
        self._validate_prior_schema()
        p = self["FlatPriorModel"].sample(size)
        idx_log10_re_pc = self["FlatPriorModel"].get_index("log10_re_pc")
        prior = self["FlatPriorModel"]
        photometry = self["PhotometryPriorModel"]
        loc, scale = photometry.loc, photometry.scale
        # Draw from the product of the Gaussian photometry prior and finite
        # uniform support, so generated walkers always satisfy both priors.
        a = (prior.lower[idx_log10_re_pc] - loc) / scale
        b = (prior.upper[idx_log10_re_pc] - loc) / scale
        p[..., idx_log10_re_pc] = truncnorm.rvs(a, b, loc=loc, scale=scale, size=size)
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
    *,
    vmem_prior_from_data=False,
):
    """Compose Plummer + NFW + constant anisotropy with explicit finite priors.

    ``config`` is a DataFrame or CSV in this order: vmem_kms, log10_re_pc,
    log10_rs_pc, log10_rhos_Msunpc3, log10_r_t_pc, bfunc_beta_ani.
    A missing CSV is created as an unfilled template, then raises ValueError.
    Caller velocity bounds are preserved unless vmem_prior_from_data is True.
    """
    dsph_model = DSphModel(
        submodels={
            "StellarModel": PlummerModel(),
            "DMModel": NFWModel(),
            "AnisotropyModel": ConstantAnisotropyModel(),
        }
    )

    names = ["vmem_kms", "log10_re_pc", "log10_rs_pc", "log10_rhos_Msunpc3",
             "log10_r_t_pc", "bfunc_beta_ani"]
    if isinstance(config, (str, os.PathLike)) and not os.path.exists(config):
        FlatPriorModel.generate_default_config_file(
            config,
            names,
        )
        raise ValueError(f"Created prior template at {config}; supply explicit finite prior bounds before inference.")

    prior = FlatPriorModel(config=config)
    if prior.data.index.tolist() != names:
        raise ValueError(f"Default model prior names/order must be {names}.")

    return SimpleDSphEstimationModel(
        args_load_data=[data],
        vmem_prior_from_data=vmem_prior_from_data,
        submodels={
            "DSphModel": dsph_model,
            "FlatPriorModel": prior,
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
