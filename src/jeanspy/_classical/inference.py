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
    """Subclassing interface for stateful likelihoods and prior terms.

    Parameters
    ----------
    args_load_data : list
        Positional arguments forwarded to the concrete load_data method.
    kwargs_load_data : dict or None, optional
        Keyword arguments forwarded to load_data; None means an empty mapping.
    *args, **kwargs
        Model component and physical parameter initialization arguments.

    Raises
    ------
    TypeError
        The data-loading arguments have the wrong container types, or an
        abstract subclass has not implemented the required interface.
    AttributeError
        The initialized concrete model does not declare prior_names.

    Notes
    -----
    Concrete subclasses define observation shapes and units, sampling-vector
    order, conversion to physical parameters, and likelihood/prior terms.
    Calling a target method updates the stateful components. lnposterior
    returns the total log posterior followed by log likelihood and individual
    log priors for emcee blobs. WBIC requires more than one observation.

    This NumPy/SciPy host interface does not support physical-parameter JAX
    tracing. See SimpleDSphEstimationModel for the spherical kinematic target
    and ``examples/docs_inference.py`` for a complete short storage example.
    """

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
        """Map a sampling-coordinate vector p to named physical parameters.

        Subclasses define vector order, transforms and units. This abstract
        interface raises NotImplementedError.
        """
        raise NotImplementedError

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Load observations in a concrete estimation model.

        Subclasses define the accepted arguments and validation. This abstract
        interface raises NotImplementedError.
        """
        raise NotImplementedError

    @cached_property
    def inverse_temparature(self):
        """Return the WBIC inverse temperature ``1/log(N_data)``."""
        n_data = self.n_data if hasattr(self, "n_data") else len(self.data)
        if n_data <= 1:
            raise ValueError("WBIC requires at least two observations")
        return 1 / np.log(n_data)

    @abstractmethod
    def _lnlikelihoods(self, *args, **kwargs):
        raise NotImplementedError

    def lnlikelihoods(self, p, *args, **kwargs):
        r"""Evaluate the explicit kinematic inference target.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Per-star log densities, shape (N,).
        """
        params = self.convert_params(p)
        self.update(params)
        return self._lnlikelihoods(*args, **kwargs)

    def _lnlikelihood(self, *args, **kwargs):
        value = np.sum(self._lnlikelihoods(*args, **kwargs))
        return -np.inf if np.isnan(value) else value

    def lnlikelihood(self, p, *args, **kwargs):
        r"""Evaluate the explicit kinematic inference target.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Scalar sum of log likelihoods.
        """
        params = self.convert_params(p)
        self.update(params)
        return self._lnlikelihood(*args, **kwargs)

    @abstractmethod
    def _lnpriors(self, p, *args, **kwargs):
        raise NotImplementedError

    def lnpriors(self, p, *args, **kwargs):
        r"""Evaluate the explicit kinematic inference target.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Sequence of log prior contributions in
        ``prior_names`` order.
        """
        params = self.convert_params(p)
        self.update(params)
        return self._lnpriors(p, *args, **kwargs)

    @property
    def blobs_dtype(self):
        """Return emcee blob fields for log likelihood and individual log priors.

        The list contains (name, float) pairs in the same order as the values
        returned by lnposterior after its first element.
        """
        return [("lnl", float), *((name, float) for name in self.prior_names)]

    def lnposterior(self, p, *args, **kwargs):
        r"""Evaluate the explicit kinematic inference target.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Tuple (logposterior, loglikelihood,
        individual prior terms) for emcee blobs.
        """
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
        r"""Evaluate the explicit kinematic inference target.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Tuple using loglikelihood/log(N) plus the original
        prior; requires N>1.
        """
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
        """Number of flattened physical parameters, cached on first access."""
        return len(self.params_all)


class FlatPriorModel(Model):
    r"""Finite uniform bounds in explicitly named sampling coordinates.

    The DataFrame is the single source of truth for evaluation and sampling.
    A generated template must be filled in before constructing this model.

    Notes
    -----
    **Inputs and units.** config is a pandas DataFrame indexed by ordered
    parameter names, with finite lower/upper columns, or a CSV path.
    sample(size) uses NumPy's random state. ``generate_default_config_file``
    writes a CSV template.

    **Returns and shape.** A validated prior object; sample returns coordinates
    with trailing parameter axis. lower/upper are array copies.
    ``extract_value_by_name`` expects exactly one parameter vector.

    **Validity.** Unique nonempty names and lower<upper. Bounds apply before
    log/power transforms. Unfilled default NaN bounds are intentionally unusable
    for inference.

    **Errors.** Invalid schema/bounds/vector shape raise ValueError or
    TypeError; missing CSV raises FileNotFoundError.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_inference.py``
    """

    required_param_names = []
    required_models = {}

    def __init__(self, config, show_init=False, submodels=None, **params):
        super().__init__(show_init, submodels or {}, **params)
        self.load_config(config)

    def load_config(self, config):
        """Load and copy uniform-prior bounds from a DataFrame or CSV path.

        CSV input uses its first column as the parameter-name index. The bounds
        are checked by validate_config before replacing stored data. Returns
        None; file, parse and validation errors propagate.
        """
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
        """Validate a DataFrame of explicit finite uniform-prior bounds.

        The nonempty index contains unique parameter names; columns must include
        unique lower and upper bounds with lower < upper in each row. Returns
        None. A non-DataFrame raises TypeError; invalid schema or bounds raise
        ValueError. Use load_config to read a CSV path first.
        """
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
        """Return a float array copy of lower bounds, shape (ndim,), in prior order."""
        return self.data["lower"].to_numpy(dtype=float, copy=True)

    @property
    def upper(self):
        """Return a float array copy of upper bounds, shape (ndim,), in prior order."""
        return self.data["upper"].to_numpy(dtype=float, copy=True)

    def get_index(self, param_name):
        """Return the index of the named parameter in the validated prior table.

        An unknown param_name raises KeyError.
        """
        return self.data.index.get_loc(param_name)

    def extract_value_by_name(self, params, name):
        """Extract one named sampling coordinate from a vector of shape (ndim,).

        Values retain the prior-coordinate units, including logarithmic units.
        A wrong shape raises ValueError; an unknown name raises KeyError.
        """
        if np.shape(params) != (len(self.data),):
            raise ValueError(f"Parameters must have shape ({len(self.data)},).")
        return params[self.get_index(name)]

    def sample(self, size=None):
        r"""Draw from the finite sampling-coordinate bounds.

        Notes
        -----
        **Inputs and units.** size is a sample count, tuple of sample axes or None;
        uses NumPy's global random state.

        **Returns and shape.** Uniform coordinates with trailing parameter axis;
        size=None returns one vector.
        """
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
    r"""Gaussian prior for ``log10(re_pc)``.

    Notes
    -----
    **Inputs and units.** loc and scale are location and standard deviation in
    log10(pc); sample(size) uses SciPy's random state.
    ``reset_prior(loc,scale)`` replaces that distribution.

    **Returns and shape.** A prior object; sample returns log10 radii, not
    physical pc.

    **Validity.** Finite loc and positive finite scale are required by the
    estimation model.

    **Errors.** Invalid prior values are rejected when composing
    SimpleDSphEstimationModel.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_inference.py``
    """

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
        """Replace the Gaussian prior on log10 half-light radius in pc.

        ``loc`` and positive ``scale`` are the mean and standard deviation in
        log10(pc). Returns None and replaces the stored log-PDF and sampler.
        This helper delegates domain behavior to scipy.stats.norm.
        """
        self.loc, self.scale = loc, scale
        self._lnprior_func = norm(loc=loc, scale=scale).logpdf
        self._sample = norm(loc=loc, scale=scale).rvs

    def _lnprior(self, log10_re_pc):
        return self._lnprior_func(log10_re_pc)

    def sample(self, size):
        r"""Draw a log-radius prior value.

        Notes
        -----
        **Inputs and units.** size is a sample count/shape or None, following SciPy
        normal-distribution sampling.

        **Returns and shape.** Samples in log10(pc), not physical radii; shape
        follows size.
        """
        return self._sample(size=size)

    def sampling_identity(self, sampled_names=()):
        """Return the Gaussian photometric-prior location and scale in log10(pc).

        The host metadata dictionary excludes random/runtime state.
        sampled_names is accepted for the common interface and is unused.
        """
        return {"loc": self.loc, "scale": self.scale}


class DotDict(dict):
    r"""Dictionary with attribute access retained for historical data access.

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
    r"""Kinematics-only classical dwarf-spheroidal estimation model.

    Notes
    -----
    **Inputs and units.** SimpleDSphEstimationModel composes DSphModel,
    FlatPriorModel and PhotometryPriorModel. ``args_load_data=[data]`` supplies
    a DataFrame with ``R_pc``, ``vlos_kms`` and ``e_vlos_kms``;
    ``kwargs_load_data`` may contain shared=True. Parameter vectors follow
    ``p_names_lnprob`` exactly. ``log10_`` names map to ``10**p``; ``bfunc_``
    names map to ``1-10**p``.

    **Returns and shape.** lnlikelihoods gives (N,) log densities; lnlikelihood
    sums them. lnpriors returns prior terms. lnposterior returns (logposterior,
    loglikelihood, individual prior terms) for emcee blobs. sample draws starting
    coordinates; ``sample_data`` simulates velocities at supplied positions.

    **Validity.** Nonempty finite 1-D data, R>0, error>=0; mean/error in km/s.
    The historical observation storage dtype is float32.
    ``vmem_prior_from_data`` defaults to False. WBIC uses
    ``inverse_temparature = 1/log(N)`` and requires N>1. Shared data cannot be
    resized.

    **Errors.** Invalid prior order/schema/data raise ValueError. FittableModel
    requires a list ``args_load_data``. Shared buffers must be released with
    ``release_shared_memory`` after workers stop.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_inference.py``
    """

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

    def sampling_identity(self):
        """Describe the persisted target using observations, priors and parameter order.

        Returns host metadata for the sampler's identity checks. Shared-memory
        handles, loggers and cached runtime state are excluded; observation
        values are included for both shared and ordinary storage.
        """
        # All physical coordinates are sampled by this model's validated schema.
        # Shared-memory handles and cached WBIC temperature are runtime state;
        # the observations themselves are hashed for shared and ordinary models.
        ignored = {"logger", "_parammap", "params", "submodels", "_data",
                   "shared", "shared_shape", "buffer_size", "inverse_temparature",
                   "shm_R_pc", "shm_vlos_kms", "shm_e_vlos_kms"}
        state = {k: v for k, v in vars(self).items() if k not in ignored}
        state["data"] = self.data
        state["parameter_order"] = self.p_names_lnprob
        state["submodels"] = {
            k: (type(v), v.sampling_identity(self.required_param_names_combined))
            for k, v in self.submodels.items()
        }
        return state

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
        """Return sampling-coordinate names in the exact order expected by lnposterior."""
        return self["FlatPriorModel"].data.index.tolist()

    def convert_params(self, p):
        r"""Map sampling coordinates to physical parameters.

        Notes
        -----
        **Inputs and units.** One parameter vector in exact prior order; ``log10_``
        and ``bfunc_`` prefixes identify the supported transforms.

        **Returns and shape.** Named physical parameters with pc, Msun/pc^3, km/s,
        radians and dimensionless quantities as appropriate. The axisymmetric result
        also incorporates ``fixed_params``.
        """
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
        """Return the observation-buffer name for this instance, or None if unshared."""
        if not self.shared:
            return None
        return f"SimpleDSphEstimationModel_{id(self)}"

    @property
    def data(self):
        """Access stored radii, velocities and velocity errors as named arrays.

        Columns R_pc, vlos_kms and e_vlos_kms have shape (N,) and units pc, km/s
        and km/s. Shared mode returns views of the shared buffers and raises
        FileNotFoundError or AttributeError if they have not been initialized.
        """
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
        """Return the number of observed stars as an integer."""
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
        """Close and release this model's three shared observation buffers.

        Returns None. Existing array views must no longer be used after their
        buffers are released.
        """
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
        """Draw sampling-coordinate vectors from the specified joint prior.

        Uniform bounds apply to every coordinate; the log-radius coordinate
        is drawn from the product of those bounds and the Gaussian photometric
        prior. ``size=None`` returns (ndim,); a sample count/shape precedes that
        parameter axis. Uses the NumPy/SciPy global random state. Invalid prior
        schemas raise ValueError before sampling.
        """
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
        """Draw conditional Gaussian LOS velocities at the stored positions.

        The mean is vmem_kms and the variance is the predicted LOS variance plus
        the squared stored measurement error, in (km/s)^2. ``size=None`` returns
        a broadcast vector of shape (N,); explicit sizes must be compatible with
        that per-star shape. Uses SciPy's global random state. Positions and
        measurement errors remain fixed; this is not a phase-space DF sampler.
        """
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
    r"""Compose Plummer + NFW + constant anisotropy with explicit finite priors.

    ``config`` is a DataFrame or CSV in this order: vmem_kms, log10_re_pc,
    log10_rs_pc, log10_rhos_Msunpc3, log10_r_t_pc, bfunc_beta_ani.
    A missing CSV is created as an unfilled template, then raises ValueError.
    Caller velocity bounds are preserved unless vmem_prior_from_data is True.

    Notes
    -----
    **Inputs and units.** data is an observed DataFrame; config is a finite
    ordered prior DataFrame or CSV path; ``photometry_prior_loc``/scale specify
    the Gaussian in log10(``re_pc``).

    **Returns and shape.** SimpleDSphEstimationModel with the standard
    Plummer/NFW/constant-anisotropy components.

    **Validity.** A convenience constructor does not choose scientifically
    justified prior bounds for the caller.

    **Errors.** Missing/invalid data or prior configuration raises; templates
    must be completed first.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_inference.py``
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
