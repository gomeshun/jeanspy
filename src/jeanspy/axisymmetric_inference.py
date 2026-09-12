"""Classical axisymmetric kinematic inference using :class:`jeanspy.sampler.Sampler`.

Priors are explicit in named sampling coordinates. ``log10_`` and ``bfunc_``
have the same meanings as in the spherical estimation model; the additional
coordinate ``cos_inclination`` maps to an inclination in radians. No prior is
inferred from observed velocities. The likelihood assumes zero mean streaming.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

from ._axisymmetric_params import resolve_params
from ._classical.inference import FlatPriorModel, PhotometryPriorModel
from .axisymmetric import AxisymmetricDSphModel, InvalidAxisymmetricModelError

__all__ = ["AxisymmetricKinematicData", "AxisymmetricDSphEstimationModel"]


@dataclass(frozen=True, eq=False)
class AxisymmetricKinematicData:
    r"""Finite matching 1-D observations, in pc and km/s, copied on construction.

    Sky x follows the line of nodes (the projected major axis for an oblate
    tracer); y is the perpendicular sky coordinate. Position-angle rotation and
    angular-to-physical conversion must be applied before constructing the data.
    Both signed coordinates and the projected center are supported.

    Notes
    -----
    **Inputs and units.** ``x_pc``, ``y_pc``, ``vlos_kms`` and ``e_vlos_kms``
    are matching nonempty finite 1-D arrays; pc and km/s. ``from_data`` accepts
    the supported mapping/table or an existing data object.

    **Returns and shape.** An immutable data container; ``as_kwargs`` returns
    the four arrays under their public names; len is N.

    **Validity.** Signed sky coordinates include the center; velocity errors
    must be nonnegative.

    **Errors.** Missing fields, mismatched shapes or invalid values raise.

    **Backend.** NumPy host arrays.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_inference.py``
    """
    x_pc: np.ndarray
    y_pc: np.ndarray
    vlos_kms: np.ndarray
    e_vlos_kms: np.ndarray

    def __post_init__(self):
        arrays = {key: np.array(value, dtype=float, copy=True)
                  for key, value in vars(self).items()}
        shape = arrays["x_pc"].shape
        if len(shape) != 1 or shape[0] == 0 or any(a.shape != shape for a in arrays.values()):
            raise ValueError("Kinematic observations must be matching nonempty 1-D arrays")
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("Kinematic observations must be finite")
        if np.any(arrays["e_vlos_kms"] < 0):
            raise ValueError("e_vlos_kms must be nonnegative")
        for key, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, key, value)

    @classmethod
    def from_data(cls, data):
        """Copy a DataFrame, mapping, or another kinematic data object."""
        names = cls.__dataclass_fields__
        if isinstance(data, cls):
            return cls(**{name: getattr(data, name) for name in names})
        if isinstance(data, pd.DataFrame) and not data.columns.is_unique:
            raise ValueError("Kinematic data column names must be unique")
        missing = set(names) - data.keys()
        if missing:
            raise ValueError(f"Missing kinematic columns: {sorted(missing)}")
        return cls(**{name: data[name] for name in names})

    def as_kwargs(self):
        """Return detached arrays for a NumPyro model or sampler call."""
        return {name: value.copy() for name, value in vars(self).items()}

    def __len__(self):
        return self.x_pc.size


def _physical_name(name):
    if name == "cos_inclination":
        return "inclination"
    return name[6:] if name.startswith(("log10_", "bfunc_")) else name


class AxisymmetricDSphEstimationModel:
    r"""Unbinned Gaussian LOS inference with the classical sampler protocol.

    ``prior`` is a :class:`FlatPriorModel`, a DataFrame, or a CSV with finite
    ``lower``/``upper`` bounds. Its row order defines the sampler coordinates.
    ``fixed_params`` supplies remaining physical parameters; sampled and fixed
    names must be disjoint. Supply exactly one of q and q_projected across them.

    An optional :class:`PhotometryPriorModel` multiplies the flat prior on
    ``log10_re_pc``. A uniform ``cos_inclination`` coordinate gives an isotropic
    orientation prior restricted to its explicitly supplied bounds. The solver
    also enforces physically admissible deprojection and nonnegative moments.

    Notes
    -----
    **Inputs and units.** data follows AxisymmetricKinematicData; prior is
    FlatPriorModel or ordered lower/upper DataFrame; ``fixed_params``
    complements sampled names; ``dsph_model`` is a NumPy AxisymmetricDSphModel.
    p is shape (ndim,) in prior order; ``log10_`` and ``bfunc_`` transforms are
    explicit.

    **Returns and shape.** Per-star/summed log likelihoods and prior terms.
    lnposterior returns posterior plus diagnostic blobs; sample(size,rng=...)
    generates admissible starting coordinates; ``sample_data(p,rng=...)`` gives
    simulated data.

    **Validity.** Gaussian LOS velocity likelihood at fixed positions;
    ``beta_z`` is distinct from spherical anisotropy. Exactly one of
    intrinsic/projected tracer flattening must be specified. Photometric prior
    is optional and explicit.

    **Errors.** Malformed schema/data or sampled/fixed collisions raise
    ValueError; inadmissible proposals give minus-infinite posterior. sample
    raises after ``max_attempts`` if no valid point is found.

    **Backend.** NumPy/SciPy CPU; stateful components, with no JAX tracing.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_inference.py``
    """
    name = "AxisymmetricDSphEstimationModel"

    def __init__(self, data, prior, *, dsph_model=None, fixed_params=None,
                 photometry_prior=None):
        self.dsph_model = AxisymmetricDSphModel() if dsph_model is None else dsph_model
        if not isinstance(self.dsph_model, AxisymmetricDSphModel):
            raise TypeError("dsph_model must be a classical AxisymmetricDSphModel")
        self.prior = (FlatPriorModel(prior.data) if isinstance(prior, FlatPriorModel)
                      else FlatPriorModel(prior))
        self.fixed_params = dict(fixed_params or {})
        self.photometry_prior = photometry_prior
        self._validate_schema()
        self.reset_data(data)

    @property
    def p_names_lnprob(self):
        """Return sampling-coordinate names in their validated prior-table order."""
        return self.prior.data.index.tolist()

    @property
    def ndim(self):
        """Return the integer number of free sampling coordinates."""
        return len(self.p_names_lnprob)

    @property
    def prior_names(self):
        """Return log-prior term names in blob order, including optional photometry."""
        names = ["flat_prior", "physical_domain"]
        return names + (["photometry_prior"] if self.photometry_prior is not None else [])

    @property
    def blobs_dtype(self):
        """Return emcee blob fields for the log likelihood and ordered log-prior terms."""
        return [("lnl", float), *((name, float) for name in self.prior_names)]

    @property
    def n_data(self):
        """Return the integer number of observed stars."""
        return len(self._data)

    @property
    def data(self):
        """Return a validated copy of the stored axisymmetric kinematic catalogue.

        The AxisymmetricKinematicData arrays have shape (N,); x_pc/y_pc are in pc
        and velocities/errors are in km/s.
        """
        return AxisymmetricKinematicData.from_data(self._data)

    @property
    def inverse_temparature(self):
        """WBIC inverse temperature; historical spelling matches the sampler."""
        if self.n_data <= 1:
            raise ValueError("WBIC requires at least two observations")
        return 1 / np.log(self.n_data)

    def reset_data(self, data):
        """Validate a replacement completely before changing any observations."""
        replacement = AxisymmetricKinematicData.from_data(data)
        self._data = replacement

    def _validate_schema(self):
        self.prior.validate_config(self.prior.data)
        physical = [_physical_name(name) for name in self.p_names_lnprob]
        if len(set(physical)) != len(physical) or set(physical) & self.fixed_params.keys():
            raise ValueError("Sampled physical names must be unique and disjoint from fixed_params")
        # Check structure independently of numerical validity of the prior midpoint.
        params = {**self.fixed_params, **dict.fromkeys(physical, 1.)}
        resolve_params(params, np)
        if "vmem_kms" not in params:
            raise ValueError("Supply an explicit sampled or fixed vmem_kms")
        if any(np.ndim(v) != 0 or not (np.isfinite(v) or (k == "r_t_pc" and v == np.inf))
               for k, v in self.fixed_params.items()):
            raise ValueError("fixed_params must contain finite scalar physical values")
        if self.photometry_prior is not None:
            if not isinstance(self.photometry_prior, PhotometryPriorModel):
                raise TypeError("photometry_prior must be a PhotometryPriorModel")
            if "log10_re_pc" not in self.p_names_lnprob:
                raise ValueError("The photometry prior requires the coordinate log10_re_pc")
            loc, scale = self.photometry_prior.loc, self.photometry_prior.scale
            if not np.isfinite(loc) or not np.isfinite(scale) or scale <= 0:
                raise ValueError("Photometry prior needs a finite location and positive finite scale")

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
        self._validate_schema()
        p = np.asarray(p, dtype=float)
        if p.shape != (self.ndim,):
            raise ValueError(f"Parameters must have shape ({self.ndim},) in prior config order")
        values = dict(self.fixed_params)
        with np.errstate(over="ignore", invalid="ignore"):
            for name, value in zip(self.p_names_lnprob, p):
                if name.startswith("log10_"):
                    value = 10.**value
                elif name.startswith("bfunc_"):
                    value = 1 - 10.**value
                elif name == "cos_inclination":
                    value = np.arccos(value)
                values[_physical_name(name)] = value
        return values

    def _lnlikelihoods(self, params):
        try:
            sigma2 = self.dsph_model.sigmalos2(self._data.x_pc, self._data.y_pc, params=params)
        except InvalidAxisymmetricModelError:
            return np.full(self.n_data, -np.inf)
        mean = params["vmem_kms"]
        if not np.isfinite(mean) or np.any(~np.isfinite(sigma2)) or np.any(sigma2 < 0):
            return np.full(self.n_data, -np.inf)
        scale = np.hypot(np.sqrt(sigma2), self._data.e_vlos_kms)
        if np.any(scale <= 0):
            return np.full(self.n_data, -np.inf)
        return norm.logpdf(self._data.vlos_kms, loc=mean, scale=scale)

    def lnlikelihoods(self, p):
        r"""Return one log likelihood per star, including measurement errors.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Per-star log densities, shape (N,).
        """
        return self._lnlikelihoods(self.convert_params(p))

    def lnlikelihood(self, p):
        r"""Evaluate the explicit kinematic inference target.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Scalar sum of log likelihoods.
        """
        return float(np.sum(self.lnlikelihoods(p)))

    def _lnpriors(self, p, params):
        _, valid = resolve_params(params, np)
        valid = valid and np.isfinite(params["vmem_kms"])
        result = [self.prior._lnprior(p), 0. if valid else -np.inf]
        if self.photometry_prior is not None:
            value = self.photometry_prior._lnprior(p[self.p_names_lnprob.index("log10_re_pc")])
            result.append(float(value) if np.isfinite(value) else -np.inf)
        return result

    def lnpriors(self, p):
        r"""Evaluate the explicit kinematic inference target.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Sequence of log prior contributions in
        ``prior_names`` order.
        """
        return self._lnpriors(p, self.convert_params(p))

    def _posterior(self, p, temperature):
        params = self.convert_params(p)
        priors = self._lnpriors(p, params)
        lnl = -np.inf
        if np.all(np.isfinite(priors)):
            lnl = float(np.sum(self._lnlikelihoods(params))) * temperature
        return (lnl + sum(priors), lnl, *priors)

    def lnposterior(self, p):
        r"""Evaluate the explicit kinematic inference target.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Tuple (logposterior, loglikelihood,
        individual prior terms) for emcee blobs.
        """
        return self._posterior(p, 1.)

    def lnposterior_wbic(self, p):
        r"""Evaluate the explicit kinematic inference target.

        Notes
        -----
        **Inputs and units.** p is one parameter vector of shape (ndim,) in
        ``p_names_lnprob`` order. Uses already loaded observations.

        **Returns and shape.** Tuple using loglikelihood/log(N) plus the original
        prior; requires N>1.
        """
        return self._posterior(p, self.inverse_temparature)

    def sample(self, size=None, *, rng=None, max_attempts=1000):
        """Draw feasible starting points from the priors, with bounded rejection.

        Pass a NumPy Generator or seed for reproducibility. Exhaustion raises
        rather than returning invalid walkers or changing the requested priors.
        """
        self._validate_schema()
        rng = np.random.default_rng(rng)
        shape = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("size must contain nonnegative integers")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        result = np.empty((prod(shape), self.ndim))
        accepted = 0
        for _ in range(max_attempts):
            if accepted == len(result):
                break
            p = rng.uniform(self.prior.lower, self.prior.upper)
            if self.photometry_prior is not None:
                k = self.p_names_lnprob.index("log10_re_pc")
                loc, scale = self.photometry_prior.loc, self.photometry_prior.scale
                a, b = (self.prior.lower[k]-loc)/scale, (self.prior.upper[k]-loc)/scale
                p[k] = truncnorm.rvs(a, b, loc=loc, scale=scale, random_state=rng)
            if np.isfinite(self.lnposterior(p)[0]):
                result[accepted] = p
                accepted += 1
        if accepted != len(result):
            raise RuntimeError(f"Only {accepted}/{len(result)} feasible prior draws in {max_attempts} attempts; "
                               "check inclination, flattening, anisotropy and quadrature convergence")
        return result.reshape(shape + (self.ndim,))

    def sample_data(self, p, *, rng=None):
        """Generate velocities conditional on the stored positions and errors."""
        params = self.convert_params(p)
        sigma2 = self.dsph_model.sigmalos2(self._data.x_pc, self._data.y_pc, params=params)
        return np.random.default_rng(rng).normal(params["vmem_kms"],
                         np.hypot(np.sqrt(sigma2), self._data.e_vlos_kms))

    def sampling_identity(self):
        """Return host metadata describing the complete sampling target.

        The dictionary contains the forward model, copied observations, prior
        table, fixed parameters, optional photometric prior and coordinate order.
        The persistence layer hashes this material; this method returns no hash.
        """
        return dict(dsph_model=self.dsph_model, data=self._data.as_kwargs(),
                    prior=self.prior.data, fixed_params=self.fixed_params,
                    photometry_prior=self.photometry_prior, parameter_order=self.p_names_lnprob)
