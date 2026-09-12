from __future__ import annotations

import json
import logging
import pickle
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal, Mapping, Sequence, cast

from ._jax_env import configure_jax_environment

configure_jax_environment()

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import xarray as xr
from numpyro.infer import MCMC

from .model_numpyro import DSphModel
from ._axisymmetric_params import validate_param_names
from ._sampling_identity import fingerprint, software_identity


logger = logging.getLogger(__name__)


#: Accepted sample-store names; install the dependencies for the selected backend.
#: These host-side I/O choices do not affect the physical differentiation graph.
StorageBackend = Literal["zarr", "h5netcdf", "netcdf4"]


_CHECKPOINT_FORMAT_VERSION = 2
_STORAGE_FORMAT_VERSION = 1
_CHECKPOINT_FILENAME = "last_state.pkl"
_METADATA_FILENAME = "metadata.json"
_CHUNKS_DIRNAME = "chunks"
_CHUNK_PREFIX = "chunk_"
_DEFAULT_STORAGE_BACKEND: StorageBackend = "zarr"
_STORAGE_CONFIGS: dict[StorageBackend, dict[str, Any]] = {
    "zarr": {
        "suffix": ".zarr",
        "writer": "zarr",
        "reader": az.from_zarr,
        "reader_kwargs": {"engine": "zarr", "consolidated": False},
        "directory_store": True,
    },
    "h5netcdf": {
        "suffix": ".nc",
        "writer": "netcdf",
        "reader": az.from_netcdf,
        "reader_kwargs": {"engine": "h5netcdf"},
        "directory_store": False,
    },
    "netcdf4": {
        "suffix": ".nc",
        "writer": "netcdf",
        "reader": az.from_netcdf,
        "reader_kwargs": {"engine": "netcdf4"},
        "directory_store": False,
    },
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pow10(value: Any) -> Any:
    return jnp.power(10.0, value)


def _normalize_storage_backend(storage_backend: str) -> StorageBackend:
    normalized = str(storage_backend).strip().lower()
    if normalized not in _STORAGE_CONFIGS:
        allowed = ", ".join(sorted(_STORAGE_CONFIGS))
        raise ValueError(f"Unsupported storage_backend {storage_backend!r}; expected one of: {allowed}")
    return normalized  # type: ignore[return-value]


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _to_host_tree(tree: Any) -> Any:
    return jax.device_get(tree)


def _to_device_leaf(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    if isinstance(value, np.generic):
        return jnp.asarray(value)
    if isinstance(value, (bool, int, float, complex)):
        return jnp.asarray(value)
    return value


def _to_device_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(_to_device_leaf, tree)


def _concat_draw_datasets(group_path: str, datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    if not datasets:
        raise ValueError(f"No datasets available for group {group_path}")

    draw_presence = ["draw" in dataset.dims for dataset in datasets]
    if any(draw_presence):
        if not all(draw_presence):
            raise ValueError(f"Group {group_path} is inconsistent across chunks")
        combined = xr.concat(
            datasets,
            dim="draw",
            data_vars="all",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )
        draw_size = int(combined.sizes["draw"])
        return combined.assign_coords(draw=np.arange(draw_size, dtype=np.int64))

    first = datasets[0]
    for dataset in datasets[1:]:
        if not first.equals(dataset):
            raise ValueError(f"Static group {group_path} changed across chunks")
    return first


@dataclass(frozen=True)
class ParameterSpec:
    r"""Describe a parameter site and its physical representation.

    Without param_name, transformed values keep the sample_name dictionary key
    and are recorded at sample_name + '_transformed' to avoid a site collision.

    Notes
    -----
    **Inputs and units.** ``sample_name`` names a NumPyro sample site;
    distribution is a distribution or zero-argument factory; ``param_name``
    names the physical parameter; transform is a callable;
    ``record_deterministic``/``deterministic_name`` control recorded transformed
    sites. exp and pow10 constructors set exponential/base-10 transforms.

    **Returns and shape.** Specification object. ``build_distribution`` returns
    a NumPyro distribution; sample returns (``physical_name``,
    ``physical_value``) and records its configured sample/deterministic sites.

    **Validity.** Priors live in the sampled coordinate. Transforming the value
    does not turn a log-uniform prior into a uniform prior in physical units.

    **Errors.** Invalid/duplicate names and inconsistent deterministic-site
    configuration raise.

    **Backend.** NumPyro/JAX.

    **Differentiation.** Transforms/distributions must support the intended JAX
    derivatives; discrete sample sites are not NUTS coordinates.

    **Examples.** ``examples/docs_inference.py``
    """

    sample_name: str
    distribution: Any
    param_name: str | None = None
    transform: Callable[[Any], Any] | None = None
    record_deterministic: bool | None = None
    deterministic_name: str | None = None

    def __post_init__(self):
        for name in (self.sample_name, self.param_name, self.deterministic_name):
            if name is not None and (not isinstance(name, str) or not name):
                raise ValueError("Parameter site names must be nonempty strings")
        if self.sample_name is None:
            raise ValueError("sample_name must be a nonempty string")
        if self.deterministic_name == self.sample_name:
            raise ValueError("deterministic_name must differ from sample_name")

    @property
    def records_deterministic(self) -> bool:
        """Whether this specification records a deterministic physical-parameter site.

        An explicit deterministic_name enables recording; otherwise an explicit
        record_deterministic flag wins, then transform/renaming determines the
        default. Returns bool.
        """
        if self.deterministic_name is not None:
            return True
        if self.record_deterministic is not None:
            return self.record_deterministic
        return self.transform is not None or (self.param_name or self.sample_name) != self.sample_name

    @property
    def resolved_deterministic_name(self) -> str:
        """Return the deterministic-site name without colliding with sample_name.

        Explicit deterministic_name takes precedence, followed by param_name.
        If the inferred name equals sample_name, append _transformed. An
        explicit deterministic_name equal to sample_name raises ValueError.
        """
        name = self.deterministic_name or self.param_name or self.sample_name
        if name == self.sample_name:
            if self.deterministic_name is not None:
                raise ValueError("deterministic_name must differ from sample_name")
            name += "_transformed"
        return name

    @classmethod
    def exp(
        cls,
        sample_name: str,
        distribution: Any,
        *,
        param_name: str | None = None,
        deterministic_name: str | None = None,
    ) -> "ParameterSpec":
        """Construct a specification whose physical value is exp(sample_value).

        ``sample_name`` identifies the NumPyro site and ``distribution`` is its
        prior in natural-logarithmic coordinates. Optional param_name chooses
        the physical dictionary key; deterministic_name chooses the recorded
        physical site. Returns a ParameterSpec with deterministic recording
        enabled. The distribution is not a prior on the exponentiated value.
        """
        return cls(
            sample_name=sample_name,
            distribution=distribution,
            param_name=param_name,
            transform=jnp.exp,
            record_deterministic=True,
            deterministic_name=deterministic_name,
        )

    @classmethod
    def pow10(
        cls,
        sample_name: str,
        distribution: Any,
        *,
        param_name: str | None = None,
        deterministic_name: str | None = None,
    ) -> "ParameterSpec":
        """Construct a specification whose physical value is 10**sample_value.

        ``sample_name`` identifies the NumPyro site and ``distribution`` is its
        prior in base-ten logarithmic coordinates. Optional param_name chooses
        the physical dictionary key; deterministic_name chooses the recorded
        physical site. Returns a ParameterSpec with deterministic recording
        enabled. The distribution is not a prior on the exponentiated value.
        """
        return cls(
            sample_name=sample_name,
            distribution=distribution,
            param_name=param_name,
            transform=_pow10,
            record_deterministic=True,
            deterministic_name=deterministic_name,
        )

    def build_distribution(self) -> Any:
        """Resolve an existing distribution or call a zero-argument factory.

        Returns the supplied NumPyro distribution unchanged, or the factory
        result. Factory exceptions propagate; downstream NumPyro execution
        checks whether the result is a usable distribution.
        """
        if isinstance(self.distribution, dist.Distribution):
            return self.distribution
        return self.distribution() if callable(self.distribution) else self.distribution

    def sample(self) -> tuple[str, Any]:
        """Create the NumPyro sample site and return its named physical value.

        Returns (resolved parameter name, transformed value). The distribution
        is defined in sample coordinates; transform is then applied and, when
        configured, a deterministic site records the physical value. Execute
        under NumPyro inference or a seeded handler. Units/shapes follow the
        distribution and physical transform.
        """
        raw_value = numpyro.sample(self.sample_name, self.build_distribution())
        resolved_name = self.param_name or self.sample_name
        value = self.transform(raw_value) if self.transform is not None else raw_value

        if self.records_deterministic:
            numpyro.deterministic(self.resolved_deterministic_name, value)

        return resolved_name, value


class JeansLikelihoodModel:
    r"""Callable spherical NumPyro model for line-of-sight velocity inference.

    Parameters
    ----------
    dsph_model : jeanspy.model_numpyro.DSphModel
        Functional JAX forward model. Its LOS variance is in (km/s)**2.
    parameter_specs : sequence of ParameterSpec
        Ordered sample sites, priors and transformations to physical parameters.
        Priors are densities in the named sampling coordinates.
    sigmalos2_kwargs : mapping or None, optional
        Static integration settings passed to the forward ``sigmalos2`` method.
    sigma2_bounds : pair of float, optional
        Positive ordered rejection limits in (km/s)**2. Values outside these
        limits are rejected with log probability minus infinity, not clipped.
    velocity_mean : str or callable, optional
        Physical parameter name (default ``vmem_kms``) or a function of the
        parameter mapping returning the velocity mean in km/s.
    observation_distribution : callable, optional
        Distribution factory accepting a mean and standard deviation in km/s;
        defaults to ``numpyro.distributions.Normal``.
    observed_name : str, optional
        NumPyro observation-site name, default ``vlos``.
    parameter_postprocess : callable or None, optional
        Optional mapping-to-mapping physical-parameter transformation. A traced
        likelihood requires a JAX-compatible callable.

    Notes
    -----
    Calling the instance takes matching 1-D arrays ``R_pc``, ``vlos_kms`` and
    ``e_vlos_kms`` (pc, km/s, km/s), and adds observation and validity sites to
    the active NumPyro trace. Radii must be finite and positive; errors finite
    and nonnegative. The standard likelihood conditions on the observed radii.
    Physical-parameter gradients use the JAX forward path and differentiable
    transforms in the admissible interior. This class does not calculate J/D
    factors or establish a positive phase-space distribution function.

    **Inputs and units.** ``dsph_model`` is the matching JAX model;
    ``parameter_specs`` is a sequence of ParameterSpec; the axisymmetric
    subclass accepts ``fixed_params`` for complementary scalars; the spherical
    class uses ``parameter_postprocess`` to assemble additional fixed
    parameters. Velocity mean and sigmalos2 options are explicit. Calling the
    spherical model takes ``R_pc``/``vlos_kms``/``e_vlos_kms``; the axisymmetric
    model takes ``x_pc``/``y_pc`` instead of ``R_pc``. Observation arrays are
    matching finite nonempty 1-D arrays.

    **Returns and shape.** __call__ returns None while registering NumPyro
    sample, deterministic and likelihood sites; ``sample_parameters`` returns
    the transformed parameter mapping. Variance includes measurement error
    squared.

    **Validity.** Positions in pc and velocities/errors in km/s, errors>=0.
    Gaussian LOS closure at fixed positions. The standard class does not add
    membership mixtures, velocity-cut normalization or binaries.

    **Errors.** Bad schema/shape and sampled/fixed collisions raise;
    inadmissible forward variances are rejected with minus-infinite density.

    **Backend.** NumPyro with JAX forward model.

    **Differentiation.** Supported continuous physical parameters are traceable.
    J/D factors are not likelihood sites. Verify derivatives for custom prior
    transforms.

    **Examples.** ``examples/docs_numpyro_inference.py``

    Raises
    ------
    ValueError
        Sample, deterministic or observation sites collide; physical parameter
        names repeat; or the variance rejection limits are invalid.
    """

    def __init__(
        self,
        dsph_model: DSphModel,
        parameter_specs: Sequence[ParameterSpec],
        *,
        sigmalos2_kwargs: Mapping[str, Any] | None = None,
        sigma2_bounds: tuple[float, float] = (1e-12, 1e12),
        velocity_mean: str | Callable[[Mapping[str, Any]], Any] = "vmem_kms",
        observation_distribution: Callable[[Any, Any], Any] = dist.Normal,
        observed_name: str = "vlos",
        parameter_postprocess: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.dsph_model = dsph_model
        self.parameter_specs = tuple(parameter_specs)
        sites = [observed_name, "valid_observations", "valid_sigmalos2", "valid_velocity_mean"]
        names = []
        for spec in self.parameter_specs:
            sites.append(spec.sample_name)
            names.append(spec.param_name or spec.sample_name)
            if spec.records_deterministic:
                sites.append(spec.resolved_deterministic_name)
        if len(sites) != len(set(sites)) or len(names) != len(set(names)):
            raise ValueError("Parameter and observation site names and physical parameter names must be unique")
        self.sigmalos2_kwargs = dict(sigmalos2_kwargs or {})
        if len(sigma2_bounds) != 2:
            raise ValueError("sigma2_bounds must contain exactly two limits")
        self.sigma2_bounds = (float(sigma2_bounds[0]), float(sigma2_bounds[1]))
        if not (np.all(np.isfinite(self.sigma2_bounds))
                and 0 < self.sigma2_bounds[0] <= self.sigma2_bounds[1]):
            raise ValueError("sigma2_bounds must be finite, positive, and ordered")
        self.velocity_mean = velocity_mean
        self.observation_distribution = observation_distribution
        self.observed_name = observed_name
        self.parameter_postprocess = parameter_postprocess

    def sample_parameters(self) -> dict[str, Any]:
        """Draw ParameterSpec values inside a NumPyro model execution.

        Returns the physical-parameter mapping after parameter_postprocess, if
        supplied. Execute under NumPyro inference or a seeded handler. Parameter
        units and shapes follow the individual specifications.
        """
        params: dict[str, Any] = {}
        for parameter_spec in self.parameter_specs:
            param_name, value = parameter_spec.sample()
            params[param_name] = value

        if self.parameter_postprocess is not None:
            params = dict(self.parameter_postprocess(dict(params)))

        return params

    def _resolve_velocity_mean(self, params: Mapping[str, Any]) -> Any:
        if callable(self.velocity_mean):
            return self.velocity_mean(params)
        return params[self.velocity_mean]

    def __call__(self, R_pc: Any, vlos_kms: Any, e_vlos_kms: Any) -> None:
        R, velocity, error = (jnp.asarray(v) for v in (R_pc, vlos_kms, e_vlos_kms))
        if (R.ndim != 1 or R.size == 0 or velocity.shape != R.shape
                or error.shape != R.shape):
            raise ValueError("R_pc, vlos_kms, and e_vlos_kms must be matching nonempty 1-D arrays")
        valid_data = jnp.all(jnp.isfinite(R) & (R > 0) & jnp.isfinite(velocity)
                             & jnp.isfinite(error) & (error >= 0))
        numpyro.factor("valid_observations", jnp.where(valid_data, 0.0, -jnp.inf))
        R = jnp.where(jnp.isfinite(R) & (R > 0), R, 1.0)
        velocity = jnp.where(jnp.isfinite(velocity), velocity, 0.0)
        error = jnp.where(jnp.isfinite(error) & (error >= 0), error, 0.0)
        params = self.sample_parameters()
        sigma2 = jnp.asarray(self.dsph_model.sigmalos2(R, params=params, **self.sigmalos2_kwargs))
        if sigma2.shape != R.shape:
            raise ValueError("sigmalos2 must return one variance per observed radius")
        valid_sigma2 = jnp.all(jnp.isfinite(sigma2) & (sigma2 >= 0))
        numpyro.factor("valid_sigmalos2", jnp.where(valid_sigma2, 0.0, -jnp.inf))
        # Reject invalid models, but keep the observation distribution well-defined.
        sigma2 = jnp.where(jnp.isfinite(sigma2) & (sigma2 >= 0), sigma2, 1.0)
        sigma2 = jnp.clip(sigma2, min=self.sigma2_bounds[0], max=self.sigma2_bounds[1])
        scale = jnp.hypot(jnp.sqrt(sigma2), error)
        loc = jnp.asarray(self._resolve_velocity_mean(params))
        if loc.ndim != 0 and loc.shape != R.shape:
            raise ValueError("velocity_mean must be scalar or match the observation shape")
        numpyro.factor("valid_velocity_mean", jnp.where(jnp.all(jnp.isfinite(loc)), 0.0, -jnp.inf))
        loc = jnp.where(jnp.isfinite(loc), loc, 0.0)
        numpyro.sample(
            self.observed_name,
            self.observation_distribution(loc, scale),
            obs=velocity,
        )


class AxisymmetricJeansLikelihoodModel(JeansLikelihoodModel):
    r"""NumPyro likelihood for signed sky coordinates and zero mean streaming.

    Uses the same ParameterSpec and NumPyroSampler contracts as the spherical
    likelihood. ``fixed_params`` and sampled physical names must be disjoint;
    optional postprocessing receives their combined dictionary. All angles in
    that dictionary are radians. Both q and q_projected parameterizations are
    supported by the axisymmetric forward model.

    Parameter names and a named velocity mean are checked at construction.
    With ``parameter_postprocess``, its output is checked after the callback
    instead; construction does not execute user callbacks or sample priors.

    ``sigma2_bounds`` are admissibility limits: variances outside the interval
    receive zero likelihood, without clipping a finite prediction to a bound.

    Notes
    -----
    **Inputs and units.** ``dsph_model`` is the matching JAX model;
    ``parameter_specs`` is a sequence of ParameterSpec; the axisymmetric
    subclass accepts ``fixed_params`` for complementary scalars; the spherical
    class uses ``parameter_postprocess`` to assemble additional fixed
    parameters. Velocity mean and sigmalos2 options are explicit. Calling the
    spherical model takes ``R_pc``/``vlos_kms``/``e_vlos_kms``; the axisymmetric
    model takes ``x_pc``/``y_pc`` instead of ``R_pc``. Observation arrays are
    matching finite nonempty 1-D arrays.

    **Returns and shape.** __call__ returns None while registering NumPyro
    sample, deterministic and likelihood sites; ``sample_parameters`` returns
    the transformed parameter mapping. Variance includes measurement error
    squared.

    **Validity.** Positions in pc and velocities/errors in km/s, errors>=0.
    Gaussian LOS closure at fixed positions. The standard class does not add
    membership mixtures, velocity-cut normalization or binaries.

    **Errors.** Bad schema/shape and sampled/fixed collisions raise;
    inadmissible forward variances are rejected with minus-infinite density.

    **Backend.** NumPyro with JAX forward model.

    **Differentiation.** Supported continuous physical parameters are traceable.
    J/D factors are not likelihood sites. Verify derivatives for custom prior
    transforms.

    **Examples.** ``examples/docs_numpyro_inference.py``
    """

    def __init__(self, dsph_model, parameter_specs, *, fixed_params=None, **kwargs):
        super().__init__(dsph_model, parameter_specs, **kwargs)
        self.fixed_params = dict(fixed_params or {})
        names = {spec.param_name or spec.sample_name for spec in self.parameter_specs}
        if names & self.fixed_params.keys():
            raise ValueError("Sampled physical names must be disjoint from fixed_params")
        if any(np.ndim(v) != 0 or not (np.isfinite(v) or (k == "r_t_pc" and v == np.inf))
               for k, v in self.fixed_params.items()):
            raise ValueError("fixed_params must contain finite scalar physical values")
        if self.parameter_postprocess is None:
            self._validate_parameter_names(names | self.fixed_params.keys())

    def _validate_parameter_names(self, names):
        validate_param_names(names)
        if not callable(self.velocity_mean) and self.velocity_mean not in names:
            raise ValueError(f"Missing velocity_mean parameter: {self.velocity_mean!r}")

    def sample_parameters(self):
        """Draw named physical parameters inside a NumPyro model execution.

        Combines fixed_params with the sampled ParameterSpec values, then applies
        parameter_postprocess if supplied. Returns a physical-parameter mapping;
        invalid parameter names raise ValueError. Run under NumPyro inference or
        a seeded handler, not as an unseeded standalone random-number call.
        """
        params = dict(self.fixed_params)
        for spec in self.parameter_specs:
            name, value = spec.sample()
            params[name] = value
        if self.parameter_postprocess is not None:
            params = dict(self.parameter_postprocess(dict(params)))
        self._validate_parameter_names(params)
        return params

    def sampling_identity(self):
        """Return model attributes for deterministic sampling-target identification.

        The host dictionary includes the forward model, prior specifications and
        fixed settings; it is metadata rather than a physical prediction.
        """
        return dict(vars(self))

    def __call__(self, x_pc, y_pc, vlos_kms, e_vlos_kms):
        x, y, velocity, error = (jnp.asarray(v, dtype=float)
                                for v in (x_pc, y_pc, vlos_kms, e_vlos_kms))
        if (x.ndim != 1 or x.size == 0
                or any(v.shape != x.shape for v in (y, velocity, error))):
            raise ValueError("x_pc, y_pc, vlos_kms and e_vlos_kms must be matching nonempty 1-D arrays")
        valid_data = jnp.all(jnp.isfinite(x) & jnp.isfinite(y) & jnp.isfinite(velocity)
                             & jnp.isfinite(error) & (error >= 0))
        numpyro.factor("valid_observations", jnp.where(valid_data, 0., -jnp.inf))
        x, y, velocity = (jnp.where(jnp.isfinite(v), v, 0.) for v in (x, y, velocity))
        error = jnp.where(jnp.isfinite(error) & (error >= 0), error, 0.)
        params = self.sample_parameters()
        sigma2 = jnp.asarray(self.dsph_model.sigmalos2(x, y, params=params, **self.sigmalos2_kwargs))
        if sigma2.shape != x.shape:
            raise ValueError("sigmalos2 must return one variance per sky position")
        admissible = (jnp.isfinite(sigma2) & (sigma2 >= self.sigma2_bounds[0])
                      & (sigma2 <= self.sigma2_bounds[1]))
        numpyro.factor("valid_sigmalos2", jnp.where(jnp.all(admissible), 0., -jnp.inf))
        sigma2 = jnp.where(admissible, sigma2, 1.)
        loc = jnp.asarray(self._resolve_velocity_mean(params))
        if loc.ndim != 0 and loc.shape != x.shape:
            raise ValueError("velocity_mean must be scalar or match the observation shape")
        numpyro.factor("valid_velocity_mean", jnp.where(jnp.all(jnp.isfinite(loc)), 0., -jnp.inf))
        loc = jnp.where(jnp.isfinite(loc), loc, 0.)
        numpyro.sample(self.observed_name,
                       self.observation_distribution(loc, jnp.hypot(jnp.sqrt(sigma2), error)),
                       obs=velocity)


@dataclass(frozen=True)
class SamplerRunResult:
    r"""Report one NumPyroSampler run and persistence request.

    Notes
    -----
    **Inputs and units.** resumed and ``write_submitted`` are booleans;
    ``checkpoint_path``/``chunk_path`` are paths or None; ``chunk_index`` is an
    integer or None.

    **Returns and shape.** Frozen result record. ``write_submitted`` does not by
    itself confirm an asynchronous write finished.

    **Validity.** Inspect returned paths after flush/close before claiming
    durable completion.

    **Errors.** No numerical-domain validation.

    **Backend.** Python metadata.

    **Differentiation.** No physical-parameter automatic differentiation on this
    API.

    **Examples.** ``examples/docs_inference.py``
    """
    resumed: bool
    checkpoint_path: Path | None
    chunk_index: int | None
    chunk_path: Path | None
    write_submitted: bool


class NumPyroSampler:
    r"""Composition-based helper around numpyro.infer.MCMC.

    It keeps the wrapped ``MCMC`` instance untouched while adding:

    - automatic reuse of ``post_warmup_state`` for repeated runs,
    - checkpoint save/load of ``last_state``,
    - chunked ArviZ 1.0 persistence using backend-backed DataTree stores,
    - optional background writes for heavy output serialization.

    Notes
    -----
    **Inputs and units.** mcmc is numpyro.infer.MCMC; ``output_dir`` is a
    filesystem path; ``storage_backend`` chooses zarr/netcdf4/h5netcdf;
    ``async_writes`` toggles the writer; ``arviz_converter`` optionally returns
    xarray.DataTree. run(``rng_key``,\*args,\*\*kwargs) forwards data to the
    model, with explicit resume/save/write flags.

    **Returns and shape.** run returns SamplerRunResult. Samples use ArviZ
    chain/draw dimensions; ``load_samples(combine=True)`` returns a combined
    DataTree, False a list. ``save_samples_chunk`` returns
    (index,path,submitted). ``save_checkpoint``/``load_checkpoint`` return
    paths; flush/close return None after awaiting writes.

    **Validity.** Use a context manager or close/flush before relying on
    completed files. resume='auto' checks available state; resume=True requires
    valid state. Metadata binds data, prior, model/source/dependencies and
    effective runtime configuration. A checkpoint contains trusted Python
    serialization; open only your trusted analysis output.

    **Errors.** Missing/inconsistent analysis metadata or checkpoint identity
    raises before restarting. Write failures propagate from flush/close. Use a
    new directory for a changed analysis.

    **Backend.** NumPyro/JAX inference, host-side ArviZ storage.

    **Differentiation.** The model may be differentiated; sampling control, disk
    I/O and checkpoint state are not differentiable.

    **Examples.** ``examples/docs_numpyro_inference.py``
    """

    def __init__(
        self,
        mcmc: MCMC,
        *,
        output_dir: str | Path,
        storage_backend: StorageBackend | str = _DEFAULT_STORAGE_BACKEND,
        async_writes: bool = True,
        arviz_converter: Callable[..., xr.DataTree] | None = None,
    ) -> None:
        self.mcmc = mcmc
        self.output_dir = Path(output_dir)
        self.storage_backend = _normalize_storage_backend(storage_backend)
        self._storage_config = _STORAGE_CONFIGS[self.storage_backend]
        self.async_writes = bool(async_writes)
        self.arviz_converter = arviz_converter or az.from_numpyro
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="jeanspy-numpyro")
            if self.async_writes
            else None
        )
        self._write_futures: list[Future[Any]] = []
        self._futures_lock = Lock()
        self._chunk_index_lock = Lock()
        self._next_chunk_index = 0
        self._analysis_identity: dict[str, str] | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_metadata_file()
        self._next_chunk_index = self._discover_next_chunk_index()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.mcmc, name)

    def __enter__(self) -> "NumPyroSampler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def checkpoint_path(self) -> Path:
        """Return the Path for the trusted, pickled NumPyro transition-state checkpoint."""
        return self.output_dir / _CHECKPOINT_FILENAME

    @property
    def metadata_path(self) -> Path:
        """Return the Path for persisted format and sampling-identity metadata."""
        return self.output_dir / _METADATA_FILENAME

    @property
    def chunks_dir(self) -> Path:
        """Return the Path containing persisted sample chunks."""
        return self.output_dir / _CHUNKS_DIRNAME

    @property
    def chunk_suffix(self) -> str:
        """Return the filename suffix for the selected sample-storage backend."""
        return self._storage_config["suffix"]

    @property
    def uses_directory_stores(self) -> bool:
        """Whether the selected backend stores each sample chunk as a directory."""
        return bool(self._storage_config["directory_store"])

    def _read_metadata_file(self) -> dict[str, Any] | None:
        if not self.metadata_path.exists():
            return None
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _initialize_metadata_file(self) -> None:
        existing = self._read_metadata_file()
        if existing is not None:
            format_version = int(existing.get("storage_format_version", 0))
            if format_version != _STORAGE_FORMAT_VERSION:
                raise ValueError(
                    "Existing sampler output metadata uses an incompatible storage format version: "
                    f"{format_version}"
                )
            stored_backend = existing.get("storage_backend")
            if stored_backend != self.storage_backend:
                raise ValueError(
                    f"Output directory {self.output_dir} was initialized with storage_backend="
                    f"{stored_backend!r}, not {self.storage_backend!r}"
                )
            return

        payload = {
            "storage_format_version": _STORAGE_FORMAT_VERSION,
            "checkpoint_format_version": _CHECKPOINT_FORMAT_VERSION,
            "storage": "arviz-datatree-store-per-chunk",
            "storage_backend": self.storage_backend,
            "chunk_suffix": self.chunk_suffix,
            "created_at": _utc_now_iso(),
        }
        tmp_path = self.metadata_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(self.metadata_path)

    def _discover_next_chunk_index(self) -> int:
        chunk_paths = self.list_chunk_paths()
        if not chunk_paths:
            return 0
        return max(self._parse_chunk_index(path) for path in chunk_paths) + 1

    def _reserve_chunk_index(self) -> int:
        with self._chunk_index_lock:
            chunk_index = self._next_chunk_index
            self._next_chunk_index += 1
        return chunk_index

    @staticmethod
    def _parse_chunk_index_from_name(name: str, suffix: str) -> int:
        base_name = name.removeprefix(_CHUNK_PREFIX)
        if suffix:
            if not base_name.endswith(suffix):
                raise ValueError(f"Chunk name {name!r} does not end with suffix {suffix!r}")
            base_name = base_name[: -len(suffix)]
        return int(base_name)

    def _parse_chunk_index(self, chunk_path: Path) -> int:
        return self._parse_chunk_index_from_name(chunk_path.name, self.chunk_suffix)

    def _chunk_name(self, chunk_index: int) -> str:
        return f"{_CHUNK_PREFIX}{chunk_index:04d}{self.chunk_suffix}"

    def list_chunk_paths(self) -> list[Path]:
        r"""Find sample chunks in index order.

        Notes
        -----
        **Inputs and units.** No arguments.

        **Returns and shape.** List of Paths; an empty list means no saved chunks
        were found.

        **Validity.** Lists recognized chunk names in this directory; it does not
        inspect their scientific content or verify identity.

        **Errors.** Filesystem access errors may propagate; an empty directory
        returns an empty list.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        if not self.chunks_dir.exists():
            return []
        chunk_paths = [
            path
            for path in self.chunks_dir.iterdir()
            if path.name.startswith(_CHUNK_PREFIX)
            and path.name.endswith(self.chunk_suffix)
            and ((self.uses_directory_stores and path.is_dir()) or ((not self.uses_directory_stores) and path.is_file()))
        ]
        return sorted(chunk_paths, key=self._parse_chunk_index)

    def _prune_successful_writes_locked(self) -> None:
        self._write_futures = [
            future for future in self._write_futures
            if not future.done() or future.cancelled() or future.exception() is not None
        ]

    def pending_write_count(self) -> int:
        """Count unfinished writes, retaining failures for flush/close."""
        with self._futures_lock:
            self._prune_successful_writes_locked()
            return sum(not future.done() for future in self._write_futures)

    def flush(self) -> None:
        r"""Wait for outstanding sample writes.

        Notes
        -----
        **Inputs and units.** No arguments.

        **Returns and shape.** None; asynchronous exceptions propagate.

        **Validity.** Waits for outstanding writes and propagates their failures; it
        does not change the MCMC target or validate a checkpoint.

        **Errors.** A pending write's exception is re-raised after waiting for the
        submitted futures.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        with self._futures_lock:
            futures = list(self._write_futures)
            self._write_futures.clear()
        error = None
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def close(self) -> None:
        r"""Finish writes and shut down the background writer.

        Notes
        -----
        **Inputs and units.** No arguments; also called by context-manager exit.

        **Returns and shape.** None; failures propagate.

        **Validity.** Finish pending writes before releasing the executor. Prefer a
        context manager to ensure this happens.

        **Errors.** Write failures propagate after the executor is shut down.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        try:
            self.flush()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    def clear_resume_state(self) -> None:
        r"""Clear the in-memory warmup-resume pointer.

        Notes
        -----
        **Inputs and units.** No arguments.

        **Returns and shape.** None; sets ``mcmc.post_warmup_state = None``. It does
        not delete or reset persisted analysis files.

        **Validity.** Affects only the in-memory ``post_warmup_state`` pointer. A
        later auto-resume can still use ``last_state`` or an on-disk checkpoint.

        **Errors.** No additional validation or documented domain exception.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        self.mcmc.post_warmup_state = None

    def _target_fingerprint(self) -> str:
        kernel = self.mcmc.sampler
        # Generated potential/postprocessing functions and compilation caches
        # change after warmup. Hash user inputs and transition configuration.
        settings = ("_kinetic_fn", "_num_steps", "_step_size", "_inverse_mass_matrix",
                    "_adapt_step_size", "_adapt_mass_matrix", "_dense_mass",
                    "_target_accept_prob", "_trajectory_length", "_algo", "_max_tree_depth",
                    "_init_strategy", "_find_heuristic_step_size", "_forward_mode_differentiation",
                    "_regularize_mass_matrix", "_moves", "_weights", "_randomize_split")
        model = getattr(kernel, '_model', None)
        return fingerprint({
            'software': software_identity('jax', 'jaxlib', 'numpyro'),
            'x64': bool(jax.config.jax_enable_x64),
            # Jeans defaults differ between CPU and GPU float32 execution.
            'platform': jax.default_backend(),
            'matmul_precision': jax.config.jax_default_matmul_precision,
            'default_dtype_bits': getattr(jax.config, 'jax_default_dtype_bits', None),
            'model': model if model is not None else getattr(kernel, '_potential_fn', None),
            'kernel': type(kernel),
            'kernel_config': {k: getattr(kernel, k) for k in settings if hasattr(kernel, k)},
            'num_chains': self.mcmc.num_chains,
            'chain_method': self.mcmc.chain_method,
            'postprocess_fn': self.mcmc.postprocess_fn,
        })

    def _bind_analysis(self, args, kwargs) -> None:
        if (self._analysis_identity is None and
                (getattr(self.mcmc, 'last_state', None) is not None or
                 getattr(self.mcmc, 'post_warmup_state', None) is not None)):
            raise ValueError("Unverified in-memory MCMC state; construct a fresh MCMC instance "
                             "and use the sampler's verified checkpoint to resume")
        model_kwargs = {k: v for k, v in kwargs.items() if k not in {'extra_fields', 'init_params'}}
        identity = {'target': self._target_fingerprint(), 'arguments': fingerprint((args, model_kwargs))}
        metadata = self._read_metadata_file()
        if metadata is None:
            raise ValueError(
                "Sampler metadata is missing; cannot verify analysis identity. "
                "Restore the original metadata.json or use a new output_dir."
            )
        stored = metadata.get('analysis_identity')
        for previous in (self._analysis_identity, stored):
            if previous is not None and previous != identity:
                raise ValueError("Sampling analysis identity mismatch (model, prior, data, schema, or solver). "
                                 "Use a new output_dir for a different analysis, even with resume=False.")
        if stored is None:
            if self.checkpoint_path.exists() or self.list_chunk_paths():
                raise ValueError("Existing output has no analysis identity; use a new output_dir. "
                                 "Legacy chains cannot be resumed safely.")
            metadata['analysis_identity'] = identity
            tmp_path = self.metadata_path.with_suffix('.tmp')
            tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')
            tmp_path.replace(self.metadata_path)
        self._analysis_identity = identity

    def save_checkpoint(self) -> Path:
        r"""Save trusted NumPyro transition state.

        Notes
        -----
        **Inputs and units.** No arguments; requires a ``last_state`` and bound
        analysis identity.

        **Returns and shape.** Path of the written checkpoint.

        **Validity.** A completed MCMC ``last_state`` and an unchanged bound
        analysis are required. The file contains Python pickle data.

        **Errors.** RuntimeError before ``last_state`` exists; ValueError for
        unverified/changed analysis. Filesystem errors propagate.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        last_state = getattr(self.mcmc, "last_state", None)
        if last_state is None:
            raise RuntimeError("Cannot save checkpoint before MCMC has produced last_state")
        if self._analysis_identity is None or self._analysis_identity['target'] != self._target_fingerprint():
            raise ValueError("Cannot checkpoint an unverified or changed analysis; use NumPyroSampler.run")

        payload = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "saved_at": _utc_now_iso(),
            "last_state": _to_host_tree(last_state),
            "analysis_identity": self._analysis_identity,
        }
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.checkpoint_path)
        self.mcmc.post_warmup_state = last_state
        return self.checkpoint_path

    def load_checkpoint(self) -> Path:
        r"""Load a trusted matching transition state.

        Notes
        -----
        **Inputs and units.** No arguments; reads this output directory's checkpoint
        and metadata.

        **Returns and shape.** Checkpoint Path; sets ``post_warmup_state`` only
        after identity checks.

        **Validity.** Read only a trusted checkpoint. Stored target/metadata
        identity is checked here; supplied observation arguments are checked later
        by run.

        **Errors.** FileNotFoundError for a missing checkpoint; ValueError for
        format/identity mismatch. Pickle and filesystem errors propagate.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        with self.checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)

        format_version = int(payload.get("format_version", 0))
        if format_version != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version {format_version}; expected {_CHECKPOINT_FORMAT_VERSION}"
            )

        identity = payload.get('analysis_identity')
        stored = (self._read_metadata_file() or {}).get('analysis_identity')
        if (not identity or identity != stored or identity['target'] != self._target_fingerprint()
                or (self._analysis_identity is not None and identity != self._analysis_identity)):
            raise ValueError("Checkpoint analysis identity mismatch; use a new output_dir for a different analysis")
        self._analysis_identity = identity
        self.mcmc.post_warmup_state = _to_device_tree(payload["last_state"])
        return self.checkpoint_path

    def _prepare_resume_state(self, resume: bool | str) -> bool:
        if resume not in {True, False, "auto"}:
            raise ValueError("resume must be True, False, or 'auto'")

        if resume is False:
            self.clear_resume_state()
            return False

        if getattr(self.mcmc, "post_warmup_state", None) is not None:
            return True

        last_state = getattr(self.mcmc, "last_state", None)
        if last_state is not None:
            self.mcmc.post_warmup_state = last_state
            return True

        if self.checkpoint_path.exists():
            self.load_checkpoint()
            return True

        if resume is True:
            raise FileNotFoundError("resume=True but no in-memory or on-disk checkpoint is available")
        return False

    def to_datatree(self, **arviz_kwargs: Any) -> xr.DataTree:
        r"""Convert the wrapped MCMC result to ArviZ storage.

        Notes
        -----
        **Inputs and units.** Keyword arguments forwarded to ``arviz_converter``.

        **Returns and shape.** xarray.DataTree with chain/draw dimensions; the
        converter must return this format.

        **Validity.** The wrapped MCMC must contain a result supported by the chosen
        converter.

        **Errors.** TypeError if the converter returns anything other than
        xarray.DataTree; converter exceptions propagate.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        datatree = self.arviz_converter(self.mcmc, **arviz_kwargs)
        if not isinstance(datatree, xr.DataTree):
            raise TypeError("Expected ArviZ converter to return xarray.DataTree")
        return datatree

    def _write_chunk_store(self, chunk_index: int, datatree: xr.DataTree) -> Path:
        chunk_path = self.chunks_dir / self._chunk_name(chunk_index)
        if chunk_path.exists():
            raise FileExistsError(f"Chunk already exists: {chunk_path}")

        tmp_parent = Path(tempfile.mkdtemp(prefix=f".{chunk_path.stem}.", dir=self.chunks_dir))
        tmp_path = tmp_parent / chunk_path.name
        try:
            if self._storage_config["writer"] == "zarr":
                datatree.to_zarr(tmp_path, mode="w", consolidated=False)
            else:
                netcdf_engine = cast(Literal["h5netcdf", "netcdf4"], self.storage_backend)
                datatree.to_netcdf(tmp_path, mode="w", engine=netcdf_engine)
            tmp_path.replace(chunk_path)
        except Exception:
            _remove_path(tmp_path)
            raise
        finally:
            try:
                datatree.close()
            except Exception:
                pass
            _remove_path(tmp_parent)

        return chunk_path

    def save_samples_chunk(
        self,
        *,
        datatree: xr.DataTree | None = None,
        wait: bool = False,
        **arviz_kwargs: Any,
    ) -> tuple[int, Path, bool]:
        r"""Write an ArviZ sample chunk.

        Notes
        -----
        **Inputs and units.** Optional datatree; wait=True writes synchronously;
        other keywords go to the converter.

        **Returns and shape.** Tuple (``chunk_index``, path, ``write_submitted``).
        False in the third entry means the write was synchronous.

        **Validity.** The sampler assigns a fresh chunk index. Wait for flush/close
        before relying on an asynchronous write.

        **Errors.** FileExistsError prevents replacing a reserved chunk. Conversion,
        array loading and write failures propagate; asynchronous failures are raised
        by flush/close.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        prepared_tree = datatree if datatree is not None else self.to_datatree(**arviz_kwargs)
        prepared_tree = prepared_tree.load()
        chunk_index = self._reserve_chunk_index()
        chunk_path = self.chunks_dir / self._chunk_name(chunk_index)

        if self._executor is None or wait:
            self._write_chunk_store(chunk_index, prepared_tree)
            return chunk_index, chunk_path, False

        future = self._executor.submit(self._write_chunk_store, chunk_index, prepared_tree)
        with self._futures_lock:
            self._prune_successful_writes_locked()
            self._write_futures.append(future)
        return chunk_index, chunk_path, True

    def _load_chunk_tree(self, chunk_path: Path) -> xr.DataTree:
        reader = self._storage_config["reader"]
        reader_kwargs = dict(self._storage_config["reader_kwargs"])
        datatree = reader(chunk_path, **reader_kwargs)
        try:
            return datatree.load()
        finally:
            datatree.close()

    def load_samples(self, *, combine: bool = True) -> xr.DataTree | list[xr.DataTree]:
        r"""Read persisted sample chunks.

        Notes
        -----
        **Inputs and units.** combine=True concatenates matching draw dimensions;
        False retains separate trees.

        **Returns and shape.** Combined xarray.DataTree or ordered list of
        DataTrees. Pending writes are flushed first.

        **Validity.** Reads this directory's chunks. Completed arrays are loaded
        into memory before closing backing stores.

        **Errors.** FileNotFoundError when no chunks exist. Pending-write, reader
        and concatenation errors propagate.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        self.flush()
        chunk_paths = self.list_chunk_paths()
        if not chunk_paths:
            raise FileNotFoundError(f"No saved chunks found in {self.chunks_dir}")

        trees = [self._load_chunk_tree(chunk_path) for chunk_path in chunk_paths]
        if not combine:
            return trees
        return self.combine_trees(trees)

    @staticmethod
    def combine_trees(trees: Sequence[xr.DataTree]) -> xr.DataTree:
        r"""Combine compatible ArviZ chunks.

        Notes
        -----
        **Inputs and units.** Nonempty sequence of DataTrees from the same analysis,
        with consistent groups, variables and nondraw coordinates.

        **Returns and shape.** DataTree concatenated over draws, with draw
        coordinates renumbered from zero. No independent sampling-identity
        verification occurs here.

        **Validity.** Supply chunks from the same analysis. Static groups must be
        equal. The draw-group concatenation uses xarray compat=override and is not a
        general identity validator.

        **Errors.** ValueError for an empty list, missing groups, inconsistent
        draw-axis presence or changed static groups. Other xarray errors propagate.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        if not trees:
            raise ValueError("trees must contain at least one DataTree")

        if len(trees) == 1:
            return trees[0]

        all_groups = set()
        for tree in trees:
            all_groups.update(group for group in tree.groups if group != "/")

        combined_mapping: dict[str, xr.Dataset] = {}
        for group_path in sorted(all_groups):
            datasets = []
            for tree in trees:
                if group_path not in tree.groups:
                    raise ValueError(f"Group {group_path} is missing from one of the chunks")
                datasets.append(tree[group_path].dataset)
            combined_mapping[group_path.lstrip("/")] = _concat_draw_datasets(group_path, datasets)

        return xr.DataTree.from_dict(combined_mapping)

    def run(
        self,
        rng_key: Any,
        *args: Any,
        resume: bool | str = "auto",
        save_checkpoint: bool = True,
        save_samples: bool = True,
        wait_for_write: bool = False,
        arviz_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> SamplerRunResult:
        r"""Run one chunk and optionally persist it.

        Notes
        -----
        **Inputs and units.** ``rng_key`` plus model data arguments; resume is
        auto/True/False. ``save_checkpoint``/``save_samples``/``wait_for_write``
        control persistence; ``arviz_kwargs`` are conversion options.

        **Returns and shape.** SamplerRunResult; wait for flush/close before relying
        on asynchronous files.

        **Validity.** Use an unchanged model, prior, observations and numerical
        configuration when resuming. Identity is checked before sampling.

        **Errors.** ValueError for identity mismatch or invalid resume mode;
        FileNotFoundError for required missing state. Model, NumPyro and storage
        exceptions propagate.

        **Differentiation.** Host-side sampling/storage control; this method has no
        physical-parameter derivative.
        """
        self._bind_analysis(args, kwargs)
        resumed = self._prepare_resume_state(resume)
        self.mcmc.run(rng_key, *args, **kwargs)

        checkpoint_path: Path | None = None
        last_state = getattr(self.mcmc, "last_state", None)
        if last_state is not None:
            self.mcmc.post_warmup_state = last_state
            if save_checkpoint:
                checkpoint_path = self.save_checkpoint()

        chunk_index: int | None = None
        chunk_path: Path | None = None
        write_submitted = False
        if save_samples:
            chunk_index, chunk_path, write_submitted = self.save_samples_chunk(
                wait=wait_for_write,
                **dict(arviz_kwargs or {}),
            )

        return SamplerRunResult(
            resumed=resumed,
            checkpoint_path=checkpoint_path,
            chunk_index=chunk_index,
            chunk_path=chunk_path,
            write_submitted=write_submitted,
        )


__all__ = [
    "AxisymmetricJeansLikelihoodModel",
    "JeansLikelihoodModel",
    "NumPyroSampler",
    "ParameterSpec",
    "SamplerRunResult",
    "StorageBackend",
]
