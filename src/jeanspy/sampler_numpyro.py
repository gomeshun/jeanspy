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


logger = logging.getLogger(__name__)


StorageBackend = Literal["zarr", "h5netcdf", "netcdf4"]


_CHECKPOINT_FORMAT_VERSION = 1
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
    """Describe one NumPyro parameter site and its physical representation."""

    sample_name: str
    distribution: Any
    param_name: str | None = None
    transform: Callable[[Any], Any] | None = None
    record_deterministic: bool | None = None
    deterministic_name: str | None = None

    @classmethod
    def exp(
        cls,
        sample_name: str,
        distribution: Any,
        *,
        param_name: str | None = None,
        deterministic_name: str | None = None,
    ) -> "ParameterSpec":
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
        return cls(
            sample_name=sample_name,
            distribution=distribution,
            param_name=param_name,
            transform=_pow10,
            record_deterministic=True,
            deterministic_name=deterministic_name,
        )

    def build_distribution(self) -> Any:
        if isinstance(self.distribution, dist.Distribution):
            return self.distribution
        return self.distribution() if callable(self.distribution) else self.distribution

    def sample(self) -> tuple[str, Any]:
        raw_value = numpyro.sample(self.sample_name, self.build_distribution())
        resolved_name = self.param_name or self.sample_name
        value = self.transform(raw_value) if self.transform is not None else raw_value

        record_deterministic = self.record_deterministic
        if record_deterministic is None:
            record_deterministic = self.transform is not None or resolved_name != self.sample_name

        if self.deterministic_name is not None or record_deterministic:
            numpyro.deterministic(self.deterministic_name or resolved_name, value)

        return resolved_name, value


class JeansLikelihoodModel:
    """Callable NumPyro model for line-of-sight velocity inference."""

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
        self.sigmalos2_kwargs = dict(sigmalos2_kwargs or {})
        self.sigma2_bounds = (float(sigma2_bounds[0]), float(sigma2_bounds[1]))
        self.velocity_mean = velocity_mean
        self.observation_distribution = observation_distribution
        self.observed_name = observed_name
        self.parameter_postprocess = parameter_postprocess

    def sample_parameters(self) -> dict[str, Any]:
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
        params = self.sample_parameters()
        sigma2 = self.dsph_model.sigmalos2(jnp.asarray(R_pc), params=params, **self.sigmalos2_kwargs)
        sigma2 = jnp.clip(sigma2, min=self.sigma2_bounds[0], max=self.sigma2_bounds[1])
        scale = jnp.sqrt(sigma2 + jnp.asarray(e_vlos_kms) ** 2)
        loc = self._resolve_velocity_mean(params)
        numpyro.sample(
            self.observed_name,
            self.observation_distribution(loc, scale),
            obs=jnp.asarray(vlos_kms),
        )


@dataclass(frozen=True)
class SamplerRunResult:
    resumed: bool
    checkpoint_path: Path | None
    chunk_index: int | None
    chunk_path: Path | None
    write_submitted: bool


class NumPyroSampler:
    """Composition-based helper around numpyro.infer.MCMC.

    It keeps the wrapped ``MCMC`` instance untouched while adding:

    - automatic reuse of ``post_warmup_state`` for repeated runs,
    - checkpoint save/load of ``last_state``,
    - chunked ArviZ 1.0 persistence using backend-backed DataTree stores,
    - optional background writes for heavy output serialization.
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
        return self.output_dir / _CHECKPOINT_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / _METADATA_FILENAME

    @property
    def chunks_dir(self) -> Path:
        return self.output_dir / _CHUNKS_DIRNAME

    @property
    def chunk_suffix(self) -> str:
        return self._storage_config["suffix"]

    @property
    def uses_directory_stores(self) -> bool:
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

    def pending_write_count(self) -> int:
        with self._futures_lock:
            self._write_futures = [future for future in self._write_futures if not future.done()]
            return len(self._write_futures)

    def flush(self) -> None:
        with self._futures_lock:
            futures = list(self._write_futures)
            self._write_futures.clear()
        for future in futures:
            future.result()

    def close(self) -> None:
        self.flush()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def clear_resume_state(self) -> None:
        self.mcmc.post_warmup_state = None

    def save_checkpoint(self) -> Path:
        last_state = getattr(self.mcmc, "last_state", None)
        if last_state is None:
            raise RuntimeError("Cannot save checkpoint before MCMC has produced last_state")

        payload = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "saved_at": _utc_now_iso(),
            "last_state": _to_host_tree(last_state),
        }
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.checkpoint_path)
        self.mcmc.post_warmup_state = last_state
        return self.checkpoint_path

    def load_checkpoint(self) -> Path:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        with self.checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)

        format_version = int(payload.get("format_version", 0))
        if format_version != _CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint format version {format_version}; expected {_CHECKPOINT_FORMAT_VERSION}"
            )

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
        prepared_tree = datatree if datatree is not None else self.to_datatree(**arviz_kwargs)
        prepared_tree = prepared_tree.load()
        chunk_index = self._reserve_chunk_index()
        chunk_path = self.chunks_dir / self._chunk_name(chunk_index)

        if self._executor is None or wait:
            self._write_chunk_store(chunk_index, prepared_tree)
            return chunk_index, chunk_path, False

        future = self._executor.submit(self._write_chunk_store, chunk_index, prepared_tree)
        with self._futures_lock:
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
    "JeansLikelihoodModel",
    "NumPyroSampler",
    "ParameterSpec",
    "SamplerRunResult",
    "StorageBackend",
]