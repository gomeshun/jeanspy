"""JeansPy package initialization."""

import importlib

__version__ = "0.1.0"

__all__ = [
    "coord",
    "dequad",
    "model",
    "jfactor",
    "polygon",
    "sampler",
    "swyft_legacy",
    "cmd_utilities",
]

_OPTIONAL_MODULES = {
    "model_numpyro": ("numpyro_cpu", "numpyro_cuda12"),
    "sampler_numpyro": ("numpyro_cpu", "numpyro_cuda12"),
}

_LAZY_MODULES = set(__all__) | set(_OPTIONAL_MODULES)
_OPTIONAL_IMPORT_ROOTS = {
    "arviz",
    "h5netcdf",
    "h5py",
    "jax",
    "jaxlib",
    "netCDF4",
    "numpyro",
    "xarray",
    "zarr",
}


def _format_optional_install_hint(extras):
    if isinstance(extras, str):
        return f"jeanspy[{extras}]"
    return " or ".join(f"jeanspy[{extra}]" for extra in extras)

def __getattr__(name):
    if name in _LAZY_MODULES:
        try:
            module = importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as exc:
            root_name = (exc.name or "").split(".")[0]
            if name in _OPTIONAL_MODULES and root_name in _OPTIONAL_IMPORT_ROOTS:
                extras = _OPTIONAL_MODULES[name]
                raise ImportError(
                    f"{name} requires optional NumPyro/JAX dependencies. "
                    f"Install {_format_optional_install_hint(extras)} to use this module."
                ) from exc
            raise
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
