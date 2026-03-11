# JeansPy
JeansPy is a Python library for the Jeans analysis. This library is designed to help researchers and astrophysicists analyze and understand the dynamics of a given system.

## Features
- Calculation of velocity dispersion profile using Jeans equations
- Visualization of the results using Matplotlib

## Installation

Create a local environment with `uv`:

```bash
uv sync
```

This repository treats `pyproject.toml` as the authoritative environment definition. The command above creates `.venv`, installs the package in editable mode, and resolves the full development and inference stack into `uv.lock`.

If you need to refresh the environment after changing dependencies, run:

```bash
uv sync
```

For GPU execution with JAX/NumPyro, the editable install currently targets the CUDA 12 plugin line. If `jax.default_backend()` fails with a driver mismatch, verify that your NVIDIA driver supports the CUDA runtime required by the installed JAX plugin.

If you prefer a pip-based install instead of `uv`, run:

```bash
pip install -r requirements.txt
pip install -e .
```

The repository follows a standard `src` layout:

- `src/jeanspy/`: installable Python package
- `tests/`: test suite
- `scripts/`: standalone analysis and plotting scripts
- `notebooks/`: exploratory notebooks
- `examples/`: sample inputs and example notebooks

## Build And Publish

Build a source distribution and wheel from the repository root:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

If the generated artifacts are valid, upload them to PyPI:

```bash
python -m twine upload dist/*
```

## Usage

```python
import jeanspy as jpy

# define your model or load the preset model
mdl = jpy.model.get_default_estimation_model(
    dsph_type="Classical",  # "Classical" or "UFD"
    dsph_name="Sculptor",
    config="priorconfig.csv",
)

# run the estimation
sampler = jpy.sampler.Sampler(mdl)
```

## JAX Runtime Configuration

The NumPyro/JAX implementation in [src/jeanspy/model_numpyro.py](src/jeanspy/model_numpyro.py) can be configured through a small set of environment variables before import:

- `JEANSPY_JAX_PLATFORM=cpu|gpu`: request the JAX platform. On CUDA installs, `gpu` is mapped to JAX's `cuda` backend automatically.
- `JEANSPY_JAX_ENABLE_X64=true|false`: choose float64 or float32 execution.
- `JEANSPY_HYP2F1_BACKEND=scipy|jax`: choose the constant-anisotropy kernel backend.
- `JEANSPY_CONSTANT_KERNEL_N_QUAD=<int>`: override the constant-anisotropy kernel quadrature order.
- `JEANSPY_SIGMALOS2_BACKEND=auto|kernel|abel`: choose the default `sigmalos2` solver.
- `JEANSPY_SIGMALOS2_JIT=auto|true|false`: control the cached `jax.jit` wrapper around `sigmalos2`.
- `JEANSPY_SIGMALOS2_N_U=<int>`: override the kernel-solver log-`u` grid size.
- `JEANSPY_SIGMALOS2_N_R=<int>`: override the Abel-solver radial grid size.
- `JEANSPY_SIGMALOS2_U_MAX=<float>`: override the default outer integration limit.

On GPU float32, `model_numpyro` now chooses more aggressive defaults automatically for the constant-anisotropy path so that the hot JIT-compiled solver stays faster than `model.py` while landing near the `1e-5` relative-accuracy range on the benchmark cases used during development.

For interactive sessions, the same module exposes lightweight runtime helpers:

```python
from jeanspy.model_numpyro import configure_runtime, get_runtime_config

configure_runtime(
    hyp2f1_backend="jax",
    sigmalos2_backend="kernel",
    sigmalos2_jit=True,
    sigmalos2_n_u=12289,
    sigmalos2_u_max=5000.0,
    constant_kernel_n_quad=64,
    jax_enable_x64=False,
)

print(get_runtime_config())
```

To reproduce the backend and precision comparison used during development, run:

```bash
python scripts/compare_runtime_modes.py
```

## License

JeansPy is distributed under the terms of the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
