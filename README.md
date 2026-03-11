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

The NumPyro/JAX implementation in [src/jeanspy/model_numpyro.py](src/jeanspy/model_numpyro.py) keeps only process-wide JAX settings in environment variables before import:

- `JEANSPY_JAX_PLATFORM=cpu|gpu`: request the JAX platform. On CUDA installs, `gpu` is mapped to JAX's `cuda` backend automatically.
- `JAX_ENABLE_X64=true|false`: choose float64 or float32 execution.

The numerical knobs that only affect a specific solver call are now explicit method arguments instead of environment variables. For example:

```python
from jeanspy.model_numpyro import ConstantAnisotropyModel, DSphModel, NFWModel, PlummerModel

dsph = DSphModel(
    submodels={
        "StellarModel": PlummerModel(),
        "DMModel": NFWModel(),
        "AnisotropyModel": ConstantAnisotropyModel(),
    }
)

sigma2 = dsph.sigmalos2(
    R_pc,
    params=params,
    backend="kernel",
    jit=True,
    n_u=1024,
    u_max=5000.0,
    constant_kernel_backend="jax",
    n_kernel=64,
)
```

For direct constant-anisotropy kernel comparisons, choose the backend per call:

```python
kernel = ConstantAnisotropyModel().kernel(
    u,
    R_pc,
    params={"beta_ani": 0.5},
    backend="scipy",
)
```

On GPU float32, `model_numpyro` defaults to `n_u=1024` for the kernel solver. That keeps the grid much smaller than the previous development setting while still staying in the sub-`1e-3` relative-error range on the representative constant-anisotropy benchmark cases used during development. If you need tighter agreement with the high-resolution reference, raise `n_u` explicitly on the relevant `sigmalos2()` call.

To reproduce the backend and precision comparison used during development, run:

```bash
python scripts/compare_runtime_modes.py
```

## License

JeansPy is distributed under the terms of the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
