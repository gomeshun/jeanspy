# JeansPy
JeansPy is a Python library for the Jeans analysis. This library is designed to help researchers and astrophysicists analyze and understand the dynamics of a given system.

## Features
- Calculation of velocity dispersion profile using Jeans equations
- Visualization of the results using Matplotlib

## Installation

Install the package with pip:

```bash
pip install jeanspy
```

This repository treats `requirements.txt` as the authoritative full environment definition, including development, visualization, and inference dependencies. To reproduce the working environment used in this repository, run:

```bash
pip install -r requirements.txt
pip install -e .
```

For GPU execution with JAX/NumPyro, the editable install currently targets the CUDA 12 plugin line. If `jax.default_backend()` fails with a driver mismatch, verify that your NVIDIA driver supports the CUDA runtime required by the installed JAX plugin.

For the development version, clone this repository and run:

```bash
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

## License

JeansPy is distributed under the terms of the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
