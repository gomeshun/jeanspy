# JeansPy
JeansPy is a Python library for the Jeans analysis. This library is designed to help researchers and astrophysicists analyze and understand the dynamics of a given system.


## Features
- Calculation of velocity dispersion profile using Jeans equations
- Visualization of the results using Matplotlib


## Installation


## Usage

```python
import jeanspy as jpy

# define your model or load preset model
mdl = jpy.model.get_default_estimation_model(
    dsph_type = "Classical",  # "Classical" or "UFD"
    dsph_name = "Sculptor",
    config = "priorconfig.csv"
)

# run the estimation
sampler = jpy.sampler.Sampler(mdl)

```


## Licence
