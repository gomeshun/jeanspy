"""A synchronized JAX prediction and physical-parameter derivative."""
# example-start
# Start Python with JEANSPY_JAX_ENABLE_X64=true for reference precision.
import jax
import jax.numpy as jnp
import numpy as np
from jeanspy.axisymmetric_numpyro import AxisymmetricDSphModel

model = AxisymmetricDSphModel(24, 24, 24)
fixed = dict(re_pc=300., rs_pc=500., q=.7, Q=.8,
             alpha=2., beta=3., gamma=.5, beta_z=-.3, inclination=1.1)

@jax.jit
def prediction(rho):
    return model.sigmalos2(jnp.array([100., 300.]), 50.,
                          params={**fixed, "rhos_Msunpc3": rho})

rho = .1
variance = jax.block_until_ready(prediction(rho))
derivative = jax.block_until_ready(jax.jacfwd(prediction)(rho))
# Gravity and second moments scale linearly with density at fixed geometry.
np.testing.assert_allclose(derivative, variance / rho, rtol=1e-5)
print(variance, derivative)
# example-end
