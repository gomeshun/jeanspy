"""Fixed quadrature and a specialized JAX kernel reference check."""
import numpy as np
from jeanspy.dequad import dequad, generate_x_w

# quadrature-start
def integrand(x):
    return np.exp(-x)

value = dequad(integrand, 0., np.inf, n=128)
np.testing.assert_allclose(value, 1., rtol=1e-8)
nodes, weights = generate_x_w(0., 1., 128)
np.testing.assert_allclose(np.sum(weights * nodes**2), 1/3, rtol=1e-8)
# These fixed rules provide no automatic error estimate. Compare node counts.
# quadrature-end

# special-functions-start
import jax
import jax.numpy as jnp
from scipy.special import hyp2f1
from jeanspy.hyp2f1_jax import hyp2f1_1b_3half
from jeanspy.baes_eta2 import (baes_eta2_kernel_jax,
                               baes_eta2_kernel_appell_reference)

w = jnp.array([.05, .2, .5])
values = jax.block_until_ready(hyp2f1_1b_3half(.8, w))
np.testing.assert_allclose(values, hyp2f1(1., .8, 1.5, np.asarray(w)), rtol=1e-5)
u, R_pc = np.array([1.1, 1.5]), np.array([.7, 2.3])
kernel = baes_eta2_kernel_jax(u, R_pc, -.5, .7, 1.4, n_kernel=128)
reference = baes_eta2_kernel_appell_reference(u, R_pc, -.5, .7, 1.4, dps=25)
np.testing.assert_allclose(kernel, reference, rtol=8e-4, atol=2e-6)
# This small reference domain is not a universal prior-domain accuracy claim.
# special-functions-end
