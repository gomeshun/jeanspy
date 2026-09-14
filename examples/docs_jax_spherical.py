"""Functional spherical JAX components and an explicit forward configuration."""
import jax
import jax.numpy as jnp
import numpy as np
from jeanspy.model_numpyro import (BaesAnisotropyModel, ConstantAnisotropyModel,
    DSphModel, NFWModel, OsipkovMerrittModel, PlummerModel, ZhaoModel,
    get_runtime_config)
from jeanspy.baes_eta2 import BaesEta2AnisotropyModel

# spherical-jax-start
tracer, halo, anisotropy = PlummerModel(), NFWModel(), ConstantAnisotropyModel()
model = DSphModel(submodels={"StellarModel": tracer, "DMModel": halo,
                             "AnisotropyModel": anisotropy})
params = dict(re_pc=300., rs_pc=500., rhos_Msunpc3=.1,
              r_t_pc=5000., beta_ani=-.2)
R_pc = jnp.array([30., 100., 300.])
variance = jax.block_until_ready(model.sigmalos2(
    R_pc, params=params, backend="kernel", n_u=128, n_kernel=64))
assert variance.shape == (3,) and np.all(np.asarray(variance) > 0)
sampled_R = tracer.sample_R(jax.random.PRNGKey(20260913), 8, re_pc=300.)
assert sampled_R.shape == (8,)
print(get_runtime_config(), variance)
# spherical-jax-end

# functional-profiles-start
zhao = ZhaoModel()
mass = zhao.enclosed_mass(R_pc, params={**params, "a": 2., "b": 4., "g": .5},
                          method="numeric", n_steps=128)
assert np.all(np.diff(np.asarray(mass)) > 0)
for component, p in [
    (OsipkovMerrittModel(), {"r_a": 1000.}),
    (BaesAnisotropyModel(), {"r_a": 1000., "beta_0": -.2, "beta_inf": .5, "eta": 2.}),
    (BaesEta2AnisotropyModel(), {"r_a": 1000., "beta_0": -.2, "beta_inf": .5}),
]:
    assert np.all(np.asarray(component.beta(R_pc, params=p)) < 1)
# functional-profiles-end
