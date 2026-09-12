"""Axisymmetric prediction and factor examples, executed by documentation CI."""
# example-start
import numpy as np
from jeanspy.axisymmetric import AxisymmetricDSphModel

params = dict(re_pc=300., rs_pc=500., rhos_Msunpc3=.1,
              q=.7, Q=.8, alpha=2., beta=3., gamma=.5,
              beta_z=-.3, inclination=1.1, r_t_pc=3000.)
model = AxisymmetricDSphModel(48, 48, 48)
x_pc = np.array([0., 100., -300.])
y_pc = np.array([0., 50., 70.])
variance = model.sigmalos2(x_pc, y_pc, params=params)
assert variance.shape == x_pc.shape
assert np.all(np.isfinite(variance) & (variance > 0))
print(np.sqrt(variance))
# example-end

# factors-start
# Use refined angular/radial settings for a scientific posterior summary.
settings = dict(n_mu=24, n_phi=24, n_radial=64)
J = model.jfactor(80000., .5, params=params, **settings)
D = model.dfactor(80000., .5, params=params, **settings)
assert np.isfinite(J) and J > 0
assert np.isfinite(D) and D > 0
print(J, D)  # GeV^2 cm^-5, GeV cm^-2
# factors-end
