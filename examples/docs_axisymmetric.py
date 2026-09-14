"""Axisymmetric prediction and factor examples, executed by documentation CI."""
# example-start
import numpy as np
from jeanspy.model import (
    AxisymmetricDSphModel, AxisymmetricPlummerModel,
    AxisymmetricZhaoModel, AxisymmetricConstantAnisotropyModel,
)

model = AxisymmetricDSphModel(
    submodels={
        "StellarModel": AxisymmetricPlummerModel(re_pc=300., q=.7),
        "DMModel": AxisymmetricZhaoModel(
            rs_pc=500., rhos_Msunpc3=.1, Q=.8,
            alpha=2., beta=3., gamma=.5, r_t_pc=3000.),
        "AnisotropyModel": AxisymmetricConstantAnisotropyModel(beta_z=-.3),
    }, inclination=1.1, n_force=48, n_vertical=48, n_los=48,
)
x_pc = np.array([0., 100., -300.])
y_pc = np.array([0., 50., 70.])
variance = model.sigmalos2(x_pc, y_pc)
assert variance.shape == x_pc.shape
assert np.all(np.isfinite(variance) & (variance > 0))
print(np.sqrt(variance))
# example-end

# factors-start
# Use refined angular/radial settings for a scientific posterior summary.
settings = dict(n_mu=24, n_phi=24, n_radial=64)
J = model.jfactor(80000., .5, **settings)
D = model.dfactor(80000., .5, **settings)
assert np.isfinite(J) and J > 0
assert np.isfinite(D) and D > 0
print(J, D)  # GeV^2 cm^-5, GeV cm^-2
# factors-end
