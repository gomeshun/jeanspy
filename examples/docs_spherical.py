"""Small spherical prediction used verbatim in the English documentation."""
# example-start
import numpy as np
from jeanspy.model import ConstantAnisotropyModel, DSphModel, NFWModel, PlummerModel

model = DSphModel(
    vmem_kms=0.,
    submodels={
        "StellarModel": PlummerModel(re_pc=200.),
        "DMModel": NFWModel(rs_pc=1000., rhos_Msunpc3=.01, r_t_pc=10000.),
        "AnisotropyModel": ConstantAnisotropyModel(beta_ani=0.),
    },
)
R_pc = np.array([50., 100., 300.])
variance = model.sigmalos2(R_pc)
sigma_kms = np.sqrt(variance)
assert variance.shape == R_pc.shape
assert np.all(np.isfinite(sigma_kms) & (sigma_kms > 0))
print(sigma_kms)
# example-end

halo = model["DMModel"]
mass = halo.enclosed_mass(R_pc)
density = halo.mass_density_3d(R_pc)
assert np.all(np.diff(mass) > 0) and np.all(density > 0)
