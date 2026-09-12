"""Finite-cone factors for spherical and flattened halos."""
import numpy as np
from jeanspy.model import NFWModel
from jeanspy.axisymmetric import ZhaoHalo
from jeanspy.axisymmetric_factors import jfactor, dfactor

# spherical-factors-start
halo = NFWModel(rs_pc=500., rhos_Msunpc3=.1, r_t_pc=5000.)
distance_pc, aperture_deg = 76000., .5
J_GeV2_cm_minus5 = halo.jfactor_ullio2016(distance_pc, aperture_deg)
# The spherical limit of the spheroidal factor integrator also supplies D.
spherical = ZhaoHalo(rho_s=.1, r_s=500., Q=1., alpha=1., beta=3.,
                      gamma=1., r_t_pc=5000.)
D_GeV_cm_minus2 = dfactor(spherical, distance_pc, aperture_deg,
                          n_mu=48, n_phi=32, n_radial=48)
assert J_GeV2_cm_minus5 > 0 and D_GeV_cm_minus2 > 0
print("log10 J, log10 D:", np.log10(J_GeV2_cm_minus5), np.log10(D_GeV_cm_minus2))
# spherical-factors-end

# flattened-factors-start
flattened = ZhaoHalo(rho_s=.1, r_s=500., Q=.8, alpha=2., beta=4.,
                     gamma=.5, r_t_pc=5000.)
J_flat = jfactor(flattened, distance_pc, aperture_deg, inclination=1.1,
                 n_mu=48, n_phi=32, n_radial=48)
D_flat = dfactor(flattened, distance_pc, aperture_deg, inclination=1.1,
                 n_mu=48, n_phi=32, n_radial=48)
assert np.isfinite([J_flat, D_flat]).all() and min(J_flat, D_flat) > 0
# Refine these independent factor quadratures for the halo/aperture in use.
# flattened-factors-end
