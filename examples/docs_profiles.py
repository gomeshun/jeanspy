"""NumPy/SciPy profiles, anisotropy and explicit units; no plotting dependency."""
import numpy as np
from jeanspy.model import (BaesAnisotropyModel, ConstantAnisotropyModel,
    ProjectedExponentialModel, NFWModel, OsipkovMerrittModel, PlummerModel,
    SersicModel, Uniform2dModel, ZhaoModel)

# profiles-start
r_pc = np.array([30., 100., 300.])
tracers = [PlummerModel(re_pc=300.), ProjectedExponentialModel(r_exp_pc=300.), SersicModel(re_pc=300., n=1.),
           Uniform2dModel(Rmax_pc=600.)]
for tracer in tracers:
    surface_pc_minus2 = tracer.density_2d(r_pc)
    assert surface_pc_minus2.shape == r_pc.shape
    assert np.all(surface_pc_minus2 > 0)
    # The uniform projected disk has no implemented three-dimensional density.
    if not isinstance(tracer, Uniform2dModel):
        density_pc_minus3 = tracer.density_3d(r_pc)
        assert np.all(density_pc_minus3 > 0)

# The exponential scale and projected half-light radius have distinct names.
np.testing.assert_allclose(tracers[1].re_pc, 1.67834699001666 * 300.)
np.testing.assert_allclose(tracers[1].cdf_R(tracers[1].re_pc), .5)
plummer = tracers[0]
np.testing.assert_allclose(plummer.cdf_R(plummer.params.re_pc), .5)
assert plummer.density_2d_truncated(np.array([700.]), 600.)[0] == 0

halos = [NFWModel(rs_pc=500., rhos_Msunpc3=.1, r_t_pc=5000.),
         ZhaoModel(rs_pc=500., rhos_Msunpc3=.1, r_t_pc=5000.,
                   alpha=2., beta=4., gamma=.5)]
for halo in halos:
    rho_Msun_pc_minus3 = halo.mass_density_3d(r_pc)
    mass_Msun = halo.enclosed_mass(r_pc)
    assert np.all(rho_Msun_pc_minus3 > 0) and np.all(np.diff(mass_Msun) > 0)
# profiles-end

# anisotropy-start
anisotropies = [ConstantAnisotropyModel(beta_ani=-.2),
                OsipkovMerrittModel(r_a=1000.),
                BaesAnisotropyModel(beta_0=-.2, beta_inf=.5,
                                    r_a=1000., eta=2.)]
for anisotropy in anisotropies:
    beta = anisotropy.beta(r_pc)
    assert np.all(np.asarray(beta) < 1)
    # beta = 1 - sigma_theta^2/sigma_r^2, for one tangential component.
    print(type(anisotropy).__name__, beta)
# anisotropy-end
