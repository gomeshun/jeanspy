import unittest

import numpy as np
import jax.numpy as jnp
from scipy.integrate import quad

from jeanspy.model import DMModel, NFWModel, ZhaoModel
from jeanspy.model_numpyro import (
    ConstantAnisotropyModel as ConstantAnisotropyModelJax,
    DSphModel as DSphModelJax,
    NFWModel as NFWModelJax,
    PlummerModel as PlummerModelJax,
    ZhaoModel as ZhaoModelJax,
)


class TestDMEnclosureMassConsistency(unittest.TestCase):
    def _numerical_enclosure_mass(self, model: DMModel, r: float) -> float:
        r_upper = min(float(r), float(model.params.r_t_pc))

        def integrand(s):
            return 4.0 * np.pi * (s**2) * model.mass_density_3d(s)

        value, _ = quad(integrand, 0.0, r_upper, epsabs=1e-10, epsrel=1e-8, limit=300)
        return value

    def _assert_model_consistency(self, model: DMModel, radii: np.ndarray, rtol: float) -> None:
        for r in radii:
            m_num = self._numerical_enclosure_mass(model, r)
            m_ana = float(np.asarray(model.enclosure_mass(np.array([r])))[0])
            self.assertTrue(np.isfinite(m_num))
            self.assertTrue(np.isfinite(m_ana))
            self.assertAlmostEqual(m_num, m_ana, delta=rtol * max(1.0, abs(m_num)))

    def test_dm_subclasses_enclosure_mass_matches_integrated_density(self):
        model_specs = [
            (
                NFWModel(rs_pc=350.0, rhos_Msunpc3=0.08, r_t_pc=4000.0),
                np.array([0.3, 3.0, 30.0, 300.0, 1500.0, 4000.0, 8000.0]),
                3e-5,
            ),
            (
                ZhaoModel(
                    rs_pc=420.0,
                    rhos_Msunpc3=0.06,
                    a=1.2,
                    b=4.0,
                    g=0.6,
                    r_t_pc=5000.0,
                ),
                np.array([0.4, 4.0, 40.0, 400.0, 2000.0, 5000.0, 9000.0]),
                2e-4,
            ),
        ]

        for model, radii, rtol in model_specs:
            with self.subTest(model=model.__class__.__name__):
                self.assertIsInstance(model, DMModel)
                self._assert_model_consistency(model, radii, rtol=rtol)


class TestDSSigmaLos2DMEquivalence(unittest.TestCase):
    def test_sigmalos2_nfw_equals_zhao_nfw_limit(self):
        params_nfw = {
            "re_pc": 200.0,
            "rs_pc": 1200.0,
            "rhos_Msunpc3": 1e-2,
            "r_t_pc": 8000.0,
            "beta_ani": 0.2,
            "vmem_kms": 0.0,
        }
        params_zhao = {**params_nfw, "a": 1.0, "b": 3.0, "g": 1.0}

        dsph_nfw = DSphModelJax(
            submodels={
                "StellarModel": PlummerModelJax(),
                "DMModel": NFWModelJax(),
                "AnisotropyModel": ConstantAnisotropyModelJax(),
            }
        )
        dsph_zhao = DSphModelJax(
            submodels={
                "StellarModel": PlummerModelJax(),
                "DMModel": ZhaoModelJax(),
                "AnisotropyModel": ConstantAnisotropyModelJax(),
            }
        )

        R_pc = jnp.asarray(np.geomspace(1.0, 1e3, 96), dtype=jnp.float32)
        s2_nfw = np.asarray(dsph_nfw.sigmalos2(R_pc, params=params_nfw, n_u=192, u_max=1500.0), dtype=np.float64)
        s2_zhao = np.asarray(dsph_zhao.sigmalos2(R_pc, params=params_zhao, n_u=192, u_max=1500.0), dtype=np.float64)

        self.assertTrue(np.isfinite(s2_nfw).all())
        self.assertTrue(np.isfinite(s2_zhao).all())
        np.testing.assert_allclose(s2_zhao, s2_nfw, rtol=8e-3, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
