"""Current JAM spectral reference in an exactly matched isotropic domain.

The external results are frozen and source-hashed. Users can regenerate them
with scripts/validate_axisymmetric_jam9.py in the separately pinned JAM 9 env.
Normal tests need no JAM installation or redistribution of its source.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from jeanspy.axisymmetric import AxisymmetricDSphModel

REFERENCE = Path(__file__).resolve().parents[1]/"validation/release/jam9_plummer_v1.json"


@pytest.mark.parametrize("case_index", [0, 1])
def test_current_jam_isotropic_projected_reference(case_index):
    report = json.loads(REFERENCE.read_text())
    case = report["cases"][case_index]
    xy = report["protocol"]["sky_coordinates_pc"]
    actual = AxisymmetricDSphModel(64, 64, 64).sigmalos2(xy["x"], xy["y"], params=case["params"])
    expected = case["jam_projected"][-1]["projected"]
    np.testing.assert_allclose(actual, expected,
        rtol=report["protocol"]["acceptance"]["finest_cross_code_relative"])


def test_jax_physical_density_gradient_on_jam_matched_case():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jeanspy.axisymmetric_numpyro import AxisymmetricDSphModel as JaxModel
    report = json.loads(REFERENCE.read_text())
    case = report["cases"][1]
    xy = report["protocol"]["sky_coordinates_pc"]
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        model = JaxModel(64, 64, 64)
        x, y = jnp.asarray(xy["x"]), jnp.asarray(xy["y"])
        def forward(rho):
            return model.sigmalos2(x, y, params={**case["params"], "rhos_Msunpc3": rho})
        rho = case["params"]["rhos_Msunpc3"]
        value = jax.block_until_ready(jax.jit(forward)(rho))
        derivative = jax.block_until_ready(jax.jit(jax.jacfwd(forward))(rho))
        np.testing.assert_allclose(value, case["jam_projected"][-1]["projected"], rtol=.005)
        # Exact physical scaling, independent of the implementation of AD.
        np.testing.assert_allclose(derivative, value/rho, rtol=1e-9)
    finally:
        jax.config.update("jax_enable_x64", previous)
