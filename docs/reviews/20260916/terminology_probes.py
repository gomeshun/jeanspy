"""Read-only API probes for the 2026-09-16 terminology audit; no MCMC."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from jeanspy import model as numpy_backend
from jeanspy import model_numpyro as jax_backend
from jeanspy.axisymmetric import AxisymmetricZhaoModel
from jeanspy.baes_eta2 import BaesEta2AnisotropyModel
from jeanspy.sersic import SersicModel

root = Path.cwd()
BASELINE = "6eb9bdb7222c7baffdc8e259df571218e5fc4cbc"
result = {"baseline": BASELINE}
radii = np.geomspace(1., 1000., 12)
scale = 100.
exp2d = numpy_backend.Exp2dModel(re_pc=1.67834699001666 * scale)
exp3d = numpy_backend.Exp3dModel(re_pc=scale)
result["exponential_models"] = {
    "exp3d_parameter_re_pc": scale,
    "exp3d_half_light_radius_pc": exp3d.half_light_radius(),
    "same_projected_profile_after_scale_conversion": bool(np.allclose(exp2d.density_2d(radii), exp3d.density_2d(radii))),
    "same_intrinsic_profile_after_scale_conversion": bool(np.allclose(exp2d.density_3d(radii), exp3d.density_3d(radii))),
}
tracer = numpy_backend.PlummerModel(re_pc=200.)
tracer.update(target="DMModel", re_pc=300.)
result["ignored_update_target"] = {"requested_target": "DMModel", "tracer_re_pc_after_update": tracer.params.re_pc}

spherical = numpy_backend.NFWModel(rs_pc=100., rhos_Msunpc3=.1, r_t_pc=500.)
flattened = AxisymmetricZhaoModel(rs_pc=100., rhos_Msunpc3=.1, r_t_pc=500., Q=1.)
result["density_outside_same_cutoff"] = {
    "radius_pc": 1000., "cutoff_pc": 500.,
    "spherical_Msunpc3": float(spherical.mass_density_3d(1000.)),
    "axisymmetric_spherical_limit_Msunpc3": float(flattened.mass_density_3d(1000., 0.)),
}
try:
    spherical.assert_roi_is_enough_small(2.)
except ValueError as exc:
    result["roi_deg_max_warning"] = {"value": spherical.roi_deg_max_warning, "actual_behavior": type(exc).__name__, "message": str(exc)}

prior = pd.DataFrame({"lower": [-30., 2., 2.5, -3., 3.5, -.3],
                      "upper": [30., 2.6, 3.5, -1., 4.5, .3]},
                     index=["vmem_kms", "log10_re_pc", "log10_rs_pc", "log10_rhos_Msunpc3", "log10_r_t_pc", "bfunc_beta_ani"])
data = pd.DataFrame({"R_pc": [100.123456789, 200.123456789],
                     "vlos_kms": [1.123456789, -1.123456789], "e_vlos_kms": [2., 2.]})
fit = numpy_backend.get_default_estimation_model(data, 2.3, .1, config=prior)
result["spherical_observation_precision"] = {
    "input_dtype": str(data.R_pc.dtype), "stored_dtype": str(fit.data.R_pc.dtype),
    "input_R_pc": float(data.R_pc.iloc[0]), "stored_R_pc": float(fit.data.R_pc[0]),
}
result["docstring_contract_checks"] = {
    "sersic_has_documented_logdensity_2d": hasattr(SersicModel, "logdensity_2d"),
    "exp3d_has_documented_logdensity_2d": hasattr(numpy_backend.Exp3dModel, "logdensity_2d"),
    "osipkov_kernel_has_documented_backend_option": "backend" in inspect.signature(jax_backend.OsipkovMerrittModel.kernel).parameters,
    "baes_kernel_has_documented_backend_option": "backend" in inspect.signature(jax_backend.BaesAnisotropyModel.kernel).parameters,
}

specialized = jax_backend.DSphModel(submodels={"StellarModel": jax_backend.PlummerModel(),
    "DMModel": jax_backend.NFWModel(), "AnisotropyModel": BaesEta2AnisotropyModel()})
# Observe dispatch only; replacing both numerical methods avoids testing a scientific prediction.
specialized.sigmalos2_abel = lambda *args, **kwargs: "abel"
specialized.sigmalos2_kernel = lambda *args, **kwargs: "kernel"
result["specialized_baes_eta2_dispatch"] = {
    option: specialized.sigmalos2(100., params={}, backend=option, jit=False)
    for option in ("auto", "kernel")
}

package = root / "src/jeanspy"
hashes = {}
paths = sorted([*package.rglob("*.py"), *package.glob("data/*.csv")])
for version in ("baseline", "working_tree"):
    digest = hashlib.sha256()
    for path in paths:
        content = (subprocess.check_output(["git", "show", BASELINE + ":" + path.relative_to(root).as_posix()])
                   if version == "baseline" else path.read_bytes())
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(content)
    hashes[version] = digest.hexdigest()
result["restart_software_source_digest"] = hashes
print(json.dumps(result, indent=2, ensure_ascii=False))
