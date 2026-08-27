"""Compatibility exports for the classical model backend.

The implementation is divided into private modules by responsibility.  This
module remains as a single import point for code that historically imported
classical model symbols from ``jeanspy._model_impl``.
"""

from ._model_anisotropy import (
    AnisotropyModel,
    BaesAnisotropyModel,
    ConstantAnisotropyModel,
    OsipkovMerrittModel,
)
from ._model_base import Model, Parameters
from ._model_estimation import (
    DotDict,
    KI17_Model,
    SimpleDSphEstimationModel,
    get_default_estimation_model,
)
from ._model_jfactor import (
    C0,
    C1,
    C_J,
    DMModel,
    GMsun_m3s2,
    NFWModel,
    R_trunc_pc,
    ZhaoModel,
    _ullio2016_inner_weight,
    _ullio2016_weight,
    im_eV,
    kg_eV,
    solar_mass_kg,
)
from ._model_priors import (
    FittableModel,
    FlatPriorModel,
    PhotometryPriorModel,
)
from ._model_profiles import (
    Exp2dModel,
    Exp3dModel,
    PlummerModel,
    SersicModel,
    StellarModel,
    Uniform2dModel,
)
from ._model_solver import DSphModel


__all__ = [
    "Parameters",
    "Model",
    "StellarModel",
    "PlummerModel",
    "SersicModel",
    "Exp2dModel",
    "Exp3dModel",
    "Uniform2dModel",
    "DMModel",
    "ZhaoModel",
    "NFWModel",
    "AnisotropyModel",
    "ConstantAnisotropyModel",
    "OsipkovMerrittModel",
    "BaesAnisotropyModel",
    "DSphModel",
    "FittableModel",
    "FlatPriorModel",
    "PhotometryPriorModel",
    "SimpleDSphEstimationModel",
    "DotDict",
    "KI17_Model",
    "get_default_estimation_model",
    "GMsun_m3s2",
    "R_trunc_pc",
    "kg_eV",
    "im_eV",
    "solar_mass_kg",
    "C0",
    "C1",
    "C_J",
    "_ullio2016_weight",
    "_ullio2016_inner_weight",
]
