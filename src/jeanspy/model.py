"""Public classical NumPy/SciPy model API.

Implementation details live in :mod:`jeanspy._classical`; this module defines
the intentionally supported import surface for the stateful classical backend.
"""

from ._classical import (
    AnisotropyModel,
    BaesAnisotropyModel,
    C_J,
    ConstantAnisotropyModel,
    DMModel,
    DSphModel,
    DotDict,
    Exp2dModel,
    Exp3dModel,
    FittableModel,
    FlatPriorModel,
    GMsun_m3s2,
    Model,
    NFWModel,
    OsipkovMerrittModel,
    Parameters,
    PhotometryPriorModel,
    PlummerModel,
    SimpleDSphEstimationModel,
    StellarModel,
    Uniform2dModel,
    ZhaoModel,
    _ullio2016_inner_weight,
    _ullio2016_weight,
    get_default_estimation_model,
)
from .sersic import SersicModel

#: Convert a numerical J factor from Msun^2/pc^5 to GeV^2/cm^5.
#: Apply this factor only to an integral that has not already been converted.
C_J: float

#: Solar gravitational parameter G*Msun, in m^3/s^2.
GMsun_m3s2: float


__all__ = [
    "AnisotropyModel",
    "BaesAnisotropyModel",
    "C_J",
    "ConstantAnisotropyModel",
    "DMModel",
    "DSphModel",
    "DotDict",
    "Exp2dModel",
    "Exp3dModel",
    "FittableModel",
    "FlatPriorModel",
    "GMsun_m3s2",
    "Model",
    "NFWModel",
    "OsipkovMerrittModel",
    "Parameters",
    "PhotometryPriorModel",
    "PlummerModel",
    "SersicModel",
    "SimpleDSphEstimationModel",
    "StellarModel",
    "Uniform2dModel",
    "ZhaoModel",
    "get_default_estimation_model",
]


# Preserve the historical public provenance used by repr/pickle/introspection
# while keeping implementation modules private.
for _name in __all__:
    _value = globals()[_name]
    if hasattr(_value, "__module__"):
        _value.__module__ = __name__

del _name, _value

# Additive axisymmetric API. Preserve the defining modules of the independent
# forward/inference classes, including the public paths introduced in PR #63.
from .axisymmetric import AxisymmetricDSphModel
from .axisymmetric_inference import AxisymmetricDSphEstimationModel, AxisymmetricKinematicData

__all__ += ["AxisymmetricDSphModel", "AxisymmetricDSphEstimationModel", "AxisymmetricKinematicData"]
