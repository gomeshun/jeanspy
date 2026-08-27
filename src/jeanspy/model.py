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


# Preserve the historical class provenance used by repr/pickle and downstream
# notebooks while keeping implementation modules private.
for _name in __all__:
    _value = globals()[_name]
    if isinstance(_value, type):
        _value.__module__ = __name__

del _name, _value
