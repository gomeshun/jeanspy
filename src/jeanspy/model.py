"""Public classical model API.

The implementation is organized in private modules, while this module keeps
the supported stateful NumPy/SciPy model imports stable.
"""

from ._model_impl import (
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
    KI17_Model,
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
    get_default_estimation_model,
)
from ._model_jfactor import _ullio2016_inner_weight, _ullio2016_weight
from .sersic import SersicModel


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
    "C_J",
]


# Keep class and factory provenance stable for repr, introspection, and pickle.
for _name in __all__:
    _value = globals()[_name]
    if callable(_value):
        try:
            _value.__module__ = __name__
        except (AttributeError, TypeError):
            pass
del _name, _value
