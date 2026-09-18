"""Public NumPy/SciPy model API.

Compose stateful stellar, halo and anisotropy components and attach explicit
likelihoods and priors for NumPy/SciPy inference.
"""

from ._numpy import (
    AnisotropyModel,
    BaesAnisotropyModel,
    C_J,
    ConstantAnisotropyModel,
    DMModel,
    DSphModel,
    DotDict,
    ProjectedExponentialModel,
    FittableModel,
    FlatPriorModel,
    GMsun_m3s2,
    Model,
    NFWModel,
    OsipkovMerrittModel,
    Parameters,
    PhotometryPriorModel,
    PlummerModel,
    SphericalDSphEstimationModel,
    StellarModel,
    Uniform2dModel,
    ZhaoModel,
    plummer_nfw_constant_anisotropy_model,
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
    "ProjectedExponentialModel",
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
    "SphericalDSphEstimationModel",
    "StellarModel",
    "Uniform2dModel",
    "ZhaoModel",
    "plummer_nfw_constant_anisotropy_model",
]


# Preserve the supported public provenance used by repr/pickle/introspection
# while keeping implementation modules private.
for _name in __all__:
    _value = globals()[_name]
    if hasattr(_value, "__module__"):
        _value.__module__ = __name__

del _name, _value

# Re-export axisymmetric models while retaining their defining public modules.
from .axisymmetric import AxisymmetricDSphModel
from .axisymmetric_inference import AxisymmetricDSphEstimationModel, AxisymmetricKinematicData

__all__ += ["AxisymmetricDSphModel", "AxisymmetricDSphEstimationModel", "AxisymmetricKinematicData"]

from .axisymmetric import (
    AxisymmetricPlummerModel,
    AxisymmetricZhaoModel,
    AxisymmetricConstantAnisotropyModel,
    AxisymmetricStellarModel,
    AxisymmetricDMModel,
    AxisymmetricAnisotropyModel,
)

__all__ += ['AxisymmetricPlummerModel', 'AxisymmetricZhaoModel', 'AxisymmetricConstantAnisotropyModel', 'AxisymmetricStellarModel', 'AxisymmetricDMModel', 'AxisymmetricAnisotropyModel']
