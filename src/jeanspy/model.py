"""Public model API.

The historical implementation remains in :mod:`jeanspy._model_impl` while the
Sérsic implementation is maintained separately in :mod:`jeanspy.sersic`.
This facade preserves the existing ``jeanspy.model`` import surface.
"""

from . import _model_impl as _impl

# Preserve the broad historical module namespace, including a few private
# helpers that downstream notebooks may still import.  Dunder attributes are
# intentionally kept from this public facade rather than copied from the
# implementation module.
for _name, _value in vars(_impl).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

# Replace the historical Sérsic implementation with the maintained public one.
from .sersic import SersicModel as SersicModel

# Keep public class/function provenance stable for repr/pickle/introspection.
for _name, _value in list(globals().items()):
    if getattr(_value, "__module__", None) == _impl.__name__:
        try:
            _value.__module__ = __name__
        except (AttributeError, TypeError):
            pass
SersicModel.__module__ = __name__

# Avoid leaking facade implementation details through ordinary inspection.
del _name, _value, _impl
