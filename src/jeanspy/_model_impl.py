"""Compatibility shim for the pre-refactor classical implementation module.

New code should import from :mod:`jeanspy.model`.  The implementation now
lives in the focused :mod:`jeanspy._classical` modules; this shim is retained
temporarily so old internal imports do not break during the v0.1.0 transition.
"""

from ._classical import *  # noqa: F401,F403
from ._classical import _ullio2016_inner_weight, _ullio2016_weight
from .sersic import SersicModel
