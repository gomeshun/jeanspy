"""JeansPy package initialization."""

import importlib

__version__ = "0.1.0"

__all__ = [
    "coord",
    "dequad",
    "model",
    "jfactor",
    "polygon",
    "sampler",
    "cmd_utilities",
]

def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
