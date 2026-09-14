"""Immutable component mapping shared by the axisymmetric backends."""
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Components(Mapping):
    stellar: object
    halo: object
    anisotropy: object

    def __iter__(self):
        return iter(("StellarModel", "DMModel", "AnisotropyModel"))

    def __len__(self):
        return 3

    def __getitem__(self, key):
        return {"StellarModel": self.stellar, "DMModel": self.halo,
                "AnisotropyModel": self.anisotropy}[key]


def component_models(submodels, expected):
    """Require the backend's concrete supported families, not just their ABCs."""
    if not isinstance(submodels, Mapping) or set(submodels) != {
        "StellarModel", "DMModel", "AnisotropyModel",
    }:
        raise ValueError("submodels must contain StellarModel, DMModel and AnisotropyModel")
    values = tuple(submodels[key] for key in ("StellarModel", "DMModel", "AnisotropyModel"))
    for value, cls in zip(values, expected):
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__module__}.{cls.__name__}, got {type(value).__name__}")
    return Components(*values)


def component_params(params, part, xp):
    """Validate one functional component without requiring unrelated physics."""
    from ._axisymmetric_params import SUPPORTED, resolve_params
    unknown = set(params) - SUPPORTED
    if unknown:
        raise ValueError(f"Unknown axisymmetric parameters: {sorted(unknown)}")
    keys, required = {
        "stellar": ({"re_pc", "q", "q_projected", "inclination"}, {"re_pc"}),
        "halo": ({"rs_pc", "rhos_Msunpc3", "Q", "alpha", "beta", "gamma", "r_t_pc"},
                 {"rs_pc", "rhos_Msunpc3"}),
        "anisotropy": ({"beta_z"}, set()),
    }[part]
    missing = required - set(params)
    if missing:
        raise ValueError(f"Missing axisymmetric parameters: {sorted(missing)}")
    base = dict(re_pc=1., rs_pc=1., rhos_Msunpc3=1., q=1.)
    if part == "stellar":
        if ("q" in params) == ("q_projected" in params):
            raise ValueError("Supply exactly one of q and q_projected")
        if "q_projected" in params:
            base.pop("q")
    base.update({k: params[k] for k in keys if k in params})
    return resolve_params(base, xp)
