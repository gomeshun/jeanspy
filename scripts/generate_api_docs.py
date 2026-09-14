"""Generate the documentation inventory from real supported module exports.

This script changes documentation only. It never adds/removes runtime exports.
Run before sphinx-build in the same environment used to build the API pages.
"""
from __future__ import annotations

import importlib
import inspect
import json
from functools import cached_property
from pathlib import Path
import re
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SOURCE = ROOT / "docs" / "source"
MODULES = {
    "model": "NumPy/SciPy",
    "axisymmetric": "Axisymmetric components and NumPy forward model",
    "axisymmetric_inference": "Axisymmetric observations and classical inference",
    "model_numpyro": "JAX",
    "axisymmetric_numpyro": "JAX axisymmetric forward model",
    "axisymmetric_factors": "Finite-cone NumPy J and D factors",
    "sampler": "Classical emcee sampling and HDF5 storage",
    "sampler_numpyro": "NumPyro likelihoods, sampling and storage",
    "sersic": "Spherical Sersic tracer",
    "baes_eta2": "Specialized Baes anisotropy kernels",
    "hyp2f1_jax": "JAX hypergeometric numerical helpers",
    "dequad": "Double-exponential integration helpers",
}


def exports(module):
    if hasattr(module, "__all__"):
        return list(module.__all__)
    # Modules without __all__: include locally defined public callables only,
    # never incidental imports such as numpy.exp or pathlib.Path.
    return [name for name, obj in vars(module).items()
            if not name.startswith("_") and callable(obj)
            and getattr(obj, "__module__", None) == module.__name__]


def source_documentation(module, name, obj):
    """Read docstrings or Sphinx's source comments; never supply API prose here."""
    if inspect.isclass(obj) or inspect.isroutine(obj):
        # Do not mistake an inherited class docstring for documentation of a
        # new public class with different parameters or scientific assumptions.
        doc = obj.__doc__ if inspect.isclass(obj) else inspect.getdoc(obj)
        if not doc or not doc.strip():
            raise ValueError(f"Missing source docstring: {module.__name__}.{name}")
        return "docstring"
    from sphinx.pycode import ModuleAnalyzer
    comments = ModuleAnalyzer.for_module(module.__name__).find_attr_docs()
    if not comments.get(("", name)):
        raise ValueError(f"Missing source doc-comment: {module.__name__}.{name}")
    return "doc-comment"


def documentable_members(cls):
    """Require source descriptions for the package's public method definitions."""
    members = []
    for name, member in inspect.getmembers(cls):
        if name.startswith("_") or not (
            callable(member) or isinstance(member, (property, cached_property))
        ):
            continue
        members.append(name)
        original = member
        if isinstance(member, property):
            original = member.fget
        elif isinstance(member, cached_property):
            original = member.func
        if inspect.ismethod(original):
            original = original.__func__
        original = inspect.unwrap(original)
        code = getattr(original, "__code__", None)
        if code and Path(code.co_filename).is_relative_to(ROOT / "src"):
            if not original.__doc__ or not original.__doc__.strip():
                raise ValueError(f"Missing source method docstring: {cls.__name__}.{name}")
    return members



def category_for(path):
    """Classify exports by user task, including mixed convenience modules."""
    module, name = path.rsplit(".", 1)
    if "sampler" in module or "inference" in module or name in {
        "FittableModel", "FlatPriorModel", "PhotometryPriorModel",
        "SimpleDSphEstimationModel", "get_default_estimation_model",
        "AxisymmetricDSphEstimationModel", "AxisymmetricKinematicData",
    }:
        return "inference"
    if module in {"jeanspy.dequad", "jeanspy.hyp2f1_jax"} or name in {
        "C_J", "C_D", "GMsun_m3s2", "Model", "Parameters", "DotDict",
        "configure_runtime", "get_runtime_config",
    }:
        return "core"
    if "axisymmetric" in module or name.startswith("Axisymmetric"):
        return "axisymmetric-models"
    return "spherical"

def main():
    destination = SOURCE / "api"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    (destination / "generated").mkdir(exist_ok=True)
    groups = {"core": ("Core utilities", []),
              "spherical": ("Spherical models", []),
              "axisymmetric-models": ("Axisymmetric models", []),
              "inference": ("Statistical inference", [])}
    index = ["# API reference", "",
             "Choose a category, then the NumPy/SciPy or JAX implementation.", "",
             "| Category | Contents |", "| --- | --- |",
             "| [Core utilities](core) | Units, parameter containers, runtime settings and numerical helpers |",
             "| [Spherical models](spherical) | Stellar profiles, dark-matter halos, anisotropy and Jeans solvers |",
             "| [Axisymmetric models](axisymmetric-models) | Flattened components, projection, Jeans solvers and J/D factors |",
             "| [Statistical inference](inference) | Observations, priors, likelihoods, MCMC and storage |", "",
             "See also the [units and shape contract](../guides/contracts.md) and",
             "[Quickstart](../quickstart.md). [Browse by import module](modules).", "",
             "```{toctree}", ":hidden:", ":maxdepth: 2", "",
             *groups, "modules", "```", ""]
    inventory = []
    seen = {}
    for short, title in MODULES.items():
        module_name = f"jeanspy.{short}"
        module = importlib.import_module(module_name)
        page = [title, "=" * len(title), "", f".. py:module:: {module_name}", "",
                f".. currentmodule:: {module_name}", "",
                "See :doc:`../guides/contracts` for units, shapes, domains and backend",
                "boundaries, and :doc:`../quickstart` for executable minimal examples.", ""]
        unique, aliases = [], []
        for name in exports(module):
            obj = getattr(module, name)
            path = f"{module_name}.{name}"
            # Constants may have equal values but are distinct documented names.
            identity = id(obj) if callable(obj) else path
            canonical = seen.setdefault(identity, path)
            try:
                signature = str(inspect.signature(obj))
            except (ValueError, TypeError):
                signature = None
            methods = []
            if inspect.isclass(obj):
                methods = documentable_members(obj)
            documentation = source_documentation(module, name, obj)
            documented_methods = [n for n in methods if inspect.getdoc(getattr(obj, n))]
            inventory.append(dict(path=path, canonical=canonical,
                                  signature=signature, methods=methods,
                                  documentation_source=documentation,
                                  documented_methods=documented_methods))
            if path == canonical:
                unique.append(path)
            else:
                aliases.append((path, canonical))
        if unique:
            page += [".. autosummary::", "", *[f"   {p.rsplit('.', 1)[1]}" for p in unique], ""]
        if aliases:
            page += ["Aliases", "-------", ""]
            for path, canonical in aliases:
                page += [f"* ``{path}`` is :py:obj:`{canonical}`."]
        (destination / f"{short}.rst").write_text("\n".join(page) + "\n")
        for path in unique:
            groups[category_for(path)][1].append(path)
    for group, (title, paths) in groups.items():
        body = [title, "=" * len(title), ""]
        for short in MODULES:
            members = [p for p in paths if p.rsplit(".", 1)[0] == "jeanspy." + short]
            if not members:
                continue
            label = MODULES[short]
            body += [label, "-" * len(label), "", ".. autosummary::",
                     "   :toctree: generated", "", *[f"   {p}" for p in members], ""]
        (destination / f"{group}.rst").write_text("\n".join(body))
    (destination / "modules.rst").write_text(
        "Import modules\n==============\n\n.. toctree::\n   :maxdepth: 1\n\n" +
        "".join(f"   {short}\n" for short in MODULES))
    (destination / "index.md").write_text("\n".join(index))
    (ROOT / "docs" / "api_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")

    # Reuse the authoritative existing guides, adapting only renderer syntax
    # and links whose original paths refer to the repository rather than site.
    for original, target in [("axisymmetric.md", "axisymmetric.md"),
                             ("ullio_jfactor_geometry.md", "ullio-geometry.md")]:
        body = (ROOT / "docs" / original).read_text()
        if original == "axisymmetric.md":
            # The repository guide also retains research measurements. Only
            # the user-facing model and numerical settings belong on the site.
            body = body.split("The recorded [CPU runtime sample]", 1)[0]
            body = body.replace("## Numerical equations and validation", "## Numerical equations")
        body = re.sub(r"```math\n(.*?)\n```", r"$$\n\1\n$$", body, flags=re.S)
        body = re.sub(r"\]\(\.\./([^)]*)\)",
                      r"](https://github.com/gomeshun/jeanspy/blob/" +
                      "1a0ad4028d26af1df389ebdfdf992285ec50f8bb/" + r"\1)", body)
        (SOURCE / "guides" / target).write_text(body)
    print(f"Documented {len(inventory)} exports in {len(MODULES)} modules")


if __name__ == "__main__":
    main()
