"""Tests verifying that core model classes have no runtime dependency on
dsph_database and accept explicitly supplied data / prior parameters."""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def test_import_model_no_dsph_database_warning():
    """Importing jeanspy.model must not emit any dsph_database warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import jeanspy.model  # noqa: F401
    dsph_warnings = [w for w in caught if "dsph_database" in str(w.message).lower()]
    assert dsph_warnings == [], f"Unexpected dsph_database warnings: {dsph_warnings}"


def test_photometry_prior_model_explicit_params():
    """PhotometryPriorModel should accept explicit loc and scale."""
    from jeanspy.model import PhotometryPriorModel

    loc, scale = 2.3, 0.1
    m = PhotometryPriorModel(loc=loc, scale=scale)
    sampled = m.sample(size=10)
    assert sampled.shape == (10,)
    lp = m._lnprior(loc)
    assert np.isfinite(lp)


def test_photometry_prior_model_nan_params():
    """PhotometryPriorModel with NaN loc/scale should be constructable;
    reset_prior can then supply real values."""
    from jeanspy.model import PhotometryPriorModel

    m = PhotometryPriorModel(loc=float("nan"), scale=float("nan"))
    m.reset_prior(loc=2.5, scale=0.2)
    assert np.isfinite(m._lnprior(2.5))


def _make_kinematic_data(n=30, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "R_pc": rng.uniform(0, 500, n),
        "vlos_kms": rng.normal(0, 10, n),
        "e_vlos_kms": rng.uniform(1, 3, n),
    })


def test_simple_dsph_estimation_model_explicit_data(tmp_path):
    """SimpleDSphEstimationModel.load_data accepts a DataFrame directly."""
    from jeanspy.model import (
        SimpleDSphEstimationModel,
        DSphModel,
        PlummerModel,
        NFWModel,
        ConstantAnisotropyModel,
        FlatPriorModel,
        PhotometryPriorModel,
    )

    data = _make_kinematic_data()

    config_path = tmp_path / "priorconfig.csv"
    dsph_model = DSphModel(submodels={
        "StellarModel": PlummerModel(),
        "DMModel": NFWModel(),
        "AnisotropyModel": ConstantAnisotropyModel(),
    })
    FlatPriorModel.generate_default_config_file(str(config_path), dsph_model.params_all.index)

    mdl = SimpleDSphEstimationModel(
        args_load_data=[data],
        submodels={
            "DSphModel": DSphModel(submodels={
                "StellarModel": PlummerModel(),
                "DMModel": NFWModel(),
                "AnisotropyModel": ConstantAnisotropyModel(),
            }),
            "FlatPriorModel": FlatPriorModel(config=str(config_path)),
            "PhotometryPriorModel": PhotometryPriorModel(loc=2.3, scale=0.1),
        },
    )
    # Data should be stored and accessible
    assert mdl.n_data == len(data)
    assert "vlos_kms" in mdl.data


def test_get_default_estimation_model(tmp_path):
    """get_default_estimation_model uses explicit data and photometry prior."""
    from jeanspy.model import get_default_estimation_model

    data = _make_kinematic_data()
    config_path = tmp_path / "priorconfig.csv"

    mdl = get_default_estimation_model(
        data=data,
        photometry_prior_loc=2.3,
        photometry_prior_scale=0.1,
        config=str(config_path),
    )
    assert mdl.n_data == len(data)
