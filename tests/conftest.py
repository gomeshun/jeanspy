import pytest


@pytest.fixture
def classical_prior_config():
    """Finite illustrative priors for the synthetic classical test models."""
    import pandas as pd

    return pd.DataFrame(
        {"lower": [-30., 2., 2.5, -3., 3.5, -.3],
         "upper": [30., 2.6, 3.5, -1., 4.5, .3]},
        index=["vmem_kms", "log10_re_pc", "log10_rs_pc",
               "log10_rhos_Msunpc3", "log10_r_t_pc", "bfunc_beta_ani"],
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-mcmc",
        action="store_true",
        default=False,
        help="run slow NumPyro MCMC execution tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "mcmc: slow NumPyro MCMC execution tests; run explicitly with --run-mcmc",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-mcmc"):
        return

    skip_mcmc = pytest.mark.skip(reason="requires --run-mcmc")
    for item in items:
        if "mcmc" in item.keywords:
            item.add_marker(skip_mcmc)
