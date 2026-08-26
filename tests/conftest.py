import pytest


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
