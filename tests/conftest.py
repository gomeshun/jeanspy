import os
from pathlib import Path

import pytest


_LEGACY_SWYFT_TEST_FILES = {"test_model.py", "swyfttest.py"}


def pytest_ignore_collect(collection_path, config):
    if os.environ.get("JEANSPY_ENABLE_LEGACY_SWYFT_TESTS") == "1":
        return False
    return Path(collection_path).name in _LEGACY_SWYFT_TEST_FILES


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