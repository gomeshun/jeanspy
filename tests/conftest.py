import os
from pathlib import Path


_LEGACY_SWYFT_TEST_FILES = {"test_model.py", "swyfttest.py"}


def pytest_ignore_collect(collection_path, config):
    if os.environ.get("JEANSPY_ENABLE_LEGACY_SWYFT_TESTS") == "1":
        return False
    return Path(collection_path).name in _LEGACY_SWYFT_TEST_FILES