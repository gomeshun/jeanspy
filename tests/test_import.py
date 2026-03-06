import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import jeanspy

def test_version_available():
    assert hasattr(jeanspy, "__version__")
