"""Regression tests for Parameters copy/deepcopy behaviour (issue #8)."""
import copy
import pickle
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from jeanspy.model import Parameters


def test_deepcopy_succeeds():
    """copy.deepcopy(Parameters(...)) must not raise."""
    p = Parameters({"a": 1, "nested": [1, 2, 3]})
    p_deep = copy.deepcopy(p)
    assert p_deep["a"] == 1
    assert p_deep["nested"] == [1, 2, 3]


def test_deepcopy_independence():
    """Mutating nested data in the deep copy must not affect the original."""
    p = Parameters({"values": [1, 2, 3]})
    p_deep = copy.deepcopy(p)
    p_deep["values"].append(99)
    assert p["values"] == [1, 2, 3], "Original was mutated by deep copy"


def test_shallow_copy_unchanged():
    """Shallow copy (Parameters.copy()) must still share nested objects."""
    p = Parameters({"values": [1, 2, 3]})
    p_shallow = p.copy()
    p_shallow["values"].append(99)
    assert p["values"] == [1, 2, 3, 99], "Shallow copy did not share nested list"


def test_pickle_roundtrip():
    """Parameters must survive a pickle round-trip."""
    p = Parameters({"x": 42, "y": [1, 2]})
    p2 = pickle.loads(pickle.dumps(p))
    assert p2["x"] == 42
    assert p2["y"] == [1, 2]
