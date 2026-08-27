import numpy as np

from jeanspy.cmd_utilities import inpoly


def test_inpoly_has_no_stdout_side_effect(capsys):
    xs_vertex = np.array([0.0, 1.0, 1.0, 0.0])
    ys_vertex = np.array([0.0, 0.0, 1.0, 1.0])
    x = np.array([0.5, 1.5])
    y = np.array([0.5, 0.5])

    result = inpoly(x, y, xs_vertex, ys_vertex)

    np.testing.assert_array_equal(result, np.array([True, False]))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
