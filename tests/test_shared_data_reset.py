"""Shared observation updates must reach existing readers without stale buffers."""
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd
import pytest

from jeanspy.model import get_default_estimation_model


@pytest.fixture
def shared_model(tmp_path):
    data = pd.DataFrame(dict(R_pc=[10., 20., 30.], vlos_kms=[-1., 0., 1.],
                             e_vlos_kms=[1., 2., 3.]))
    model = get_default_estimation_model(data, 2.3, .1, config=str(tmp_path / "prior.csv"))
    model.load_data(data, shared=True)
    try:
        yield model
    finally:
        model.release_shared_memory()


def test_reset_updates_existing_shared_readers(shared_model):
    model = shared_model
    fields = ("R_pc", "vlos_kms", "e_vlos_kms")
    handles = {field: getattr(model, f"shm_{field}") for field in fields}
    readers = {field: SharedMemory(name=handle.name) for field, handle in handles.items()}
    replacement = pd.DataFrame(dict(R_pc=[40., 50., 60.], vlos_kms=[99., 100., 101.],
                                    e_vlos_kms=[4., 5., 6.]))
    try:
        # Repeat to catch reopening/leaking handles and stale reattachment paths.
        for offset in (0., 10.):
            new = replacement + offset  # float64 input; shared layout remains float32
            model.reset_data(new)
            assert model.n_data == 3
            for field in fields:
                assert getattr(model, f"shm_{field}") is handles[field]
                observed = np.ndarray((3,), dtype=model.dtype, buffer=readers[field].buf)
                np.testing.assert_array_equal(observed, new[field].to_numpy(dtype=model.dtype))
                np.testing.assert_array_equal(model.data[field], observed)
            bounds = model["FlatPriorModel"].data.loc["vmem_kms"]
            assert bounds["lower"] == new.vlos_kms.min()
            assert bounds["upper"] == new.vlos_kms.max()
    finally:
        for reader in readers.values():
            reader.close()


@pytest.mark.parametrize("n", [2, 4])
def test_shared_resize_is_rejected_without_mutating_model(shared_model, n):
    model = shared_model
    original = {field: values.copy() for field, values in model.data.items()}
    old_bounds = model["FlatPriorModel"].data.copy()
    new = pd.DataFrame({field: np.arange(n, dtype=float) for field in original})
    with pytest.raises(ValueError, match="Cannot resize shared kinematic data"):
        model.reset_data(new)
    assert model.n_data == 3
    assert model.shared_shape == (3,)
    for field, values in original.items():
        np.testing.assert_array_equal(model.data[field], values)
    pd.testing.assert_frame_equal(model["FlatPriorModel"].data, old_bounds)
