"""Shared observation updates must reach existing readers without stale buffers."""
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd
import pytest

from jeanspy.model import get_default_estimation_model


@pytest.fixture
def shared_model(classical_prior_config):
    data = pd.DataFrame(dict(R_pc=[10., 20., 30.], vlos_kms=[-1., 0., 1.],
                             e_vlos_kms=[1., 2., 3.]))
    model = get_default_estimation_model(data, 2.3, .1, config=classical_prior_config,
                                         vmem_prior_from_data=True)
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


@pytest.mark.parametrize("size", [8, 20])
def test_existing_handle_size_mismatch_preserves_other_buffers(shared_model, size):
    model = shared_model
    original = {field: values.copy() for field, values in model.data.items()}
    old_bounds = model["FlatPriorModel"].data.copy()
    original_handle = model.shm_e_vlos_kms
    wrong = SharedMemory(create=True, size=size)
    model.shm_e_vlos_kms = wrong
    try:
        new = pd.DataFrame({field: values + 100. for field, values in original.items()})
        with pytest.raises(ValueError, match="expected 12 bytes for e_vlos_kms"):
            model.reset_data(new)
        assert model.n_data == 3
        assert model.shared_shape == (3,)
        assert model.buffer_size == 12
        # In particular the earlier fields must not have been copied yet.
        model.shm_e_vlos_kms = original_handle
        for field, values in original.items():
            np.testing.assert_array_equal(model.data[field], values)
        pd.testing.assert_frame_equal(model["FlatPriorModel"].data, old_bounds)
    finally:
        model.shm_e_vlos_kms = original_handle
        wrong.close()
        wrong.unlink()


@pytest.mark.parametrize("size", [8, 20])
def test_stale_attachment_failure_cleans_new_segments_and_restores_state(classical_prior_config, size):
    data = pd.DataFrame(dict(R_pc=[10., 20., 30.], vlos_kms=[-1., 0., 1.],
                             e_vlos_kms=[1., 2., 3.]))
    model = get_default_estimation_model(data, 2.3, .1, config=classical_prior_config)
    basename = f"SimpleDSphEstimationModel_{id(model)}"
    stale = SharedMemory(name=basename + "_e_vlos_kms", create=True, size=size)
    old_bounds = model["FlatPriorModel"].data.copy()
    try:
        with pytest.raises(ValueError, match="expected 12 bytes for e_vlos_kms"):
            model.load_data(data + 100., shared=True)
        assert model.shared is False
        assert model.n_data == 3
        assert not hasattr(model, "shared_shape")
        assert not hasattr(model, "buffer_size")
        for field in data:
            np.testing.assert_array_equal(model.data[field], data[field].values)
            assert not hasattr(model, f"shm_{field}")
        pd.testing.assert_frame_equal(model["FlatPriorModel"].data, old_bounds)
        # New segments from this failed attempt are gone; the stale segment
        # belongs to somebody else and must remain available.
        for field in ("R_pc", "vlos_kms"):
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=basename + "_" + field)
        attached = SharedMemory(name=stale.name)
        attached.close()
    finally:
        stale.close()
        stale.unlink()
    # A clean retry must succeed after the stale segment is removed.
    model.load_data(data + 100., shared=True)
    try:
        np.testing.assert_array_equal(model.data.vlos_kms, [99., 100., 101.])
    finally:
        model.release_shared_memory()
