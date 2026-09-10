"""Exercise binary libraries together, independently of pytest import order."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize('order', [('h5py', 'netCDF4'), ('netCDF4', 'h5py')])
def test_hdf5_netcdf_import_orders_write_and_read(tmp_path, order):
    program = '''
import importlib
import sys
from pathlib import Path
for module in sys.argv[1:3]:
    importlib.import_module(module)
import h5py
import netCDF4
import numpy as np
root = Path(sys.argv[3])
with netCDF4.Dataset(root / 'test.nc', 'w') as ds:
    ds.createDimension('x', 2)
    ds.createVariable('x', 'f8', ('x',))[:] = [1., 2.]
with h5py.File(root / 'test.h5', 'w') as ds:
    ds['x'] = [3., 4.]
with netCDF4.Dataset(root / 'test.nc') as ds:
    np.testing.assert_array_equal(ds['x'][:], [1., 2.])
with h5py.File(root / 'test.h5') as ds:
    np.testing.assert_array_equal(ds['x'][:], [3., 4.])
'''
    result = subprocess.run([sys.executable, '-c', program, *order, str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
