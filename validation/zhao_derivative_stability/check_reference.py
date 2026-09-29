"""Reproduce the four fixed-quadrature Fisher references using GCC binary128.

Build reference.cpp as explained in README.md; the Python runtime needs NumPy.
The calculation is scalar and deliberately independent of JAX/autodiff.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    library = ctypes.CDLL(str(args.library.resolve()))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.evaluate.argtypes = [ctypes.c_int]*4 + [ctypes.c_double]*2 + [pointer]*11
    library.evaluate.restype = ctypes.c_int
    fixtures = json.loads((ROOT/'fisher_cases.json').read_text())
    nodes, weights = np.polynomial.legendre.leggauss(128)
    start = time.monotonic()
    rows = []
    for case, data in fixtures['cases'].items():
        for point in data['points']:
            arrays = [np.ascontiguousarray(v, dtype=np.float64) for v in [
                point['coordinates'], data['R_pc'], data['e_vlos_kms'], data['vlos_kms'],
                (nodes+1)/2, weights/2, (nodes+1)/2, weights/2,
                np.zeros(6), np.zeros(len(data['R_pc'])), np.zeros((len(data['R_pc']), 6)),
            ]]
            status = library.evaluate(
                len(data['R_pc']), 128, 128, 128, 1e5, 29.,
                *[value.ctypes.data_as(pointer) for value in arrays],
            )
            assert status == 0, (case, point['label'], status)
            actual = float(arrays[-3][1])
            error = actual - point['binary128_log_prior']
            qr_disagreement = float(arrays[-3][4])
            assert abs(error) < 1e-11, (case, point['label'], error)
            assert qr_disagreement < 1e-16, (case, point['label'], qr_disagreement)
            row = dict(case=case, label=point['label'], log_prior=actual,
                       difference_from_saved_reference=error,
                       householder_vs_reorthogonalized_qr=qr_disagreement)
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(
        arithmetic='GCC __float128 / libquadmath; results rounded to binary64',
        numpy_version=np.__version__,
        source_sha256=hashlib.sha256((ROOT/'reference.cpp').read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256((ROOT/'fisher_cases.json').read_bytes()).hexdigest(),
        rows=rows, elapsed_seconds=time.monotonic()-start,
    )
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
