"""Audit mass/gradient parity and the limits of the small-radius rewrite.

Run from the repository root; --baseline-ref must contain the pre-change source.
This is deterministic numerical validation, not a sampler or coverage study.
"""
import argparse
from contextlib import contextmanager
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

from jeanspy import _zhao, model_jax

ROOT = Path(__file__).resolve().parents[1]
BASELINE_REF = 'd4bace441fd4d1d5b39c11de68ff89e410542ffa'


@contextmanager
def mass_implementation(impl):
    original = model_jax._zhao_mass
    model_jax._zhao_mass = impl
    try:
        yield
    finally:
        model_jax._zhao_mass = original


def mass_audit(old):
    names = _zhao.PARAM_NAMES
    radii = jnp.array([.01, .1, .9, 1., 1.1, 10., 100.]) * 1000
    params = [np.array([1000., .01, a, b, g, 5e4]) for a, b, g in
              product([.5, 1., 3., 5.], [2., 3., 6., 10.], [0., 1., 2.9])]

    def functions(impl):
        def objective(v):
            return jnp.log(impl(radii, dict(zip(names, v)), xp=jnp))
        return jax.jit(objective), jax.jit(jax.jacfwd(objective))

    before, after = functions(old), functions(_zhao.enclosed_mass)
    max_value = max_gradient = max_numpy_value = 0.
    for v in params:
        a, b = np.asarray(before[0](v)), np.asarray(after[0](v))
        ja, jb = np.asarray(before[1](v)), np.asarray(after[1](v))
        max_value = max(max_value, float(np.max(abs(np.expm1(b-a)))))
        # Scale physical-parameter derivatives to comparable dimensionless units.
        scale = np.array([1000., .01, 1., 1., 1., 5e4])
        max_gradient = max(max_gradient, float(np.max(abs((jb-ja)*scale))))
        p = dict(zip(names, v))
        na = old(np.asarray(radii), p, xp=np)
        nb = _zhao.enclosed_mass(np.asarray(radii), p, xp=np)
        max_numpy_value = max(max_numpy_value, float(np.max(abs(nb/na-1))))

    edges = []
    for radius in [1e-6, 1e-3, .1, 1., 10.]:
        p = dict(rs_pc=1000., rhos_Msunpc3=.01, alpha=3., beta=6.,
                 gamma=0., r_t_pc=jnp.inf)
        values = {}
        for label, impl in [('before', old), ('after', _zhao.enclosed_mass)]:
            def objective(rs):
                return jnp.log(impl(radius, dict(p, rs_pc=rs), xp=jnp))
            values[label] = float(jax.jit(jax.grad(objective))(1000.)) * 1000.
        x = radius / 1000.
        exact = 3*x**3/(1+x**3)
        edges.append(dict(
            radius_pc=radius, exact=exact, **values,
            relative_error_before=abs(values['before']/exact-1),
            relative_error_after=abs(values['after']/exact-1),
        ))
    return dict(
        parameter_sets=len(params), radii_per_set=len(radii),
        max_relative_mass_change=max_value,
        max_absolute_dimensionless_score_change=max_gradient,
        max_relative_numpy_mass_change=max_numpy_value,
        core_edges=edges,
    )


def fisher_audit(old):
    fixture_path = ROOT / 'validation/zhao_derivative_stability/fisher_cases.json'
    fixtures = json.loads(fixture_path.read_text())
    report = {}
    for case, data in fixtures['cases'].items():
        outputs = {}
        for label, impl in [('before', old), ('after', _zhao.enclosed_mass)]:
            with mass_implementation(impl):
                model = model_jax.DSphModel(submodels={
                    'StellarModel': model_jax.PlummerModel(),
                    'DMModel': model_jax.ZhaoModel(),
                    'AnisotropyModel': model_jax.ConstantAnisotropyModel(),
                })

                def log_variance(t):
                    p = dict(rhos_Msunpc3=10**t[0], rs_pc=10**t[1], alpha=t[2],
                             beta=t[3], gamma=t[4], beta_ani=1-10**(-t[5]),
                             re_pc=29., r_t_pc=jnp.inf)
                    v = model.sigmalos2(
                        jnp.asarray(data['R_pc']), params=p, solver='kernel',
                        n_u=128, n_kernel=128, u_max=1e5, kernel_backend='jax',
                        dm_mass_method='numeric', dm_mass_n_steps=128,
                    )
                    return jnp.log(jnp.clip(v, 1e-12, 1e12) +
                                   jnp.asarray(data['e_vlos_kms'])**2)

                def evaluate(t):
                    v = log_variance(t)
                    jac = jax.jacfwd(log_variance)(t) / jnp.sqrt(2.)
                    scales = jnp.linalg.norm(jac, axis=0)
                    _, triangular = jnp.linalg.qr(jac/scales, mode='reduced')
                    logprior = (jnp.log(jnp.abs(jnp.diag(triangular))).sum() +
                                jnp.log(scales).sum() +
                                .5*jax.scipy.special.logsumexp(-v))
                    return v, jac, logprior

                fn = jax.jit(evaluate)
                values = []
                for point in data['points']:
                    v, jac, q = fn(jnp.asarray(point['coordinates']))
                    values.append((np.asarray(v), np.asarray(jac), float(q)))
                outputs[label] = values
        rows = []
        for i, point in enumerate(data['points']):
            a, b = outputs['before'][i], outputs['after'][i]
            ref = point['binary128_log_prior']
            rows.append(dict(
                label=point['label'], reference_log_prior=ref,
                before_log_prior=a[2], after_log_prior=b[2],
                before_error=a[2]-ref, after_error=b[2]-ref,
                max_relative_total_variance_change=float(np.max(abs(np.expm1(b[0]-a[0])))),
                max_absolute_score_change=float(np.max(abs(b[1]-a[1]))),
            ))
        report[case] = rows
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default=BASELINE_REF)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    code = subprocess.check_output(
        ['git', 'show', f'{args.baseline_ref}:src/jeanspy/_zhao.py'], cwd=ROOT, text=True)
    module = types.ModuleType('baseline_zhao')
    exec(compile(code, 'baseline_zhao.py', 'exec'), module.__dict__)
    start = time.monotonic()
    report = dict(
        baseline_ref=args.baseline_ref, backend=jax.default_backend(),
        device=str(jax.devices()[0]), jax_version=jax.__version__,
        numpy_version=np.__version__, x64=bool(jax.config.jax_enable_x64),
        baseline_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
        revised_source_sha256=hashlib.sha256((ROOT/'src/jeanspy/_zhao.py').read_bytes()).hexdigest(),
        mass=mass_audit(module.enclosed_mass),
        fisher=fisher_audit(module.enclosed_mass),
    )
    report['elapsed_seconds'] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
