#!/usr/bin/env python3
"""Render retained, synchronized timings without running a benchmark."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'validation/release/campaign'
OUT = ROOT / 'docs/source/_static/validation'
CONDITIONS = ['cpu-float64', 'gpu-float64', 'cpu-float32', 'gpu-float32']
COLORS = ['#24649c', '#b86613', '#317b58', '#9452a1']
MARKERS = ['o', 's', '^', 'D']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.hashsalt': 'jeanspy-release-performance-v1'})
manifest = {'inputs': [], 'outputs': [],
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def record(path, kind):
    manifest[kind].append({'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


def read(name):
    path = SOURCE / name
    record(path, 'inputs')
    return json.loads(path.read_text())


def write(path, text):
    path.write_text(text)
    record(path, 'outputs')


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png', 'svg']:
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else ({'Date': None} if ext == 'svg' else {})
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, dpi=190, metadata=metadata)
        if ext == 'svg':
            path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines()) + '\n')
        record(path, 'outputs')
    plt.close(fig)


rows = []
memory = []
failures = []
for condition in CONDITIONS:
    for kind in ['spherical', 'axisymmetric']:
        stem = f'spherical-performance-{condition}-v2' if kind == 'spherical' else f'performance-{condition}'
        data = read(stem + '.json')
        supervisor = read(stem + ('-supervisor.json' if kind == 'spherical' else '-v1-supervisor.json'))
        memory.append({'kind': kind, 'condition': condition,
            'host_rss_gib': supervisor['peak_host_rss_bytes'] / 2**30,
            'gpu_gib': supervisor['peak_gpu_bytes'] / 2**30})
        for case in data['cases']:
            if case['kind'] == kind:
                if not case['status'].startswith('completed_'):
                    raise ValueError(f'Cannot plot an incomplete timing: {stem} {case["n_data"]}')
                rows.append({'kind': kind, 'condition': condition, 'source': stem + '.json', **case})
            elif case['status'] == 'execution_failed':
                failures.append({'condition': condition, 'kind': case['kind'],
                    'n_data': case['n_data'], 'source': stem + '.json', 'exception': case['exception']})

fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.7), layout='constrained')
for col, kind in enumerate(['spherical', 'axisymmetric']):
    for condition, color, marker in zip(CONDITIONS, COLORS, MARKERS):
        selected = [r for r in rows if r['kind'] == kind and r['condition'] == condition]
        for row_index, key in enumerate(['jax_warm_prediction', 'warm_value_and_gradient']):
            summaries = [r[key] if row_index == 0 else next(g[key] for g in r['gradient_workloads'] if g['count'] == 4) for r in selected]
            x = np.array([r['n_data'] for r in selected])
            y = np.array([s['median_seconds'] for s in summaries])
            low = np.array([s['p10_seconds'] for s in summaries])
            high = np.array([s['p90_seconds'] for s in summaries])
            ax = axes[row_index, col]
            ax.plot(x, y, color=color, marker=marker, label=condition, ms=4)
            ax.fill_between(x, low, high, color=color, alpha=.14)
            ax.set(xscale='log', yscale='log', xticks=[8, 64, 256, 1024])
            ax.set_xticklabels(['8', '64', '256', '1024'])
            ax.grid(axis='y', color='#e8e8e8', linewidth=.5)
    reference = [r for r in rows if r['kind'] == kind and r['condition'] == 'cpu-float64']
    axes[0, col].plot([r['n_data'] for r in reference], [r['numpy_warm_prediction']['median_seconds'] for r in reference],
        color='#444444', linestyle='--', marker='x', ms=4, label='NumPy float64 (5 repeats)')
    axes[0, col].set_title(kind.capitalize() + (' (corrected v2)' if kind == 'spherical' else ' (primary v1)'), loc='left', fontsize=11)
    axes[1, col].set_xlabel('Number of evaluation positions')
axes[0, 0].set_ylabel('Warm prediction time [s]')
axes[1, 0].set_ylabel('Warm value + gradient time [s]\n4 active physical parameters')
axes[0, 0].legend(fontsize=7.4, frameon=False, loc='upper left')
save(fig, 'forward_performance')

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), layout='constrained')
for ax, kind in zip(axes, ['spherical', 'axisymmetric']):
    for condition, color, marker in zip(CONDITIONS, COLORS, MARKERS):
        row = next(r for r in rows if r['kind'] == kind and r['condition'] == condition and r['n_data'] == 256)
        gs = row['gradient_workloads']
        ax.plot([g['count'] for g in gs], [g['warm_value_and_gradient']['median_seconds'] for g in gs],
            color=color, marker=marker, label=condition, ms=4)
    ax.set(title=kind.capitalize(), xlabel='Active physical parameters', ylabel='Warm value + gradient time [s]', yscale='log')
    ax.set_xticks([1, 2, 4] if kind == 'spherical' else [1, 2, 4, 10])
    ax.grid(axis='y', color='#e8e8e8', linewidth=.5)
axes[0].legend(fontsize=7.5, frameon=False, ncol=2, loc='center', bbox_to_anchor=(.53,.63))
save(fig, 'gradient_parameter_performance')

lines = ['# Synchronized forward-model performance', '',
    'These measurements evaluate fixed physical points and synthetic coordinates; they do not measure posterior convergence. Each JAX result is synchronized before the timer stops. Shading shows the 10th–90th percentiles of 20 repeated evaluations, not a confidence interval over independent hardware sessions. The NumPy reference uses five repetitions. Spherical and axisymmetric panels use different vertical scales; compare their labeled values.', '',
    'The primary protocol was fixed in commit `704b762`. Its 16 spherical attempts failed because the harness omitted the required classical NFW truncation parameter. Those exceptions remain in the primary records. The corrected protocol and runner, fixed in commit `c74aeb7` after diagnosing this setup error, explicitly set the same infinite cutoff for both backends and repeat only the spherical measurements. No runtime library code, physical value, timing repetition or acceptance threshold was changed.', '',
    '```{figure} ../_static/validation/forward_performance.svg',
    ':alt: Warm prediction and four-parameter likelihood-gradient timings versus data count, for four device and precision conditions.',
    ':width: 100%', '',
    'Prediction and scalar Gaussian log-likelihood gradient timings for spherical NFW and axisymmetric Zhao models. The NumPy spherical double-exponential and JAX transformed quadratures solve the same physical problem with different numerical rules. Axisymmetric backends use the same 32-by-32-by-32 rule. The CPU is an AMD Threadripper 3990X restricted to logical CPUs 0–7; the GPU is one RTX 3090. BLAS/OpenMP thread counts are one.', '```', '',
    'All 32 completed geometry/data-count/condition cases pass their declared checks: finite positive values, at most 0.5% NumPy/JAX prediction differences, finite derivatives, and, for axisymmetry, at most 0.5% order-32/order-64 prediction differences at up to 16 fixed positions. These checks do not replace the broader [physical-gradient study](gradients.md), which finds nonfinite float32 axisymmetric gradients in separate tests with different physical points and quadrature orders.', '',
    '## Measured stages at 256 positions', '',
    'All entries below are milliseconds. First-call time includes tracing, compilation and execution. Host input transfer and result transfer are measured separately; CPU device-put is also reported. Warm values are medians. The gradient has four active physical parameters.', '',
    '| Geometry | Condition | Host preparation | Input transfer | First prediction | Warm prediction | First value + gradient | Warm value + gradient | Prediction to host |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
selected = [r for r in rows if r['n_data'] == 256]
for row in selected:
    g = next(g for g in row['gradient_workloads'] if g['count'] == 4)
    values = [row['host_preparation_seconds'], row['input_host_to_device_seconds'], row['jax_first_prediction_seconds'],
        row['jax_warm_prediction']['median_seconds'], g['first_value_and_gradient_seconds'],
        g['warm_value_and_gradient']['median_seconds'], row['prediction_device_to_host_seconds']]
    lines.append(f"| {row['kind']} | {row['condition']} | " + ' | '.join(f'{1000*v:.4g}' for v in values) + ' |')
lines += ['', '```{figure} ../_static/validation/gradient_parameter_performance.svg',
    ':alt: Value and gradient timings versus number of active parameters at 256 positions.', ':width: 100%', '',
    'Changing the differentiated subset leaves the physical point and data fixed. Nearly constant axisymmetric costs across 1–10 parameters reflect this reverse-mode workload and are not a general scaling guarantee for additional model components.', '```', '',
    '## Complete warm timing table', '',
    'All times are milliseconds. NumPy timings are float64 in every condition; its repeated measurements alongside different JAX conditions are retained as measured.', '',
    '| Geometry | Condition | Positions | NumPy prediction | JAX prediction | JAX value + gradient (4 parameters) | Max NumPy/JAX relative error |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for row in rows:
    grad = next(g for g in row['gradient_workloads'] if g['count'] == 4)
    values = [row['numpy_warm_prediction']['median_seconds'], row['jax_warm_prediction']['median_seconds'], grad['warm_value_and_gradient']['median_seconds']]
    lines.append(f"| {row['kind']} | {row['condition']} | {row['n_data']} | " + ' | '.join(f'{1000*v:.4g}' for v in values) + f" | {row['numpy_jax_max_relative_difference']:.3g} |")
lines += ['', '## Memory scope and reproducibility', '',
    'These are supervisor-observed process high-water marks across all four sizes and all parameter subsets in one condition. GPU allocation includes allocator/runtime reservations. They are not isolated incremental kernel requirements. The primary axisymmetric condition also includes the retained failed spherical setup attempts.', '',
    '| Geometry | Condition | Peak host RSS [GiB] | Observed GPU allocation [GiB] |', '| --- | --- | ---: | ---: |']
for row in memory:
    lines.append(f"| {row['kind']} | {row['condition']} | {row['host_rss_gib']:.3f} | {row['gpu_gib']:.3f} |")
lines += ['', 'Every repetition, warning, first call, transfer, model-construction cost, derivative and source hash is retained in `validation/release/campaign/performance-*.json` and `spherical-performance-*-v2.json`; supervisor records give process limits and memory observations. `validation/release/performance_summary.json` ties the displayed rows to those records. Regenerate the public figures and this page with `python scripts/render_release_performance.py`. The source archives and reproduction commands on the [retained benchmark source page](retained-sources.md) preserve the original runner and runtime hashes. Current runtime code is equivalent after removing docstrings, as separately checked; these timings were not rerun on the release branch. New experiments must respect the existing cumulative-budget boundary.', '']
write(ROOT / 'docs/source/validation/performance.md', '\n'.join(lines))
write(ROOT / 'validation/release/performance_summary.json', json.dumps({'rows': rows, 'memory': memory, 'primary_setup_failures': failures}, indent=2) + '\n')

write(ROOT / 'validation/release/performance_figure_manifest.json', json.dumps(manifest, indent=2) + '\n')
print(f'Rendered {len(rows)} completed cases; retained {len(failures)} original setup failures.')
