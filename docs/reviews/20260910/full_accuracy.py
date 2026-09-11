"""Run the repository's full 246-case accuracy matrix with shared stateless models.

Only model construction is cached by anisotropy type to avoid compiling an
identical JAX function for every parameter dictionary. Reference settings,
candidate grids, parameter values, error metric and enforcement are unchanged.
"""
import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('accuracy_contract', root / 'scripts/benchmark_sigmalos2_accuracy_contract.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
original = module._make_model
models = {}

def shared_model(case):
    if case.anisotropy not in models:
        models[case.anisotropy] = original(case)
    return models[case.anisotropy]

module._make_model = shared_model
print('Reuse stateless models; all numerical settings unchanged.', flush=True)
module.run(module._full_cases(), enforce=True)
