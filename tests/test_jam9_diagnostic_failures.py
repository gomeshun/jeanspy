"""Diagnostic failures must preserve completed work without requiring JAM."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import warnings

import pytest

spec = importlib.util.spec_from_file_location(
    "diagnose_jam9_warnings", Path(__file__).resolve().parents[1] / "scripts/diagnose_jam9_warnings.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_later_exception_and_missing_warning_source_preserve_moments(tmp_path):
    output = tmp_path / "report.json"
    report = {"status": "running", "runs": []}
    save = lambda: module.atomic_save(output, report)
    module.run_configuration(report, save, {"nrad": 45}, lambda: ([[float("nan")]], False))
    def failing_call():
        warnings.warn_explicit("retained warning", RuntimeWarning, str(tmp_path / "missing.py"), 7)
        raise RuntimeError("deliberate test exception")
    module.run_configuration(report, save, {"nrad": 75}, failing_call)
    saved = json.loads(output.read_text())
    assert saved["runs"][0]["moments"] == [["nan"]]
    failed = saved["runs"][1]
    assert failed["status"] == "exception"
    assert "deliberate test exception" in failed["exception"]
    assert failed["warnings"][0]["message"] == "retained warning"
    assert "FileNotFoundError" in failed["warnings"][0]["source_read_exception"]


def test_timeout_keeps_completed_configuration(tmp_path):
    output = tmp_path / "report.json"
    module.atomic_save(output, {"status": "running", "runs": [
        {"status": "completed", "moments": [[1.0]]}, {"status": "running"}]})
    code = module.supervise([sys.executable, "-c", "import time; time.sleep(60)"], output, .1)
    saved = json.loads(output.read_text())
    assert code == 1
    assert saved["status"] == "stopped_time_limit"
    assert saved["runs"][0]["moments"] == [[1.0]]
    assert saved["runs"][1]["status"] == "stopped_time_limit"
    assert saved["elapsed_seconds"] < 10


def test_early_exit_is_not_reported_as_completed(tmp_path):
    output = tmp_path / "report.json"
    module.atomic_save(output, {"status": "running", "runs": []})
    assert module.supervise([sys.executable, "-c", "pass"], output, 10) == 1
    assert json.loads(output.read_text())["status"] == "worker_exited_without_completion"


def test_internal_worker_cannot_replace_an_old_report(tmp_path):
    output = tmp_path / "report.json"
    output.write_text('{"status": "completed", "runs": [{"moments": [[1.0]]}]}\n')
    before = output.read_bytes()
    with pytest.raises(ValueError, match="active supervisor"):
        module.worker(output)
    assert output.read_bytes() == before


@pytest.mark.skipif(sys.platform != "linux", reason="The frozen diagnostic targets Linux")
def test_address_space_bound_is_set_in_a_child():
    code = ("import importlib.util, resource; "
            f"s=importlib.util.spec_from_file_location('diagnostic', {str(Path(module.__file__))!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "bound=m.apply_address_space_bound(); "
            "assert 0 < bound <= 16*1024**3; "
            "assert resource.getrlimit(resource.RLIMIT_AS)==(bound,bound)")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


@pytest.mark.parametrize("name", ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"])
def test_thread_constraint_fails_before_numerical_import(name, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv(name, "2")
    with pytest.raises(ValueError, match=name):
        module.require_single_thread()
