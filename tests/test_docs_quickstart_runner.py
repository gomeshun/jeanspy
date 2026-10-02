"""Documentation recovery must separate released sources from build tooling."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_quickstart.py"
    spec = importlib.util.spec_from_file_location("quickstart_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def released_source(tmp_path):
    root = tmp_path / "released-source"
    for name in ("examples/docs_quickstart_emcee.py", "examples/docs_quickstart_numpyro.py",
                 "examples/docs_quickstart_data.py", "src/jeanspy/__init__.py",
                 "scripts/run_quickstart.py", "uv.lock"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Released content: {name}\n")
    return root


def fake_outputs(backend, directory, environment, **kwargs):
    target = directory / backend
    target.mkdir()
    (target / "observations.csv").write_text("R_pc,vlos_kms,e_vlos_kms\n1,2,3\n")
    for name in ("trace", "autocorrelation", "posterior", "observations"):
        (target / f"{name}.png").write_bytes(b"fake test figure")
    return backend, f"Successful {backend} output\n"


def test_example_executes_selected_source_with_requested_timeout(runner, tmp_path, monkeypatch):
    source = tmp_path / "source"
    directory = tmp_path / "build"
    directory.mkdir()
    environment = {"EXAMPLE_ENV": "value"}
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "output\n", "warning\n")

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.run_example("numpyro", directory, environment, source_root=source,
                              example_timeout=1800) == ("numpyro", "output\n")
    command, options = calls[0]
    assert command == [runner.sys.executable, str(source / "examples/docs_quickstart_numpyro.py"),
                       "--output-dir", str(directory / "numpyro")]
    assert options == dict(cwd=source, env=environment, text=True,
                           capture_output=True, timeout=1800)
    assert (directory / "numpyro.stdout.txt").read_text() == "output\n"
    assert (directory / "numpyro.stderr.txt").read_text() == "warning\n"


@pytest.mark.parametrize("stdout,stderr,expected", [
    (b"partial\xff", b"warning", "partial\ufffd"),
    ("partial text", "warning", "partial text"),
    (None, None, ""),
])
def test_timeout_retains_partial_logs(runner, tmp_path, monkeypatch, stdout, stderr, expected):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=stdout, stderr=stderr)

    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="numpyro example timed out after 1800s"):
        runner.run_example("numpyro", tmp_path, {}, example_timeout=1800)
    assert (tmp_path / "numpyro.stdout.txt").read_text() == expected
    assert (tmp_path / "numpyro.stderr.txt").read_text() == ("warning" if stderr else "")


def test_failed_example_retains_logs(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 1, "before failure", "failure detail"))
    with pytest.raises(RuntimeError, match="emcee example failed"):
        runner.run_example("emcee", tmp_path, {})
    assert (tmp_path / "emcee.stdout.txt").read_text() == "before failure"
    assert (tmp_path / "emcee.stderr.txt").read_text() == "failure detail"


def test_external_helper_preserves_source_and_tooling_provenance(
        runner, released_source, tmp_path, monkeypatch):
    helper = tmp_path / "newer-tooling/run_quickstart.py"
    helper.parent.mkdir()
    helper.write_text("Newer execution helper, outside released source\n")
    monkeypatch.setattr(runner, "__file__", str(helper))
    calls = []

    def run(backend, directory, environment, **kwargs):
        calls.append((directory, environment, kwargs))
        return fake_outputs(backend, directory, environment, **kwargs)

    monkeypatch.setattr(runner, "run_example", run)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "test-version")
    runner.main(source_root=released_source, example_timeout=1800, tooling_commit="a" * 40)
    destination = released_source / "docs/source/_static/quickstart"
    metadata = json.loads((destination / "execution.json").read_text())
    source_runner = released_source / "scripts/run_quickstart.py"
    assert metadata["source_sha256"]["scripts/run_quickstart.py"] == hashlib.sha256(
        source_runner.read_bytes()).hexdigest()
    assert metadata["tooling"] == {
        "runner_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "commit": "a" * 40,
        "example_timeout_seconds": 1800,
    }
    assert metadata["source_sha256"]["scripts/run_quickstart.py"] != metadata["tooling"]["runner_sha256"]
    assert len(metadata["source_sha256"]) == 6
    for directory, environment, kwargs in calls:
        assert directory.parent == released_source / "docs/_build/quickstart"
        assert kwargs == dict(source_root=released_source, example_timeout=1800)
        assert environment["JEANSPY_JAX_PLATFORM"] == "cpu"
        assert environment["JEANSPY_JAX_ENABLE_X64"] == "true"
        assert environment["PYTHONUNBUFFERED"] == "1"
    assert (destination / "numpyro.txt").read_text() == "Successful numpyro output\n"
    assert not (helper.parent / "docs").exists()


@pytest.mark.parametrize("changed_file", ["source", "tooling"])
def test_source_or_tooling_mutation_rejected_before_publication(
        runner, released_source, tmp_path, monkeypatch, changed_file):
    helper = tmp_path / "runner.py"
    helper.write_text("Original helper\n")
    monkeypatch.setattr(runner, "__file__", str(helper))

    def run(backend, directory, environment, **kwargs):
        if backend == "emcee":
            target = released_source / "src/jeanspy/__init__.py" if changed_file == "source" else helper
            target.write_text("Changed during execution\n")
        return fake_outputs(backend, directory, environment, **kwargs)

    monkeypatch.setattr(runner, "run_example", run)
    with pytest.raises(RuntimeError, match="changed during execution"):
        runner.main(source_root=released_source)
    assert not (released_source / "docs/source/_static/quickstart").exists()


def test_different_observations_rejected_before_publication(runner, released_source, monkeypatch):
    def run(backend, directory, environment, **kwargs):
        result = fake_outputs(backend, directory, environment, **kwargs)
        if backend == "numpyro":
            (directory / backend / "observations.csv").write_text("different data")
        return result

    monkeypatch.setattr(runner, "run_example", run)
    with pytest.raises(RuntimeError, match="different mock observations"):
        runner.main(source_root=released_source)
    assert not (released_source / "docs/source/_static/quickstart").exists()


@pytest.mark.parametrize("timeout", [0, -1])
def test_invalid_timeout_rejected_before_execution(runner, tmp_path, timeout):
    with pytest.raises(ValueError, match="positive"):
        runner.main(tmp_path / "absent", example_timeout=timeout)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("commit", ["main", "a" * 39, "g" * 40, ""])
def test_invalid_tooling_commit_rejected_before_execution(runner, tmp_path, commit):
    with pytest.raises(ValueError, match="full 40-character"):
        runner.main(tmp_path / "absent", tooling_commit=commit)
    assert not (tmp_path / "absent").exists()
