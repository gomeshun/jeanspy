"""Validate the release-recovery trust boundary without network or MCMC work."""
import io
import json
from pathlib import Path
import urllib.error
import urllib.request

import pytest

yaml = pytest.importorskip("yaml")
WORKFLOW = Path(__file__).parents[1] / ".github/workflows/docs.yml"
pytestmark = pytest.mark.skipif(
    not WORKFLOW.is_file(), reason="GitHub workflow definitions are checkout-only, not sdist inputs"
)


class UniqueKeysLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate workflow key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeysLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _workflow():
    return yaml.load(WORKFLOW.read_text(), Loader=UniqueKeysLoader)


def _resolve(tmp_path, monkeypatch, release, *, tag="v0.1.0", event="workflow_dispatch",
             network_error=None):
    step = next(s for s in _workflow()["jobs"]["build"]["steps"] if s.get("id") == "release")
    source = step["run"].split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    output = tmp_path / "output"
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps({"release": release}))
    for key, value in {"RELEASE_TAG": tag, "GITHUB_EVENT_NAME": event,
                       "GITHUB_EVENT_PATH": str(event_file), "GITHUB_OUTPUT": str(output),
                       "GITHUB_API_URL": "https://api.github.com",
                       "GITHUB_REPOSITORY": "gomeshun/jeanspy"}.items():
        monkeypatch.setenv(key, value)
    requests = []

    def fetch(request, timeout):
        requests.append(request.full_url)
        assert timeout == 30
        assert "Authorization" not in request.headers
        if network_error:
            raise network_error
        return io.StringIO(json.dumps(release))

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    exec(compile(source, str(WORKFLOW), "exec"), {})
    result = dict(line.split("=", 1) for line in output.read_text().splitlines())
    return result, requests


@pytest.fixture
def published():
    return {"tag_name": "v0.1.0", "draft": False, "published_at": "2026-10-02T04:33:30Z",
            "prerelease": False, "target_commitish": "5" * 40}


def test_manual_recovery_requires_published_identity(tmp_path, monkeypatch, published):
    result, requests = _resolve(tmp_path, monkeypatch, published)
    assert result == {"tag": "v0.1.0", "prerelease": "false", "target_commit": "5" * 40}
    assert requests == ["https://api.github.com/repos/gomeshun/jeanspy/releases/tags/v0.1.0"]


def test_release_event_uses_its_payload(tmp_path, monkeypatch, published):
    result, requests = _resolve(tmp_path, monkeypatch, published, event="release")
    assert result["tag"] == "v0.1.0"
    assert requests == []


def test_development_dispatch_remains_development(tmp_path, monkeypatch):
    result, requests = _resolve(tmp_path, monkeypatch, None, tag="")
    assert result == {"tag": "", "prerelease": "false", "target_commit": ""}
    assert requests == []


@pytest.mark.parametrize("tag", ["main", "../v0.1.0", "v0.1.0\n", "v0.1.0;echo bad"])
def test_invalid_tag_rejected(tmp_path, monkeypatch, published, tag):
    with pytest.raises(SystemExit, match="explicit supported version tag"):
        _resolve(tmp_path, monkeypatch, published, tag=tag)


@pytest.mark.parametrize("change", [{"draft": True}, {"published_at": None},
                                    {"tag_name": "v9.9.9"}, {"prerelease": "false"}])
def test_unpublished_or_mismatched_release_rejected(tmp_path, monkeypatch, published, change):
    with pytest.raises(SystemExit, match="existing published GitHub Release"):
        _resolve(tmp_path, monkeypatch, published | change)


def test_missing_release_is_not_created(tmp_path, monkeypatch, published):
    error = urllib.error.HTTPError("https://api.github.com/", 404, "Not Found", {}, None)
    with pytest.raises(urllib.error.HTTPError):
        _resolve(tmp_path, monkeypatch, published, network_error=error)


@pytest.mark.parametrize("tag", ["v0.1.0", "v0.2.0rc1"])
def test_prerelease_flag_is_preserved(tmp_path, monkeypatch, published, tag):
    result, _ = _resolve(tmp_path, monkeypatch,
                         published | {"tag_name": tag, "prerelease": True}, tag=tag)
    assert result["prerelease"] == "true"


def test_workflow_preserves_release_source_and_full_validation():
    workflow = _workflow()
    build = workflow["jobs"]["build"]
    steps = build["steps"]
    preserve = next(i for i, s in enumerate(steps) if s.get("name", "").startswith("Preserve documentation runner"))
    checkout = next(i for i, s in enumerate(steps) if s.get("name", "").startswith("Check out the released source"))
    install = next(i for i, s in enumerate(steps) if s.get("name") == "Install locked documentation environment")
    assert preserve < checkout < install
    assert steps[checkout]["with"]["ref"] == "refs/tags/${{ steps.release.outputs.tag }}"
    assert "$RUNNER_TEMP/jeanspy-run-quickstart.py" in steps[preserve]["run"]
    identity = next(s for s in steps if s.get("id") == "identity")
    assert 'DOCS_REF="$(git rev-parse HEAD)"' in identity["run"]
    assert 'test "$RELEASE_TAG" = "v$PACKAGE_VERSION"' in identity["run"]
    assert 'test "$DOCS_REF" = "$RELEASE_TARGET_COMMIT"' in identity["run"]
    assert "inputs.release_tag" in build["env"]["PYTEST_ADDOPTS"]
    for name in ("Execute MCMC notebooks and refresh their displayed outputs",
                 "Execute both Quickstart workflows and render their stored posteriors"):
        step = next(s for s in steps if s.get("name") == name)
        assert "steps.release.outputs.tag != ''" in step["if"]
    quickstart = next(s for s in steps if s.get("name", "").startswith("Execute both Quickstart"))
    assert '--source-root "$GITHUB_WORKSPACE" --example-timeout 1800' in quickstart["run"]
    assert '--tooling-commit "$JEANSPY_DOCS_TOOLING_COMMIT"' in quickstart["run"]
    publish = workflow["jobs"]["publish"]
    assert "github.ref == 'refs/heads/main'" in publish["if"]
    assert "github.event_name != 'pull_request'" in publish["if"]
    assert "inputs.release_tag" not in publish["if"]
    stage = next(s for s in publish["steps"] if s.get("name", "").startswith("Stage development"))
    assert stage["env"]["SOURCE_COMMIT"] == "${{ needs.build.outputs.commit }}"
    assert stage["env"]["IS_PRERELEASE"] == "${{ needs.build.outputs.prerelease }}"
    assert build["timeout-minutes"] == 90


def test_failure_artifact_contains_logs_only():
    step = next(s for s in _workflow()["jobs"]["build"]["steps"]
                if s.get("name") == "Retain Quickstart failure logs without raw chains")
    assert step["if"] == "failure()"
    assert step["with"]["path"].splitlines() == [
        "docs/_build/quickstart/run-*/*.stdout.txt", "docs/_build/quickstart/run-*/*.stderr.txt"]
