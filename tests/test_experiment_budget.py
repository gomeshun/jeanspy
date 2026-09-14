"""Campaign bounds include failed executions and survive interrupted accounting."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "validation/release/budget_protocol.json"


def invoke(tmp_path, label="test"):
    return subprocess.run([sys.executable, str(ROOT / "scripts/run_budgeted_experiment.py"),
        "--ledger", str(tmp_path / "ledger.json"), "--label", label,
        "--protocol", str(PROTOCOL), "--record", str(tmp_path / (label + ".json")),
        "--stdout", str(tmp_path / (label + ".txt")), "--seconds", "2", "--",
        sys.executable, "-c", "raise SystemExit(3)"], capture_output=True, text=True, timeout=15)


def test_failed_experiment_is_charged_and_cannot_be_replaced(tmp_path):
    result = invoke(tmp_path)
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert result.returncode != 0
    assert ledger["charged_seconds"] > 0
    assert ledger["experiments"][0]["experiment_status"] == "failed"
    original = (tmp_path / "ledger.json").read_bytes()
    assert invoke(tmp_path).returncode != 0
    assert (tmp_path / "ledger.json").read_bytes() == original


def test_open_accounting_blocks_new_child(tmp_path):
    ledger = dict(budget_protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        total_seconds=86400, charged_seconds=0, experiments=[dict(label="old",status="accounting_open")])
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    result = invoke(tmp_path)
    assert "accounting entry is still open" in result.stderr
    assert not (tmp_path / "test.json").exists()


def test_remaining_budget_blocks_new_child(tmp_path):
    ledger = dict(budget_protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        total_seconds=86400, charged_seconds=86390, experiments=[])
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    result = invoke(tmp_path)
    assert "exceeds remaining" in result.stderr
    assert not (tmp_path / "test.json").exists()
