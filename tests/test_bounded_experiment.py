"""The experiment supervisor may stop only the child group it just created."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="Linux process supervision")


@pytest.mark.parametrize("program,seconds,host_gib,expected", [
    ("print('done')", 5, 1, "completed"),
    ("import time; time.sleep(10)", .2, 1, "stopped_time_limit"),
    ("import time; time.sleep(10)", 5, .001, "stopped_host_memory_limit"),
])
def test_child_completion_and_predeclared_limits(tmp_path, program, seconds, host_gib, expected):
    record = tmp_path/"record.json"
    result = subprocess.run([sys.executable, str(ROOT/"scripts/run_bounded_experiment.py"),
        "--record", str(record), "--stdout", str(tmp_path/"stdout.txt"),
        "--seconds", str(seconds), "--host-gib", str(host_gib), "--", sys.executable,
        "-c", program], text=True, capture_output=True, timeout=15)
    report = json.loads(record.read_text())
    assert report["status"] == expected
    assert result.returncode == (0 if expected == "completed" else 1)
    if expected != "completed":
        assert report["stop_identity"]["birth_ticks"] == report["process_identity"]["birth_ticks"]
        assert report["stop_identity"]["pgrp"] == report["process_identity"]["pid"]
        assert report["exit_code"] < 0
