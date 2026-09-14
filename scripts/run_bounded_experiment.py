#!/usr/bin/env python3
"""Run one local experiment with recorded time/RSS/GPU-memory stop rules.

The child gets its own process group. Only this newly started, identity-checked
group can be stopped by this supervisor. No existing process is attached.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    root = Path(f"/proc/{pid}")
    text = (root / "stat").read_text()
    tail = text[text.rfind(")")+2:].split()
    return dict(pid=pid, birth_ticks=int(tail[19]), pgrp=int(tail[2]),
                argv=(root/"cmdline").read_bytes().replace(b"\0", b" ").decode(),
                cwd=str((root/"cwd").resolve()))


def rss_bytes(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])*1024
    return 0


def gpu_bytes(pid):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                             "--format=csv,noheader,nounits"], text=True,
                             capture_output=True, timeout=5, check=True)
    memory = 0
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            memory += int(fields[1])*1024**2
    return memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--host-gib", type=float, default=16.)
    parser.add_argument("--gpu-gib", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not Path("/proc/self/stat").is_file():
        parser.error("this resource supervisor requires Linux /proc")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.seconds <= 0 or args.host_gib <= 0:
        parser.error("a command and positive bounds are required")
    if args.record.exists() or args.stdout.exists():
        parser.error("record/output already exists; choose a new experiment path")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    if args.gpu_gib is not None:
        if args.gpu_gib <= 0:
            parser.error("GPU bound must be positive")
        # An unavailable monitor must fail before starting the experiment.
        gpu_bytes(os.getpid())
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=command,
        cwd=str(Path.cwd()), bounds=dict(seconds=args.seconds, host_rss_gib=args.host_gib,
                                          gpu_gib=args.gpu_gib),
        status="starting", peak_host_rss_bytes=0, peak_gpu_bytes=0,
        supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment={key:os.environ[key] for key in ["JEANSPY_JAX_PLATFORM", "JEANSPY_JAX_ENABLE_X64",
            "JAX_PLATFORMS", "JAX_ENABLE_X64", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR",
            "XLA_PYTHON_CLIENT_MEM_FRACTION"] if key in os.environ})
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    def save():
        temporary = args.record.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(args.record)
    with args.stdout.open("wb") as stream:
        started = time.monotonic()
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        initial = identity(child.pid)
        assert initial["pgrp"] == child.pid
        report.update(status="running", process_identity=initial)
        save()
        def stop(reason):
            if child.poll() is not None:
                return
            current = identity(child.pid)
            if (current["birth_ticks"], current["pgrp"]) != (initial["birth_ticks"], initial["pgrp"]):
                raise RuntimeError("process identity changed; refusing to send a signal")
            report.update(status=reason, stop_identity=current)
            save()
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                current = identity(child.pid)
                if (current["birth_ticks"], current["pgrp"]) != (initial["birth_ticks"], initial["pgrp"]):
                    raise RuntimeError("process identity changed before SIGKILL")
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        try:
            while child.poll() is None:
                elapsed = time.monotonic()-started
                try:
                    resident = rss_bytes(child.pid)
                    device = gpu_bytes(child.pid) if args.gpu_gib is not None else 0
                except FileNotFoundError:
                    if child.poll() is not None:
                        break
                    raise
                report["peak_host_rss_bytes"] = max(report["peak_host_rss_bytes"], resident)
                report["peak_gpu_bytes"] = max(report["peak_gpu_bytes"], device)
                report["elapsed_seconds"] = elapsed
                if elapsed > args.seconds:
                    stop("stopped_time_limit"); break
                if resident > args.host_gib*1024**3:
                    stop("stopped_host_memory_limit"); break
                if args.gpu_gib is not None and device > args.gpu_gib*1024**3:
                    stop("stopped_gpu_memory_limit"); break
                save()
                time.sleep(.5)
        except BaseException as error:
            report["supervisor_error"] = repr(error)
            stop("stopped_supervisor_error")
            raise
        finally:
            report.update(exit_code=child.poll(), elapsed_seconds=time.monotonic()-started,
                          finished_utc=datetime.now(timezone.utc).isoformat())
            if report["status"] == "running":
                report["status"] = "completed" if child.returncode == 0 else "failed"
            stream.flush()
            report["stdout_sha256"] = hashlib.sha256(args.stdout.read_bytes()).hexdigest()
            save()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
