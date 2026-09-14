#!/usr/bin/env python3
"""Serialize bounded experiments and account for a fixed cumulative budget.

An interrupted accounting entry blocks subsequent launches. Its elapsed time
must be reconciled from the preserved supervisor record before continuing;
starting a new ledger is not an extension of the approved campaign.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--budget-protocol", type=Path,
                        default=ROOT / "validation/release/budget_protocol.json")
    parser.add_argument("--label", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    budget = json.loads(args.budget_protocol.read_text())
    budget_hash = hashlib.sha256(args.budget_protocol.read_bytes()).hexdigest()
    if budget["total_seconds"] != 86400 or args.seconds <= 0 or not command:
        parser.error("this campaign has one fixed 86400-second budget; provide a positive bound and command")
    if not args.protocol.is_file() or args.record.exists() or args.stdout.exists():
        parser.error("a frozen protocol and unused output paths are required")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.with_suffix(".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another experiment owns the campaign lock")
        if args.ledger.exists():
            ledger = json.loads(args.ledger.read_text())
            if ledger["budget_protocol_sha256"] != budget_hash:
                parser.error("the recorded budget protocol changed")
        else:
            ledger = dict(schema_version=1, campaign=budget["campaign"],
                budget_protocol_sha256=budget_hash, total_seconds=budget["total_seconds"],
                charged_seconds=0., experiments=[])
        if any(row["status"] == "accounting_open" for row in ledger["experiments"]):
            parser.error("an earlier accounting entry is still open; reconcile it before launching")
        if any(row["label"] == args.label for row in ledger["experiments"]):
            parser.error("experiment labels are immutable and cannot be reused")
        remaining = ledger["total_seconds"] - ledger["charged_seconds"]
        grace = budget["reserved_teardown_seconds"]
        if remaining < args.seconds + grace:
            parser.error(f"requested bound plus teardown reserve exceeds remaining {remaining:.3f} seconds")
        entry = dict(label=args.label, status="accounting_open", command=command,
            started_utc=datetime.now(timezone.utc).isoformat(),
            requested_seconds=args.seconds, teardown_reserve_seconds=grace,
            remaining_before_seconds=remaining, protocol=str(args.protocol.resolve()),
            protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
            supervisor_record=str(args.record.resolve()), stdout=str(args.stdout.resolve()))
        ledger["experiments"].append(entry)
        write_json(args.ledger, ledger)
        supervisor = [sys.executable, str(ROOT / "scripts/run_bounded_experiment.py"),
            "--record", str(args.record), "--stdout", str(args.stdout),
            "--seconds", str(args.seconds), "--host-gib", str(budget["max_host_rss_gib"])]
        if args.gpu:
            supervisor += ["--gpu-gib", str(budget["max_gpu_memory_gib"])]
        started = time.monotonic()
        try:
            result = subprocess.run(supervisor + ["--"] + command, check=False)
        except BaseException:
            # Leave the entry open. A missing final supervisor record or a
            # surviving process must not be treated as zero execution time.
            raise
        elapsed = time.monotonic() - started
        entry.update(status="accounted", elapsed_seconds=elapsed,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            supervisor_exit_code=result.returncode)
        if args.record.is_file():
            record = json.loads(args.record.read_text())
            entry["experiment_status"] = record["status"]
            entry["supervisor_record_sha256"] = hashlib.sha256(args.record.read_bytes()).hexdigest()
        else:
            entry["experiment_status"] = "supervisor_failed_before_record"
        ledger["charged_seconds"] += elapsed
        ledger["remaining_seconds"] = ledger["total_seconds"] - ledger["charged_seconds"]
        write_json(args.ledger, ledger)
        print(json.dumps(dict(label=args.label, elapsed_seconds=elapsed,
                              remaining_seconds=ledger["remaining_seconds"])))
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
