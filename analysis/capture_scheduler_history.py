"""Capture a dated, read-only Slurm state snapshot for the history ledger.

The run ledger answers "what config was meant to run?".  This snapshot answers
the smaller question "what did Slurm report?".  Neither proves that a result is
scientifically valid.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FIELDS = ("JobIDRaw", "JobID", "State", "ExitCode", "Elapsed", "Start", "End")


def _chunks(values, size=200):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _known_job_ids(root: Path) -> list[str]:
    path = root / "experiments/history/runs.jsonl"
    ids = {
        str(row["scheduler_job_id"])
        for line in path.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("scheduler_job_id")
    }
    extras = json.loads(
        (root / "experiments/history/scheduler-extra-jobs.json").read_text()
    )
    for group in extras["groups"]:
        ids.update(str(x) for x in group.get("job_ids", []))
        for first, last in group.get("job_ranges", []):
            ids.update(str(x) for x in range(first, last + 1))
    return sorted(ids)


def capture(root: Path) -> dict:
    records = {}
    queries = _known_job_ids(root)
    for chunk in _chunks(queries):
        proc = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                ",".join(chunk),
                "--format=" + ",".join(FIELDS),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            values = line.split("|")
            if len(values) != len(FIELDS):
                continue
            row = dict(zip(FIELDS, values))
            row["State"] = row["State"].split()[0].rstrip("+")
            records[row["JobIDRaw"]] = row
    rows = [records[key] for key in sorted(records, key=lambda x: (len(x), x))]
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "query_job_ids": len(queries),
        "returned_records": len(rows),
        "state_counts": dict(sorted(Counter(row["State"] for row in rows).items())),
        "warning": "Slurm state proves execution state only, not scientific validity.",
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = capture(args.root.resolve())
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
