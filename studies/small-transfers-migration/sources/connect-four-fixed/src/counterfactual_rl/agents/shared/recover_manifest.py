"""Rebuild a sweep manifest from a run_experiments submission log.

Why: run_experiments.py writes its manifest only AFTER the last job is submitted.
A 1280-run sweep spends hours in that loop (the throttle blocks on free slots), so
if the submitting process dies the job_id -> config map is lost even though every
job is queued and running fine. The submission log already carries both halves —
each accepted job prints as:

    [ 55] Job 270278: {'map_name': '8x8', ..., 'slip_prob': 0.0, 'seed': 14}

so the manifest is recoverable at any point, including mid-submission.

Usage:
    python -m counterfactual_rl.agents.shared.recover_manifest \
        --log ~/graded_slip_logs/submit_dense.log --out /path/to/manifest.json
"""

import argparse
import ast
import json
import re

_LINE = re.compile(r"^\s*\[\s*\d+\]\s*Job\s+(\d+):\s*(\{.*\})\s*$")


def parse_log(log_path):
    """Return {job_id: overrides} for every successfully submitted job in the log."""
    manifest = {}
    with open(log_path) as f:
        for line in f:
            m = _LINE.match(line.rstrip("\n"))
            if not m:
                continue
            job_id, payload = m.group(1), m.group(2)
            # The log prints a Python dict repr, not JSON (single quotes, True/None).
            manifest[job_id] = ast.literal_eval(payload)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="run_experiments stdout log")
    ap.add_argument("--out", required=True, help="manifest JSON to write")
    args = ap.parse_args()

    manifest = parse_log(args.log)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Recovered {len(manifest)} jobs -> {args.out}")


if __name__ == "__main__":
    main()
