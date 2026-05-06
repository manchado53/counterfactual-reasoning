"""SLURM submission throttle — limits concurrent queued/running jobs per user."""

import os
import subprocess
import time


def count_my_queued_jobs():
    """Return number of jobs currently pending or running under this user."""
    result = subprocess.run(
        ['squeue', '-u', os.environ.get('USER', ''), '-h', '-t', 'PENDING,RUNNING', '-o', '%i'],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    return len(lines)


def wait_for_slot(max_concurrent, poll_interval=30):
    """Block until queued/running job count is below max_concurrent.

    Args:
        max_concurrent: Maximum jobs allowed in squeue at once.
        poll_interval:  Seconds between squeue polls (default 30).
    """
    while True:
        n = count_my_queued_jobs()
        if n < max_concurrent:
            return
        print(f"  [{n} jobs queued, limit={max_concurrent}] waiting {poll_interval}s ...",
              flush=True)
        time.sleep(poll_interval)
