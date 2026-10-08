"""SLURM submission throttle — limits concurrent queued/running jobs per user."""

import getpass
import subprocess
import time


def count_my_queued_jobs(job_ids=None):
    """Return number of jobs currently pending or running under this user.

    Args:
        job_ids: If provided, only count jobs whose IDs are in this set.
                 If None, count all user jobs (original behaviour).
    """
    result = subprocess.run(
        ['squeue', '-u', getpass.getuser(), '-h', '-t', 'PENDING,RUNNING', '-o', '%i'],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if job_ids is not None:
        lines = [l for l in lines if l in job_ids]
    return len(lines)


def wait_for_slot(max_concurrent, poll_interval=30, job_ids=None):
    """Block until queued/running job count is below max_concurrent.

    Args:
        max_concurrent: Maximum jobs allowed in squeue at once.
        poll_interval:  Seconds between squeue polls (default 30).
        job_ids:        If provided, only count jobs in this set (experiment-scoped throttle).
    """
    while True:
        n = count_my_queued_jobs(job_ids=job_ids)
        if n < max_concurrent:
            return
        print(f"  [{n} jobs queued, limit={max_concurrent}] waiting {poll_interval}s ...",
              flush=True)
        time.sleep(poll_interval)
