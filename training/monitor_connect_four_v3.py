"""Read-only persistent watch of this experiment's Slurm array and run records."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--mode', choices=['smoke','train'], required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1] / 'experiments/connect_four/fixed-eval-v3'
    missing = 0
    while True:
        q = subprocess.run(['squeue','-h','-r','-j',args.job,'-o','%i|%T|%M|%R'],capture_output=True,text=True)
        rows = q.stdout.strip().splitlines()
        runs = {}
        for d in (root/'runs').glob(f'{args.mode}_{args.job}_*'):
            record = {}
            for name in ['attempt.json','complete.json']:
                try:
                    data = json.loads((d/name).read_text())
                    record[name] = {k:data[k] for k in ['status','error','passed','steps','seconds'] if k in data}
                except (OSError, ValueError):
                    pass
            try:
                tail = (d/'progress.jsonl').read_text().splitlines()[-1]
                data = json.loads(tail)
                record['progress'] = {k:data[k] for k in ['block','steps','episodes','seconds','loss','fade']}
            except (OSError, ValueError, IndexError):
                pass
            runs[d.name] = record
        report = dict(time=time.time(), job=args.job, queue=rows, queue_error=q.stderr.strip(), runs=runs)
        print(json.dumps(report),flush=True)
        (root/'runs'/f'monitor-{args.job}-latest.json').write_text(json.dumps(report,indent=2)+'\n')
        missing = missing+1 if not rows else 0
        if missing >= 3:
            accounting = subprocess.run(['sacct','-j',args.job,'--format=JobID,State,Elapsed,ExitCode','-P'],capture_output=True,text=True)
            print(accounting.stdout,flush=True)
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
