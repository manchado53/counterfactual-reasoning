"""Re-run bounded source tests or a fresh common-runner smoke. No job submission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',required=True,type=Path)
    p.add_argument('--test',help='One pytest node ID; each subprocess has a 180 second cap')
    p.add_argument('--recipe',type=Path)
    p.add_argument('--task',type=int,default=0)
    args = p.parse_args()
    if bool(args.test) == bool(args.recipe):
        p.error('Choose --test or --recipe')
    root = Path(__file__).resolve().parents[2]
    out = Path(tempfile.mkdtemp(prefix='cce-small-transfer-check-'))
    env = dict(os.environ,PYTHONNOUSERSITE='1',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    prefix = 'import os; os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]); '
    if args.test:
        code = prefix+'import pytest; raise SystemExit(pytest.main('+repr(['-q',args.test])+'))'
    else:
        argv = ['experiments.run','train','--recipe',str(args.recipe),'--task',str(args.task),
                '--output',str(out/'run'),'--device','cpu','--smoke','--max-wall-seconds','175']
        code = prefix+'import sys,runpy; sys.argv='+repr(argv)+'; runpy.run_module("experiments.run",run_name="__main__")'
    started = time.monotonic()
    command = [str(args.python),'-c',code]
    try:
        result = subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=180)
        report = dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        report = dict(command=command,returncode=124,error='180-second subprocess timeout')
    report.update(seconds=time.monotonic()-started,output=str(out),max_cores=4)
    if (out/'run/attempt.json').exists():
        attempt = json.loads((out/'run/attempt.json').read_text())
        report['attempt_status'] = attempt['status']
        report['resolved_config'] = attempt['resolved_config']
        report['source_hashes'] = attempt['source_hashes']
    (out/'check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('command','source_hashes','resolved_config')},indent=2))
    raise SystemExit(report['returncode'])


if __name__ == '__main__':
    main()
