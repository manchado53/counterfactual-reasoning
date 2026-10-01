"""Frozen dense temporal-exploration suite; no imports from the parent worktree."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT.parent / 'research+cce-robotics-transfer'
WORK = ROOT / 'work/temporal_replay_20260930_v2'
LOGS = Path.home() / 'jaxnav_logs'
COMMIT = '0165ecb2a255786b3e62b10a79e158904fb46a12'
ARMS = [('per', 5), ('cce_mul', 10), ('cce_add', 20), ('cce_only', 15)]
PACKAGES = ['jax', 'jaxlib', 'jaxmarl', 'flax', 'optax', 'numpy', 'scipy', 'chex']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def baseline(task):
    log = LOGS / f'278560_{task}.out'
    text = log.read_text()
    overrides = json.loads(next(x[8:] for x in text.splitlines() if x.startswith('CONFIG: ')))
    run = Path(re.search(r'Run dir: (.+)', text)[1])
    cfg = {}
    for line in (run / 'metrics.log').read_text().splitlines():
        if not line.startswith('# ') or ': ' not in line:
            continue
        key, raw = line[2:].split(': ', 1)
        if not re.fullmatch(r'[a-z_][a-z_0-9]*', key) and key != 'PER_parameters':
            continue
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        cfg[key] = value
    assert all(cfg[k] == v for k, v in overrides.items()), (task, cfg, overrides)
    assert (cfg['sparse_reward'], cfg['goal_rew'], cfg['coll_rew']) == (False, 4., -4.)
    assert cfg['fixed_eval_seed'] == 20260923 and cfg['n_episodes'] == 250000
    assert cfg['n_envs'] == 256 and cfg['collect_steps'] == 32
    tree = ast.parse((PARENT / 'src/counterfactual_rl/agents/jax_nav/config.py').read_text())
    defaults = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CONFIG' for t in n.targets))
    assert not set(defaults) - set(cfg), f'Missing saved defaults: {set(defaults) - set(cfg)}'
    assert cfg['PER_parameters'] == {'eps': .01, 'beta': .25, 'maximum_priority': 1.}
    return cfg, str(log), str(run)


def matrix():
    rows = []
    for arm, offset in ARMS:
        reference = None
        for seed in range(5):
            cfg, log, run = baseline(offset + seed)
            assert cfg['seed'] == seed
            common = {k: v for k, v in cfg.items() if k != 'seed'}
            if reference is not None:
                assert common == reference, (arm, seed)
            reference = common
            for hold in (1, 20):
                rows.append(dict(task=len(rows), arm=arm, seed=seed, hold=hold,
                                 parent_log=log, parent_run=run,
                                 config={**cfg, 'explore_hold_steps': hold,
                                         'experiment': 'dense_temporal_replay_20260930'}))
    assert len(rows) == 40
    return rows


def prepare():
    rows = matrix()
    if WORK.exists():
        raise RuntimeError(f'Refusing to replace existing suite: {WORK}')
    subprocess.run(['git', '-C', str(PARENT), 'diff', '--exit-code', COMMIT, '--', 'src'], check=True)
    WORK.mkdir(parents=True)
    archive = WORK / 'source.tar'
    subprocess.run(['git', '-C', str(PARENT), 'archive', '--format=tar',
                    f'--output={archive}', COMMIT, 'src'], check=True)
    runtime = WORK / 'runtime'
    runtime.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(runtime, filter='data')
    # Artifact copies, not edits to the original source or scripts.
    shutil.copy2(PARENT / 'analysis/claim2/train_temporal_exploration.py', WORK / 'historical_uniform_trainer.py')
    vendor = runtime / 'vendor'
    vendor.mkdir()
    jaxmarl = Path(importlib.util.find_spec('jaxmarl').origin).parent
    shutil.copytree(jaxmarl, vendor / 'jaxmarl', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    scripts = runtime / 'scripts'
    scripts.mkdir()
    for name in ['temporal_replay_suite.py', 'train_temporal_replay.py']:
        shutil.copy2(ROOT / 'analysis/claim2' / name, scripts / name)
    shutil.copy2(ROOT / 'slurm/temporal_replay.sbatch', WORK / 'temporal_replay.sbatch')
    versions = {p: importlib.metadata.version(p) for p in PACKAGES}
    dump(WORK / 'versions.json', {'python': sys.version, 'packages': versions})
    (WORK / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    conda = os.environ.get('CONDA_EXE') or str(Path.home() / 'miniconda3/bin/conda')
    if not Path(conda).exists():
        conda = shutil.which('conda') or str(Path.home() / 'anaconda3/bin/conda')
    (WORK / 'conda-explicit.txt').write_text(subprocess.check_output([conda, 'list', '--explicit', '-p', sys.prefix], text=True))
    dump(WORK / 'manifest.json', {'source_commit': COMMIT, 'branch': 'experiment/jaxnav-temporal-replay-arms',
                                 'test_seed': 202609301, 'rows': rows})
    hashes = {str(p.relative_to(WORK)): digest(p) for p in runtime.rglob('*') if p.is_file()}
    for name in ['source.tar', 'historical_uniform_trainer.py', 'manifest.json', 'versions.json', 'temporal_replay.sbatch']:
        hashes[name] = digest(WORK / name)
    dump(WORK / 'hashes.json', hashes)
    print(f'PREPARED {WORK}: 40 resolved configs, {len(hashes)} frozen files')


def verify(work):
    work = Path(work)
    for name, expected in json.loads((work / 'hashes.json').read_text()).items():
        assert digest(work / name) == expected, f'Source drift: {name}'
    versions = json.loads((work / 'versions.json').read_text())
    assert sys.version == versions['python']
    assert {p: importlib.metadata.version(p) for p in PACKAGES} == versions['packages']


def start_monitor(jobs, label):
    output = (WORK / f'monitor-{label}.log').open('a')
    p = subprocess.Popen([sys.executable, str(__file__), 'monitor', '--jobs', ','.join(jobs)],
                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                         env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    output.close()
    return p.pid


def submit(smoke, dry):
    verify(WORK)
    rows = json.loads((WORK / 'manifest.json').read_text())['rows']
    tasks = [r['task'] for r in rows if not smoke or r['seed'] == 0]
    record = WORK / ('smoke-jobs.json' if smoke else 'train-jobs.json')
    if record.exists():
        raise RuntimeError(f'Already submitted or partially submitted: {record}')
    if not smoke and not dry:
        sj = json.loads((WORK / 'smoke-jobs.json').read_text())
        for group in sj['groups']:
            for task in group['tasks']:
                report = json.loads((WORK / f'smoke-{task}.json').read_text())
                assert report['passed'] and report['task'] == task
                assert report['episodes'] >= 1000 and report['fixed_eval_n'] == 1000
    groups = []
    for name, partition, gres, selected, cap in [
        ('per', 'teaching', 'gpu:t4:1', [t for t in tasks if t < 10], 10),
        ('cce', 'dgxh100', 'gpu:h100:1', [t for t in tasks if t >= 10], 16),
    ]:
        cmd = ['sbatch', '--parsable', '--no-requeue', '--account=undergrad_research',
               f'--partition={partition}', f'--gres={gres}', '--cpus-per-task=8', '--mem=48G',
               f'--time={"01:00:00" if smoke else "7-00:00:00"}',
               '--exclude=dh-node12', f'--array={",".join(map(str, selected))}%{cap}',
               f'--job-name=hold_{"smoke" if smoke else "train"}_{name}',
               f'--output={LOGS}/%A_%a.out', f'--error={LOGS}/%A_%a.err',
               str(WORK / 'temporal_replay.sbatch'), str(WORK), 'smoke' if smoke else 'train']
        print(('DRY RUN ' if dry else 'SUBMIT ') + ' '.join(cmd), flush=True)
        if dry:
            continue
        if not groups:
            with record.open('x') as f:
                json.dump({'groups': [], 'status': 'submitting'}, f)
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        assert job.isdigit(), job
        monitor = start_monitor([job], f'{job}')
        groups.append(dict(name=name, job=job, tasks=selected, monitor_pid=monitor))
        dump(record, {'groups': groups, 'status': 'submitted' if len(groups) == 2 else 'partial'})
        print(f'SUBMITTED {job}, monitor PID {monitor}', flush=True)


def monitor(jobs):
    old = None
    while True:
        proc = subprocess.run(['sacct', '-n', '-P', '-j', jobs, '--format=JobIDRaw,State,ExitCode', '-X'],
                              text=True, capture_output=True)
        if proc.returncode:
            print('Monitor query failed: ' + proc.stderr, flush=True)
        else:
            lines = [x for x in proc.stdout.splitlines() if x.strip()]
            errors = []
            for job in jobs.split(','):
                for p in LOGS.glob(job + '_*.err'):
                    # Bounded tail; do not repeatedly scan huge progress-bar logs.
                    with p.open('rb') as f:
                        f.seek(max(0, p.stat().st_size - 65536))
                        tail = f.read().decode(errors='replace')
                    if any(s in tail for s in ['Traceback (most recent', 'CUDA_ERROR', 'OUT_OF_MEMORY', 'AssertionError']):
                        errors.append(str(p))
            state = (lines, errors)
            if state != old:
                print(json.dumps({'time': time.time(), 'jobs': lines, 'error_logs': errors}), flush=True)
                old = state
            active = ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'REQUEUED', 'SUSPENDED')
            if lines and not any(any(s in line for s in active) for line in lines):
                print('Monitor ended: all reported tasks terminal.', flush=True)
                return
        time.sleep(60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'dry-run', 'smoke', 'launch', 'monitor'])
    p.add_argument('--jobs')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'monitor':
        monitor(args.jobs)
    elif args.command == 'dry-run':
        submit(True, True)
        submit(False, True)
    else:
        submit(args.command == 'smoke', False)


if __name__ == '__main__':
    main()
