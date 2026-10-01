"""Freeze, validate, and submit the paired episode-carry pilot."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT.parent / 'jaxnav-temporal-replay-arms/work/temporal_replay_20260930_v2'
WORK = ROOT / 'work/episode_carry_pilot_20260930'
LOGS = Path.home() / 'jaxnav_logs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def prepare():
    assert not WORK.exists(), 'Refusing overwrite'
    from temporal_replay_suite import baseline
    old = json.loads((OLD / 'manifest.json').read_text())
    rows = []
    for arm in ['uniform', 'per', 'cce_mul', 'cce_add', 'cce_only']:
        for seed in [0, 1]:
            if arm == 'uniform':
                cfg, log, run = baseline(seed)
                cfg.update(explore_hold_steps=20)
            else:
                ref = next(r for r in old['rows'] if r['arm'] == arm and r['seed'] == seed and r['hold'] == 20)
                cfg, log, run = ref['config'].copy(), ref['parent_log'], ref['parent_run']
            cfg['experiment'] = 'episode_carry_pilot_20260930'
            assert cfg['n_envs'] == 256 and cfg['collect_steps'] == 32
            assert cfg['sparse_reward'] is False and cfg['goal_rew'] == 4. and cfg['coll_rew'] == -4.
            assert cfg['fixed_eval_seed'] == 20260923 and cfg['eval_episodes'] == 1000
            for variant in ['old', 'fixed']:
                rows.append(dict(task=len(rows), arm=arm, seed=seed, variant=variant,
                                 parent_log=log, parent_run=run, config=cfg.copy()))
    WORK.mkdir(parents=True)
    ignore = shutil.ignore_patterns('runs', '__pycache__', '*.pyc')
    shutil.copytree(OLD / 'runtime/vendor', WORK / 'vendor', ignore=ignore)
    for variant in ['old', 'fixed']:
        shutil.copytree(OLD / 'runtime/src', WORK / variant / 'src', ignore=ignore)
        source = (OLD / 'runtime/scripts/train_temporal_replay.py' if variant == 'old'
                  else ROOT / 'analysis/claim2/train_temporal_replay.py')
        shutil.copy2(source, WORK / variant / 'train_temporal_replay.py')
    for name in ['dqn_vectorized.py', 'consequence_dqn_vectorized.py']:
        rel = Path('src/counterfactual_rl/agents/jax_nav') / name
        shutil.copy2(ROOT / rel, WORK / 'fixed' / rel)
    (WORK / 'scripts').mkdir()
    for name in ['carry_pilot_suite.py', 'train_carry_pilot.py']:
        shutil.copy2(ROOT / 'analysis/claim2' / name, WORK / 'scripts' / name)
    shutil.copy2(ROOT / 'slurm/carry_pilot.sbatch', WORK / 'carry_pilot.sbatch')
    shutil.copy2(OLD / 'versions.json', WORK / 'versions.json')
    dump(WORK / 'manifest.json', dict(rows=rows, blocks=2000, eval_every_blocks=50,
         env_steps=16384000, old_runtime=str(OLD), source_commit=old['source_commit'],
         comparison='paired old/fixed within each arm; equal environment steps; episode-based epsilon unchanged',
         endpoint='mean success over last four scheduled checks; two seeds shown separately, no significance claim'))
    paths = [p for p in WORK.rglob('*') if p.is_file()]
    dump(WORK / 'hashes.json', {str(p.relative_to(WORK)): digest(p) for p in paths})
    verify(WORK)
    print(f'PREPARED: {WORK} (20 runs)', flush=True)


def verify(work):
    for name, expected in json.loads((work / 'hashes.json').read_text()).items():
        assert digest(work / name) == expected, f'Source drift: {name}'
    versions = json.loads((work / 'versions.json').read_text())
    assert sys.version == versions['python']
    for name, expected in versions['packages'].items():
        assert importlib.metadata.version(name) == expected, name


def seal(work):
    """Finalize generated launcher copies before ANY submission; never change a live suite."""
    assert not list(work.glob('*-jobs.json')), 'Cannot reseal a submitted suite'
    for name in ['carry_pilot_suite.py', 'train_carry_pilot.py']:
        shutil.copy2(ROOT / 'analysis/claim2' / name, work / 'scripts' / name)
    old_hashes = json.loads((OLD / 'hashes.json').read_text())
    for name, expected in old_hashes.items():
        assert digest(OLD / name) == expected, f'Parent snapshot drift: {name}'
    source_files = list((work / 'old/src').rglob('*.py'))
    changed = []
    for path in source_files:
        rel = path.relative_to(work / 'old')
        assert digest(path) == digest(OLD / 'runtime' / rel)
        if digest(path) != digest(work / 'fixed' / rel):
            changed.append(str(rel))
    assert sorted(changed) == [
        'src/counterfactual_rl/agents/jax_nav/consequence_dqn_vectorized.py',
        'src/counterfactual_rl/agents/jax_nav/dqn_vectorized.py']
    assert digest(work / 'old/train_temporal_replay.py') == digest(OLD / 'runtime/scripts/train_temporal_replay.py')
    paths = [p for p in work.rglob('*') if p.is_file() and p.name != 'hashes.json'
             and '__pycache__' not in p.parts and p.suffix != '.pyc']
    dump(work / 'hashes.json', {str(p.relative_to(work)): digest(p) for p in paths})
    verify(work)
    print('SEALED: old snapshot unchanged; fixed source differs in exactly two trainer files')


def monitor(work, job):
    prior = None
    while True:
        p = subprocess.run(['sacct', '-n', '-P', '-X', '-j', job, '--format=JobIDRaw,State,ExitCode'],
                           text=True, capture_output=True)
        errors = []
        for path in LOGS.glob(job + '_*.err'):
            with path.open('rb') as f:
                f.seek(max(0, path.stat().st_size - 65536))
                tail = f.read().decode(errors='replace')
            if any(s in tail for s in ['Traceback (most recent', 'OUT_OF_MEMORY', 'CUDA_ERROR', 'AssertionError']):
                errors.append(str(path))
        current = (p.stdout, p.returncode, errors)
        if current != prior:
            print(json.dumps(dict(time=time.time(), states=p.stdout, error_logs=errors, query_error=p.stderr)), flush=True)
            prior = current
        if p.returncode == 0 and p.stdout.strip() and not any(s in p.stdout for s in ['RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'REQUEUED', 'SUSPENDED']):
            return
        time.sleep(60)


def validate_smokes(work):
    rows = json.loads((work / 'manifest.json').read_text())['rows']
    reports = [json.loads((work / f"smoke-{r['task']}.json").read_text()) for r in rows if r['seed'] == 0]
    assert len(reports) == 10 and all(r['passed'] for r in reports)
    assert len({r['eval_bank_hash'] for r in reports}) == 1
    assert all(r['env_steps'] == 65536 for r in reports)


def submit(work, mode, dry, gated=False):
    verify(work)
    spec = json.loads((work / 'manifest.json').read_text())
    rows = [r for r in spec['rows'] if mode == 'train' or r['seed'] == 0]
    record = work / f'{mode}-jobs.json'
    assert not record.exists(), f'Already submitted: {record}'
    dependency = None
    if gated:
        assert mode == 'train'
        smoke_jobs = json.loads((work / 'smoke-jobs.json').read_text())
        assert smoke_jobs['status'] == 'submitted' and len(smoke_jobs['groups']) == 2
        dependency = 'afterok:' + ':'.join(g['job'] for g in smoke_jobs['groups'])
    elif mode == 'train' and not dry:
        validate_smokes(work)
    groups = []
    for group, partition, gpu, cap in [('baseline', 'teaching', 't4', 4), ('cce', 'dgxh100', 'h100', 4)]:
        tasks = [r['task'] for r in rows if (r['arm'].startswith('cce')) == (group == 'cce')]
        cmd = ['sbatch', '--parsable', '--no-requeue', '--account=undergrad_research',
               f'--partition={partition}', f'--gres=gpu:{gpu}:1', '--cpus-per-task=8', '--mem=48G',
               f'--time={"01:00:00" if mode == "smoke" else "7-00:00:00"}', '--exclude=dh-node12',
               f'--array={",".join(map(str, tasks))}%{cap}', f'--job-name=carry_{mode}_{group}',
               f'--output={LOGS}/%A_%a.out', f'--error={LOGS}/%A_%a.err',
               str(work / 'carry_pilot.sbatch'), str(work), mode]
        if dependency:
            cmd.insert(1, f'--dependency={dependency}')
        print(('DRY RUN: ' if dry else 'SUBMIT: ') + ' '.join(cmd), flush=True)
        if dry:
            continue
        if not groups:
            with record.open('x') as f:
                json.dump({'status': 'submitting', 'groups': []}, f)
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        assert job.isdigit()
        with (work / f'monitor-{job}.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, str(work / 'scripts/carry_pilot_suite.py'),
                                     'monitor', '--work', str(work), '--job', job],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        groups.append(dict(job=job, tasks=tasks, group=group, monitor_pid=proc.pid))
        dump(record, dict(status='submitted' if len(groups) == 2 else 'partial', groups=groups))
        print(f'SUBMITTED {job}; monitor PID {proc.pid}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'seal', 'submit', 'monitor'])
    p.add_argument('--work', type=Path, default=WORK)
    p.add_argument('--mode', choices=['smoke', 'train'], default='smoke')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--job')
    p.add_argument('--gated', action='store_true', help='Queue training behind both successful smoke arrays')
    a = p.parse_args()
    if a.command == 'prepare':
        prepare()
    elif a.command == 'seal':
        seal(a.work.resolve())
    elif a.command == 'submit':
        submit(a.work.resolve(), a.mode, a.dry_run, a.gated)
    else:
        monitor(a.work.resolve(), a.job)


if __name__ == '__main__':
    main()
