"""Materialize immutable historical sources and configs, never today's defaults."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent
OLD = Path('/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning')
SEALED = OLD / '.claude/worktrees/jaxnav-decay-carry/work/decay_20261007_v1'
WORK = ROOT / 'work/value-score-replay'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, expected in json.loads((SEALED/'hashes.json').read_text()).items():
        if not (name.startswith(('fixed/src/', 'vendor/')) or name == 'scripts/decay_buffer.py'):
            continue
        source = SEALED/name
        assert digest(source) == expected, name
        dest = WORK/'jaxnav'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        hashes[str(dest.relative_to(WORK))] = expected
    revision = subprocess.check_output(['git', 'rev-parse', '807367d'], cwd=ROOT, text=True).strip()
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision, 'src'], cwd=ROOT, text=True).splitlines()
    for name in paths:
        if not name.endswith('.py'):
            continue
        dest = WORK/'frozenlake'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT))
        hashes[str(dest.relative_to(WORK))] = digest(dest)
    rows = []
    evidence = []
    for env in ('frozenlake', 'jaxnav'):
        for seed in range(3):
            if env == 'jaxnav':
                log = Path('/home/ad.msoe.edu/manchadoa/jaxnav_decay_logs')/f'280874_{33+seed}.out'
                cfg = next(json.loads(l[8:]) for l in log.open() if l.startswith('CONFIG: '))
                reference = [281092, 281093, 280874][seed]
            else:
                reference = 257599 + seed
                log = OLD/f'src/counterfactual_rl/agents/frozen_lake/runs/{reference}/metrics.log'
                cfg = {}
                for line in log.read_text().splitlines():
                    if line.startswith('# ') and ': ' in line:
                        key, value = line[2:].split(': ', 1)
                        if key in ('Map', 'Episodes') or key.startswith('FrozenLake'):
                            continue
                        try:
                            value = ast.literal_eval(value)
                        except (SyntaxError, ValueError):
                            pass
                        cfg[key] = value
            assert cfg['seed'] == seed
            evidence.append(dict(environment=env, seed=seed, run=reference, log=str(log), sha256=digest(log)))
            for arm in ('cce', 'value'):
                rows.append(dict(task=len(rows), environment=env, seed=seed, arm=arm,
                                 reference_run=reference, config=cfg))
    manifest = dict(rows=rows, source_hashes=hashes, evidence=evidence,
                    frozenlake_revision=revision,
                    frozenlake_source_status='reconstructed pre-run Git source; no historical source hash available',
                    jaxnav_source_status='all source files verified against sealed run hashes',
                    versions=json.loads((SEALED/'versions.json').read_text()),
                    driver_hashes={name:digest(BUNDLE/name) for name in
                                   ('run.py', 'value_scores.py', 'train.sbatch')})
    (BUNDLE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'{len(rows)} rows; {len(hashes)} pinned source files')


if __name__ == '__main__':
    main()
