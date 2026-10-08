"""Recover manifests and declared study arms without launching any job.

Overrides are retained as overrides. Never fill unknown run-time defaults.
"""
import hashlib
import base64
import json
import runpy
import subprocess
from pathlib import Path
from port_sources import ROOT, STUDY, add


def main():
    roots = [Path(line[9:]) for line in subprocess.check_output(
        ['git', 'worktree', 'list', '--porcelain'], text=True).splitlines() if line.startswith('worktree ')]
    files, seen, manifests, scans = {}, {}, [], []
    for root in roots:
        for env in ('smax', 'chess', 'cvrp', 'cvrp_neural'):
            directory = root / 'src/counterfactual_rl/agents' / env / 'experiments'
            paths = sorted(directory.rglob('*.json')) if directory.exists() else []
            scans.append(dict(path=str(directory), exists=directory.exists(), files=len(paths)))
            for path in paths:
                if path.stat().st_size > 2_000_000:
                    manifests.append(dict(source=str(path), status='blocked-size-cap'))
                    continue
                content = path.read_text()
                digest = hashlib.sha256(content.encode()).hexdigest()
                dest = STUDY / 'manifests' / f'{digest}.json'
                if digest not in seen:
                    files[dest] = content.rstrip('\n') + '\n'
                    exact = dest.with_suffix('.bytes.json')
                    files[exact] = json.dumps(dict(sha256=digest, base64=base64.b64encode(content.encode()).decode()), indent=2) + '\n'
                    seen[digest] = str(exact)
                data = json.loads(content)
                rows = list(data.values()) if isinstance(data, dict) else data
                config_rows = [r for r in rows if isinstance(r, dict) and 'algorithm' in r]
                manifests.append(dict(environment=env, source=str(path), archive=seen[digest], sha256=digest,
                    rows=len(config_rows), seeds=sorted({r['seed'] for r in config_rows if 'seed' in r}),
                    arms=sorted({json.dumps({k: r[k] for k in ('algorithm', 'mu', 'priority_mixing', 'mu_c', 'mu_delta', 'consequence_metric') if k in r}, sort_keys=True) for r in config_rows}),
                    code_status='manifest-recovered', evidence_status='submission record; results not revalidated',
                    config_status='overrides only; exact run-time defaults/source revision unresolved'))
        directory = root / 'paper/repro/manifests'
        for path in sorted(directory.glob('*main*json')):
            content = path.read_text()
            digest = hashlib.sha256(content.encode()).hexdigest()
            dest = STUDY / 'manifests' / f'{digest}.json'
            if digest not in seen:
                files[dest] = content.rstrip('\n') + '\n'
                exact = dest.with_suffix('.bytes.json')
                files[exact] = json.dumps(dict(sha256=digest, base64=base64.b64encode(content.encode()).decode()), indent=2) + '\n'
                seen[digest] = str(exact)
            data = json.loads(content)
            manifests.append(dict(environment='smax', source=str(path), archive=seen[digest], sha256=digest,
                 rows=len(data), code_status='paper-manifest-recovered',
                 evidence_status='not revalidated; may overlap agent manifests', config_status='overrides-only'))
    declared = []
    for env, rel in [('smax', 'experiments.py'), ('chess', 'run_experiments.py')]:
        source = ROOT / f'src/counterfactual_rl/agents/{env}/{rel}'
        content = source.read_text()
        dest = STUDY / 'definitions' / env / rel
        files[dest] = content
        # Both inspected files only define functions/constants at import time.
        module = runpy.run_path(str(source))
        for name, spec in module['EXPERIMENTS'].items():
            overrides = module['generate_runs'](spec)
            declared.append(dict(environment=env, name=name, source=str(dest),
                sha256=hashlib.sha256(content.encode()).hexdigest(), rows=overrides,
                status='declared source study; submission and completion not implied'))
    history = json.loads((ROOT / STUDY / 'history-manifest.json').read_text())
    for row in history['files']:
        if row['source'].endswith('agents/cvrp/budget_experiments.py'):
            module = runpy.run_path(str(ROOT / row['archive']))
            declared.append(dict(environment='cvrp-budget', name='budget-dial', source=row['archive'],
                sha256=row['sha256'], rows=module['build_runs'](),
                status='declared source study; budget learner not ported'))
    if sum(len(v.encode()) for v in files.values()) > 10 * 1024**2:
        raise ValueError('10 MiB manifest cap')
    catalog = dict(generic_recipes={env: dict(path=f'configs/experiment/{env}-historical.json',
          status='generic recovered learner: three algorithms, seed 0; not a historical study replay')
          for env in ('smax', 'chess', 'cvrp', 'rpdqn')}, manifests=manifests, declared_studies=declared,
          recovered_logged_configs='studies/legacy-environments-migration/logged-recipe-coverage.json',
          saved_run_headers='studies/legacy-environments-migration/run-headers.json',
          scan=scans, gaps=['No exact source revision is bound to historical job IDs.',
          'Saved manifests often contain only overrides; they are not complete configs.',
          'Older SMAC needs StarCraft2, smac, old analyzer/PPO modules; archive only.',
          'Budget/time-window/stranding are separate cvrp_budget ports; exact header recovery is in logged-recipe-coverage.json. Neural routing is rpdqn.',
          'Logs/checkpoints outside scanned roots and deleted/remote-only manifests are not covered.',
          'Figure/results recovery belongs to the analysis study workstream.'])
    files[STUDY / 'study-catalog.json'] = json.dumps(catalog, indent=2) + '\n'
    fresh = {p: text for p, text in files.items() if not (ROOT / p).exists()}
    add(fresh)
    for p, text in files.items():
        if p in fresh:
            continue
        old = (ROOT / p).read_text()
        if old == text:
            continue
        if p != STUDY / 'study-catalog.json':
            raise ValueError(f'Unexpected archive change: {p}')
        patch = f'*** Begin Patch\n*** Update File: {p}\n@@\n'
        patch += ''.join('-' + line + '\n' for line in old.splitlines())
        patch += ''.join('+' + line + '\n' for line in text.splitlines())
        patch += '*** End Patch\n'
        subprocess.run(['apply_patch'], input=patch, text=True, cwd=ROOT, check=True, capture_output=True)
    print(f'{len(manifests)} manifest locations, {len(seen)} distinct contents, {len(declared)} declared studies')


if __name__ == '__main__':
    main()
