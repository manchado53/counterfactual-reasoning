"""Make recipes only when a log contains every setting accepted by a frozen port.

These replay logged settings, not a proven historical code revision. No missing
setting is filled from current defaults. Incompatible rows remain cataloged.
"""
import ast
import importlib
import json
import subprocess
from pathlib import Path
from port_sources import ROOT, STUDY, add


def main():
    recovered = {env: [] for env in ('smax', 'chess', 'cvrp', 'rpdqn', 'cvrp_budget')}
    rejected = []
    headers = json.loads((ROOT / STUDY / 'run-headers.json').read_text())['rows']
    for source in headers:
        env = source['environment']
        if env == 'cvrp_neural':
            env = 'rpdqn'
        if env not in recovered:
            continue
        cfg = {}
        for line in source['header'].splitlines():
            key, sep, value = line[1:].strip().partition(': ')
            if sep and key.isidentifier() and (key == key.lower() or key in ('M', 'B', 'C', 'PER_parameters')):
                try:
                    cfg[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    cfg[key] = value
        if env == 'cvrp' and (cfg.get('budget_mult') is not None or cfg.get('budget_units') is not None):
            env = 'cvrp_budget'
            cfg['learner_variant'] = 'option_a' if 'allow_stranding' in cfg else 'budget'
        cfg.update(collector_variant='historical', training_budget_kind='chunks' if env == 'chess' else 'episodes')
        build = importlib.import_module(f'experiments.learners.{env}.build')
        try:
            build.validate_config(cfg)
        except (ValueError, TypeError) as exc:
            rejected.append(dict(environment=env, run=source['run'], reason=str(exc),
                                 source=source['source'], header_sha256=source['header_sha256']))
            continue
        rows = recovered[env]
        row = dict(task=len(rows)+1, seed=cfg['seed'], collector='historical', config=cfg,
            source_run=source['run'], source_log=source['source'], source_header_sha256=source['header_sha256'])
        if env == 'cvrp_budget':
            manifest = STUDY / f'budget-{cfg["learner_variant"]}-source-manifest.json'
            row['port_source_manifest'] = str(manifest)
            row['port_source_commit'] = json.loads((ROOT / manifest).read_text())['commit']
            row['source_binding'] = 'config schema matches this pinned protocol; original job revision not proven'
        rows.append(row)
    files = {}
    for env, rows in recovered.items():
        if rows:
            files[Path(f'configs/experiment/{env}-logged-settings.json')] = json.dumps(dict(
                schema_version=1, environment=env, rows=rows,
                study_status='complete logged settings, runnable on frozen port; original job source linkage unverified',
                source_headers=str(STUDY / 'run-headers.json')), indent=2) + '\n'
    files[STUDY / 'logged-recipe-coverage.json'] = json.dumps(dict(
        accepted={k: len(v) for k, v in recovered.items()}, rejected=rejected,
        scope='Only exact config key sets accepted. No dropped flags or filled historical defaults. '
              'Port protocol metadata is new and explicit; original job code revision remains unresolved.'), indent=2) + '\n'
    fresh = {p: s for p, s in files.items() if not (ROOT / p).exists()}
    add(fresh)
    for p, s in files.items():
        if p in fresh:
            continue
        old = (ROOT / p).read_text()
        if old == s:
            continue
        patch = f'*** Begin Patch\n*** Update File: {p}\n@@\n'
        patch += ''.join('-' + line + '\n' for line in old.splitlines())
        patch += ''.join('+' + line + '\n' for line in s.splitlines())
        patch += '*** End Patch\n'
        subprocess.run(['apply_patch'], input=patch, text=True, cwd=ROOT, check=True, capture_output=True)
    print({k: len(v) for k, v in recovered.items()}, 'unresolved', len(rejected))


if __name__ == '__main__':
    main()
