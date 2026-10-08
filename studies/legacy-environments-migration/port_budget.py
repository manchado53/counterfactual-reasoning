"""Freeze two distinct budget-routing protocols before moving their imports."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from port_sources import ROOT, STUDY, add


def main():
    source_root = ROOT.parent / 'cce-cvrp-logistics'
    for variant, ref in [('budget', '14cb90f'), ('option_a', 'c2bdb00')]:
        commit = subprocess.check_output(['git', 'rev-parse', ref], cwd=source_root, text=True).strip()
        mappings = {}
        base = Path('experiments/learners/cvrp_budget') / variant
        for name in ('__init__.py', 'config.py', 'dqn.py', 'consequence_dqn.py', 'train.py'):
            mappings[base / name] = Path('src/counterfactual_rl/agents/cvrp') / name
        for name in ('cvrp.py', 'routing_budget.py'):
            mappings[Path('experiments/envs/cvrp_budget') / variant / name] = Path('src/counterfactual_rl/envs') / name
        for name in ('__init__.py', 'buffers.py', 'consequence_buffers.py',
                     'timing/__init__.py', 'timing/timer.py', 'timing/plot.py'):
            mappings[base / '_historical' / name] = Path('src/counterfactual_rl/agents/shared') / name
        for name in ('oracle.py', 'budget_oracle.py'):
            mappings[base / '_historical' / name] = Path('src/counterfactual_rl/analysis/claim1/cvrp') / name
        mappings[base / '_historical/score_metrics.py'] = Path('src/counterfactual_rl/analysis/metrics.py')
        sources = {str(p): subprocess.check_output(['git', 'show', f'{commit}:{p}'], cwd=source_root)
                   for p in mappings.values()}
        test_path = 'tests/test_routing_budget.py'
        sources[test_path] = subprocess.check_output(['git', 'show', f'{commit}:{test_path}'], cwd=source_root)
        if sum(map(len, sources.values())) > 2 * 1024**2:
            raise ValueError('2 MiB per-variant source cap')
        archived = {}
        rows = {}
        for p, raw in sources.items():
            dest = STUDY / 'budget-sources' / variant / p
            archived[dest] = raw.decode()
            exact = dest.with_suffix(dest.suffix + '.bytes.json')
            archived[exact] = json.dumps({'base64': base64.b64encode(raw).decode()}) + '\n'
            rows[p] = dict(sha256=hashlib.sha256(raw).hexdigest(), archive=str(dest), exact_bytes=str(exact))
        manifest = dict(variant=variant, commit=commit, source_worktree=str(source_root),
            dirty_status=subprocess.check_output(['git', 'status', '--porcelain', '--', 'src'], cwd=source_root, text=True),
            capture='git objects at pinned commit; source worktree clean in scoped src paths',
            sources=rows, ports={str(k): str(v) for k, v in mappings.items()})
        archived[STUDY / f'budget-{variant}-source-manifest.json'] = json.dumps(manifest, indent=2) + '\n'
        add(archived)
        ports = {}
        env_path = f'experiments.envs.cvrp_budget.{variant}'
        learner_path = f'experiments.learners.cvrp_budget.{variant}'
        for dest, src in mappings.items():
            s = sources[str(src)].decode().replace('from ..shared.', 'from ._historical.')
            s = s.replace('counterfactual_rl.envs.cvrp', env_path + '.cvrp')
            s = s.replace('counterfactual_rl.envs.routing_budget', env_path + '.routing_budget')
            s = s.replace('from counterfactual_rl.envs import routing_budget', f'from {env_path} import routing_budget')
            s = s.replace('counterfactual_rl.analysis.claim1.cvrp.', learner_path + '._historical.')
            s = s.replace('counterfactual_rl.analysis.metrics', learner_path + '._historical.score_metrics')
            s = s.replace("run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)",
                          "run_dir = os.environ.get('CCE_RUN_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)")
            ports[dest] = s
        ports[Path('experiments/envs/cvrp_budget') / variant / '__init__.py'] = '"""Pinned budget environment variant."""\n'
        add(ports)
    add({Path('experiments/learners/cvrp_budget/__init__.py'): '"""Distinct historical budget-routing protocols."""\n',
         Path('experiments/envs/cvrp_budget/__init__.py'): '"""Budget routing, separate from distance-reward CVRP."""\n'})


if __name__ == '__main__':
    main()
