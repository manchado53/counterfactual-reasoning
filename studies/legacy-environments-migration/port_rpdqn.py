"""Freeze the dirty RP-DQN worktree sources, then make independent new-path copies."""
import hashlib
import json
import subprocess
from pathlib import Path
from port_sources import ROOT, STUDY, add


def main():
    source_root = ROOT.parent / 'cce-rpdqn'
    mapping = {}
    for name in ('__init__.py', 'config.py', 'dqn.py', 'consequence_dqn.py',
                 'dqn_vectorized.py', 'consequence_dqn_vectorized.py', 'train.py'):
        mapping[Path('experiments/learners/rpdqn') / name] = Path('src/counterfactual_rl/agents/cvrp_neural') / name
    mapping[Path('experiments/envs/rpdqn.py')] = Path('src/counterfactual_rl/envs/cvrp_neural.py')
    for name in ('__init__.py', 'buffers.py', 'consequence_buffers.py', 'cf_rollout.py',
                 'timing/__init__.py', 'timing/timer.py', 'timing/plot.py'):
        mapping[Path('experiments/learners/rpdqn/_historical') / name] = Path('src/counterfactual_rl/agents/shared') / name
    mapping[Path('experiments/learners/rpdqn/_historical/score_metrics.py')] = Path('src/counterfactual_rl/analysis/metrics.py')
    sources = {str(p): (source_root / p).read_text() for p in mapping.values()}
    if sum(len(v.encode()) for v in sources.values()) > 2 * 1024**2:
        raise ValueError('2 MiB source cap')
    manifest = dict(source_root=str(source_root), commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip(),
        dirty_status=subprocess.check_output(['git', 'status', '--porcelain', '--', 'src'], cwd=source_root, text=True),
        sources={p: dict(sha256=hashlib.sha256(s.encode()).hexdigest(), archive=str(STUDY / 'rpdqn-sources' / p)) for p, s in sources.items()},
        ports={str(k): str(v) for k, v in mapping.items()})
    add({**{STUDY / 'rpdqn-sources' / p: s for p, s in sources.items()},
         STUDY / 'rpdqn-source-manifest.json': json.dumps(manifest, indent=2) + '\n'})
    ports = {}
    for dest, src in mapping.items():
        s = sources[str(src)].replace('from ..shared.', 'from ._historical.')
        s = s.replace('counterfactual_rl.envs.cvrp_neural', 'experiments.envs.rpdqn')
        s = s.replace('counterfactual_rl.analysis.metrics', 'experiments.learners.rpdqn._historical.score_metrics')
        s = s.replace("run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)",
                      "run_dir = os.environ.get('CCE_RUN_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)")
        ports[dest] = s
    add(ports)


if __name__ == '__main__':
    main()
