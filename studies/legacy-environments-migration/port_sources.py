"""One-shot, bounded mechanical port. Refuses to overwrite any destination.

Run from repository root. Original bytes are saved before rewritten imports.
This is source preservation, not scientific validation.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path('studies/legacy-environments-migration')


def add(files):
    if not files:
        return
    patch = '*** Begin Patch\n'
    for path, content in files.items():
        if (ROOT / path).exists():
            raise FileExistsError(path)
        patch += f'*** Add File: {path}\n'
        patch += ''.join('+' + line + '\n' for line in content.splitlines())
    patch += '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, cwd=ROOT, check=True, capture_output=True)


def main():
    mappings = {}
    for env in ('smax', 'chess', 'cvrp'):
        base = Path('src/counterfactual_rl/agents') / env
        names = ['__init__.py', 'config.py', 'dqn.py', 'consequence_dqn.py', 'train.py']
        names += {'smax': ['policies.py', 'utils.py'],
                  'chess': ['policies.py', 'array_buffer.py'], 'cvrp': ['sanity.py']}[env]
        for name in names:
            mappings[Path('experiments/learners') / env / name] = base / name
        mappings[Path('experiments/envs') / (env + '.py')] = Path('src/counterfactual_rl/envs') / (env + '.py')
        for name in ('buffers.py', 'consequence_buffers.py', 'metrics.py', 'consequence_diagnostics.py',
                     'timing/__init__.py', 'timing/timer.py', 'timing/plot.py'):
            mappings[Path('experiments/learners') / env / '_historical' / name] = Path('src/counterfactual_rl/agents/shared') / name
        for name, src in [('score_metrics.py', 'analysis/metrics.py'),
                          ('action_selection.py', 'utils/action_selection.py')]:
            mappings[Path('experiments/learners') / env / '_historical' / name] = Path('src/counterfactual_rl') / src
        if env == 'smax':
            for name in ('smax_utils.py', 'smax_jax_utils.py'):
                mappings[Path('experiments/learners/smax/_historical') / name] = Path('src/counterfactual_rl/utils') / name
    sources = {str(p): (ROOT / p).read_text() for p in set(mappings.values())}
    if sum(len(s.encode()) for s in sources.values()) > 5 * 1024**2:
        raise ValueError('5 MiB source cap')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src'], cwd=ROOT, text=True)
    manifest = dict(commit=head, source_status=dirty, sources={
        p: dict(sha256=hashlib.sha256(s.encode()).hexdigest(), archive=str(STUDY / 'sources' / p))
        for p, s in sorted(sources.items())}, ports={str(k): str(v) for k, v in mappings.items()})
    add({**{STUDY / 'sources' / p: s for p, s in sources.items()},
         STUDY / 'source-manifest.json': json.dumps(manifest, indent=2) + '\n'})
    ports = {}
    for dest, src in mappings.items():
        text = sources[str(src)]
        env = dest.parts[2] if dest.parts[1] == 'learners' else dest.stem
        text = text.replace('from ..shared.', 'from ._historical.')
        text = text.replace('counterfactual_rl.envs.' + env, 'experiments.envs.' + env)
        for old, new in [('analysis.metrics', 'score_metrics'), ('utils.action_selection', 'action_selection'),
                         ('utils.smax_utils', 'smax_utils'), ('utils.smax_jax_utils', 'smax_jax_utils')]:
            text = text.replace('counterfactual_rl.' + old, f'experiments.learners.{env}._historical.{new}')
        if dest.name == 'metrics.py':
            text = text.replace("self.dir = os.path.join(base, 'runs', job_id)",
                                "self.dir = os.environ.get('CCE_RUN_DIR') or os.path.join(base, 'runs', job_id)")
        if env == 'cvrp' and dest.name == 'dqn.py':
            text = text.replace("run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)",
                                "run_dir = os.environ.get('CCE_RUN_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)")
        ports[dest] = text
    for env in ('smax', 'chess', 'cvrp'):
        ports[Path('experiments/learners') / env / '_historical/__init__.py'] = '"""Frozen historical dependencies; not the current ranked replay core."""\n'
    add(ports)


if __name__ == '__main__':
    main()
