"""Capture scoped source trees and make mechanical new-path copies via apply_patch.

One-shot capture helper. Refuses to overwrite existing files.
"""
from pathlib import Path
import hashlib
import json
import subprocess

DEST = Path(__file__).resolve().parents[2]
ROOT = DEST.parents[2]
SOURCES = {'root': ROOT, 'doorkey-fixed': DEST.parent / 'doorkey-jaxnav-transfer',
           'connect-four-fixed': DEST.parent / 'connect-four-jaxnav-transfer',
           'andon': DEST.parent / 'andon-vending-bench-cce'}


def add(path, text):
    target = DEST / path
    if target.exists():
        raise FileExistsError(target)
    patch = '*** Begin Patch\n*** Add File: ' + str(target) + '\n'
    patch += ''.join('+' + line + '\n' for line in text.splitlines()) + '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, check=True, stdout=subprocess.DEVNULL)


def main():
    manifest = {}
    for label, root in SOURCES.items():
        files = set()
        for pattern in ('src/counterfactual_rl/agents/doorkey/*',
                        'src/counterfactual_rl/agents/connect_four/*',
                        'src/counterfactual_rl/envs/doorkey.py',
                        'src/counterfactual_rl/envs/connect_four_transfer.py',
                        'src/counterfactual_rl/agents/shared/*.py',
                        'src/counterfactual_rl/agents/shared/timing/*.py',
                        'src/counterfactual_rl/analysis/metrics.py',
                        'analysis/claim2/*doorkey*', 'analysis/claim2/*connect_four*',
                        'tests/test*doorkey*.py', 'tests/test*connect_four*.py',
                        'docs/research/*doorkey*', 'docs/research/*connect_four*',
                        'requirements.txt', 'pyproject.toml'):
            files.update(p for p in root.glob(pattern) if p.is_file() and p.suffix in ('.py','.json','.md','.txt','.toml','.sh'))
        record = {'path': str(root), 'head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                  'status': subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True), 'files': {}}
        for file in sorted(files):
            rel = str(file.relative_to(root))
            content = file.read_bytes()
            record['files'][rel] = hashlib.sha256(content).hexdigest()
            add('studies/small-transfers-migration/sources/' + label + '/' + rel, content.decode())
        manifest[label] = record
    add('studies/small-transfers-migration/source-manifest.json',json.dumps(manifest,indent=2)+'\n')
    # All sources, including dirty edits, are recoverable before port edits begin.
    for env, label in [('doorkey','doorkey-fixed'), ('connect_four','connect-four-fixed')]:
        src = SOURCES[label]
        base = 'experiments/learners/' + env + '/'
        add(base+'__init__.py','')
        add(base+'_source_shared/__init__.py','')
        for name in ('buffers.py','consequence_buffers.py','draw_log.py'):
            add(base+'_source_shared/'+name,(src/'src/counterfactual_rl/agents/shared'/name).read_text())
        envfile = 'doorkey.py' if env == 'doorkey' else 'connect_four_transfer.py'
        add('experiments/envs/'+envfile,(src/'src/counterfactual_rl/envs'/envfile).read_text())
        for name in (('train.py','adapter.py','source_recipe.json') if env == 'doorkey' else ('transfer.py','transfer_run.py')):
            text = (src/'src/counterfactual_rl/agents'/env/name).read_text()
            text = text.replace('from ..shared.', 'from ._source_shared.')
            text = text.replace('from counterfactual_rl.agents.shared.', 'from ._source_shared.')
            text = text.replace('from counterfactual_rl.envs', 'from experiments.envs')
            add(base+name,text)
    # Older DoorKey/lava remains a separate protocol, with its own replay code.
    src = SOURCES['andon']
    base = 'experiments/learners/doorkey/legacy/'
    add(base+'__init__.py','')
    add(base+'shared/__init__.py','')
    for name in ('buffers.py','consequence_buffers.py'):
        add(base+'shared/'+name,(src/'src/counterfactual_rl/agents/shared'/name).read_text())
    for name in ('__init__.py','timer.py','plot.py'):
        add(base+'shared/timing/'+name,(src/'src/counterfactual_rl/agents/shared/timing'/name).read_text())
    add(base+'metrics.py',(src/'src/counterfactual_rl/analysis/metrics.py').read_text())
    add('experiments/envs/doorkey_legacy.py',(src/'src/counterfactual_rl/envs/doorkey.py').read_text())
    for name in ('config.py','dqn.py','consequence_dqn.py','dqn_vectorized.py','consequence_dqn_vectorized.py'):
        text = (src/'src/counterfactual_rl/agents/doorkey'/name).read_text()
        text = text.replace('from ..shared.', 'from .shared.')
        text = text.replace('from counterfactual_rl.envs.doorkey', 'from experiments.envs.doorkey_legacy')
        text = text.replace('from counterfactual_rl.analysis.metrics', 'from .metrics')
        add(base+name,text)
    print('Captured',sum(len(r['files']) for r in manifest.values()),'files and ported source modules')


if __name__ == '__main__':
    main()
