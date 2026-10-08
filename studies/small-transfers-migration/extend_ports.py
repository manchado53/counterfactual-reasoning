"""One-shot source capture and mechanical old ConnectFour port; no jobs."""
import ast
from copy import deepcopy
import hashlib
import json
import subprocess
from capture import DEST, ROOT, add

archive = DEST/'studies/small-transfers-migration/sources/root'
deps = ['src/counterfactual_rl/agents/chess/array_buffer.py',
        'src/counterfactual_rl/utils/action_selection.py']
records = {}
for name in deps:
    original = ROOT/name
    content = original.read_bytes()
    add(str((archive/name).relative_to(DEST)),content.decode())
    records[name] = dict(sha256=hashlib.sha256(content).hexdigest(),
        status=subprocess.check_output(['git','status','--porcelain','--',name],cwd=ROOT,text=True))
add('studies/small-transfers-migration/extra-source-manifest.json',json.dumps(dict(
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=records),indent=2)+'\n')

base = 'experiments/learners/connect_four/legacy/'
add(base+'__init__.py','')
add(base+'shared/__init__.py','')
files = {name:'src/counterfactual_rl/agents/connect_four/'+name for name in
         ('config.py','dqn.py','consequence_dqn.py','policies.py','opponent.py','opponent_mcts.py')}
files.update({'array_buffer.py':deps[0], 'action_selection.py':deps[1],
              'metrics.py':'src/counterfactual_rl/analysis/metrics.py'})
for name in ('metrics.py','consequence_diagnostics.py'):
    files['shared/'+name] = 'src/counterfactual_rl/agents/shared/'+name
for name in ('__init__.py','timer.py','plot.py'):
    files['shared/timing/'+name] = 'src/counterfactual_rl/agents/shared/timing/'+name
for target,source in files.items():
    text = (archive/source).read_text()
    text = text.replace('from ..chess.array_buffer','from .array_buffer')
    text = text.replace('from ..shared.','from .shared.')
    text = text.replace('from counterfactual_rl.utils.action_selection','from .action_selection')
    text = text.replace('from counterfactual_rl.analysis.metrics','from .metrics')
    if target == 'shared/consequence_diagnostics.py':
        text = text.replace('from .metrics import compute_all_consequence_metrics',
                            'from ..metrics import compute_all_consequence_metrics')
    if target == 'shared/metrics.py':
        text = text.replace("self.dir = os.path.join(base, 'runs', job_id)",
                            "self.dir = os.environ.get('CCE_RUN_DIR') or os.path.join(base, 'runs', job_id)")
    add(base+target,text)

def definitions(path):
    ns = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns

ns = definitions(archive/'src/counterfactual_rl/agents/connect_four/run_experiments.py')
defaults = definitions(archive/'src/counterfactual_rl/agents/connect_four/config.py')['DEFAULT_CONNECT4_CONFIG']
for study in ns['EXPERIMENTS'].values():
    rows = []
    for overrides in study['runs']:
        cfg = deepcopy(defaults)
        cfg.update(study['fixed']); cfg.update(overrides)
        cfg.update(protocol='legacy-connect-four',collector_variant='old-chunk',
                   training_budget_kind='chunks',early_stop_win_rate=None,
                   diagnostics_n_step_slices=10,diagnostics_n_scatter_snapshots=10)
        alg = cfg['algorithm']
        arm = ('uniform' if alg=='dqn-uniform' else 'per' if alg=='dqn' else
               'cce_only' if cfg['mu']==1 else 'cce_mul' if cfg['priority_mixing']=='multiplicative' else 'cce_add')
        rows.append(dict(task=len(rows),arm=arm,seed=cfg['seed'],collector='old-chunk',config=cfg))
    add('configs/experiment/connect_four-'+study['name'].replace('_','-')+'-legacy.json',
        json.dumps(dict(schema_version=1,environment='connect_four',study=study['name'],
            provenance='Current captured source definitions, not recovered historical full headers; eval_opponent is inert in source',rows=rows),indent=2)+'\n')

dk = DEST/'studies/small-transfers-migration/sources/andon/src/counterfactual_rl/agents/doorkey'
study = definitions(dk/'run_experiments.py')['DOORKEY_HEADROOM']
defaults = definitions(dk/'config.py')['DEFAULT_CONFIG']
rows = []
for overrides in study['runs']:
    cfg = deepcopy(defaults); cfg.update(study['fixed']); cfg.update(overrides)
    cfg.update(protocol='legacy-doorkey',collector_variant='old',training_budget_kind='episodes',early_stop_win_rate=None)
    rows.append(dict(task=len(rows),arm='uniform' if cfg['algorithm']=='dqn-uniform' else 'per',
                     seed=cfg['seed'],collector='old',config=cfg))
add('configs/experiment/doorkey-headroom-legacy.json',json.dumps(dict(schema_version=1,
    environment='doorkey',study='doorkey_headroom',provenance='Exact captured source definitions plus original defaults',rows=rows),indent=2)+'\n')
