"""Resolve original launch dictionaries without importing scheduler code."""
import ast
from copy import deepcopy
import json
from capture import DEST, add

base = DEST/'studies/small-transfers-migration/sources/andon/src/counterfactual_rl/agents/doorkey'
ns = {}
exec((base/'config.py').read_text(), ns)
tree = ast.parse((base/'run_experiments.py').read_text())
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and
            (t.id.startswith('DOORKEY_') or t.id in ('_LAVA_BASE','_HEADROOM_BASE')) for t in node.targets):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(base/'run_experiments.py'),'exec'),ns)
for name in ('DOORKEY_CLAIM2','DOORKEY_LAVA_CLAIM2','DOORKEY_CLAIM1','DOORKEY_LAVA_CLAIM1'):
    study = ns[name]
    rows = []
    for run in study['runs']:
        cfg = deepcopy(ns['DEFAULT_CONFIG'])
        cfg.update(study['fixed'])
        cfg.update(run)
        cfg.update(protocol='legacy-doorkey',collector_variant='old',training_budget_kind='episodes')
        cfg.setdefault('early_stop_win_rate',None)
        algorithm = cfg['algorithm']
        arm = ('uniform' if algorithm == 'dqn-uniform' else 'per' if algorithm == 'dqn'
               else 'cce_only' if cfg['mu'] == 1 else 'cce_mul' if cfg['priority_mixing'] == 'multiplicative' else 'cce_add')
        rows.append(dict(task=len(rows),arm=arm,seed=cfg['seed'],collector='old',config=cfg))
    add('configs/experiment/'+study['name'].replace('_','-')+'-legacy.json',
        json.dumps(dict(schema_version=1,environment='doorkey',study=study['name'],
            provenance='Exact source launch dictionary plus original defaults; no completed-run claim',rows=rows),indent=2)+'\n')
