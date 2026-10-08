"""Extract source config functions without importing a training runtime."""
import ast
import json
from capture import DEST, add


def extract(path, names):
    text = path.read_text()
    tree = ast.parse(text)
    return '\n\n'.join(ast.get_source_segment(text,n) for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in names) or
           (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)))


for env in ('doorkey','connect_four'):
    base = DEST/'experiments/learners'/env
    if env == 'doorkey':
        code = 'import json\nfrom pathlib import Path\nHERE = Path(__file__).resolve().parent\n'
        code += extract(base/'train.py', {'RECIPE','ARMS','config_for'})
    else:
        code = extract(base/'transfer.py', {'ARMS','RECIPE'})+'\n\n'
        code += extract(base/'transfer_run.py', {'effective_config'})
        code += '\n\nconfig_for = effective_config\n'
    add('experiments/learners/'+env+'/config.py',code+'\n')
    namespace = {'__file__': str(base/'config.py')}
    exec(code, namespace)
    rows = []
    for arm in namespace['ARMS']:
        for seed in (0,1):
            cfg = namespace['config_for'](arm,seed)
            cfg.update(protocol='fixed-jaxnav-transfer', collector_variant='fixed',
                       training_budget_kind='env_steps', early_stop_win_rate=None)
            rows.append(dict(task=len(rows),arm=arm,seed=seed,collector='fixed',config=cfg))
    add('configs/experiment/'+env+'-fixed-transfer.json',json.dumps(dict(schema_version=1,environment=env,rows=rows),indent=2)+'\n')
