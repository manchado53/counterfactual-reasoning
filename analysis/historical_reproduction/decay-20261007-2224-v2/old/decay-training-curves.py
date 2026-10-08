"""Plot a read-only snapshot of production decay logs; no tail filling."""
import json, hashlib, ast, re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning')
OUT = Path(__file__).parent
manifest = ROOT / '.claude/worktrees/jaxnav-decay-carry/work/decay_20261007_v1/manifest.json'
rows = json.loads(manifest.read_text())['rows']
stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M %Z')
data = {}; sources = []; seen = set(); banks = set()

def read(path, progress=False):
    blob = path.read_bytes()
    lines = blob.splitlines(keepends=True)
    result = []
    for line in lines:
        if not line.endswith(b'\n'): continue
        if progress:
            if not line.startswith(b'PROGRESS '): continue
            line = line[len(b'PROGRESS '):]
        result.append(json.loads(line))
    if result:
        ep = np.array([r['episode'] for r in result])
        assert np.all(np.diff(ep) > 0), path
        assert all(np.isfinite(r['win_rate']) and 0 <= r['win_rate'] <= 1 for r in result), path
    sources.append(dict(path=str(path), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob), count=len(result)))
    return result

for row in rows:
    if row['mode'] != 'train': continue
    arm, seed = row['arm'], row['seed']
    if (arm, seed) not in seen:
        seen.add((arm, seed))
        hist = read(Path(row['parent_run']) / 'eval.jsonl')
        data.setdefault((arm, 'Historical: old / no decay'), []).append((seed, hist))
    array = {'h100':280874, 'v100':280899, 't4':280876}[row['gpu']]
    path = Path(f'/home/ad.msoe.edu/manchadoa/jaxnav_decay_logs/{array}_{row["task"]}.out')
    if not path.exists(): continue
    records = read(path, True)
    if not records: continue
    assert all(r['seed'] == seed and r['arm'] == arm and r['task'] == row['task'] for r in records)
    banks.update(r['eval_bank_hash'] for r in records)
    name = ('Old / decay' if row['collector'] == 'old' else
            'Fixed / decay' if row['decay'] else 'Fixed / no decay')
    group = data.setdefault((arm, name), [])
    assert seed not in [s for s, _ in group]
    group.append((seed, records))
assert len(banks) == 1, banks

# Use the audit only as a file index; reload metrics and configs from each run.
baseline_runs = {}
audit = json.loads((OUT/'jaxnav-five-arms-audit.json').read_text())
check_keys = ['map_id','map_size','fill','goal_radius','goal_rew','coll_rew',
              'sparse_reward','max_steps','fixed_eval_seed','eval_episodes',
              'gamma','alpha','batch_size','buffer_capacity','target_update_freq',
              'n_steps_per_update','double_dqn','hidden_dim','n_layers',
              'epsilon_start','epsilon_end','epsilon_decay_episodes','n_envs','collect_steps']
reference = next(r['config'] for r in rows if r['mode']=='train')
for entry in audit['runs']:
    if entry['arm'] not in ('Uniform','PER'): continue
    run = Path(entry['run'])
    config = {}
    for line in (run/'metrics.log').read_text().splitlines():
        if not line.startswith('#'): break
        match = re.match(r'^# ((?:[a-z_][a-z_0-9]*|PER_parameters)): (.*)$',line)
        if match:
            try: config[match[1]] = ast.literal_eval(match[2])
            except (ValueError,SyntaxError): config[match[1]] = match[2]
    for key in check_keys: assert config[key] == reference[key], (run,key)
    assert config['seed'] == entry['seed']
    baseline_runs.setdefault(entry['arm'],[]).append((entry['seed'],read(run/'eval.jsonl')))
for arm, runs in baseline_runs.items():
    assert len(runs)==5 and {s for s,_ in runs}==set(range(5)), arm

edges = np.arange(0, 200001, 5000)
x = (edges[:-1] + edges[1:]) / 2 / 1000
def binned(records):
    ep = np.array([r['episode'] for r in records])
    y = np.array([r['win_rate'] for r in records])
    result = np.full(len(x), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        ix = (ep >= lo) & (ep < hi)
        if hi <= ep[-1] and ix.any(): result[i] = y[ix].mean()
    return result

groups = ['Historical: old / no decay', 'Old / decay', 'Fixed / no decay', 'Fixed / decay']
colors = ['#777777', '#e58606', '#2687bf', '#9c43a0']
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=True)
summary = []
for ax, arm, title in zip(axes, ['mul','add','only'], ['CCE Multiply', 'CCE Add 50/50', 'CCE Only']):
    ax.axvspan(100, 150, color='#ffe6aa', alpha=.5)
    ax.axvline(100, color='#bb943d', lw=1, ls=':')
    ax.axvline(150, color='#bb943d', lw=1, ls=':')
    for baseline, color, style in [('Uniform','#14976c','--'),('PER','#c63445','-.')]:
        values = np.array([binned(r) for _,r in baseline_runs[baseline]])
        assert np.all(np.isfinite(values))
        ax.plot(x, values.mean(axis=0), color=color, ls=style, lw=2.3,
                label=f'{baseline}: historical old, mean of 5/5 seeds')
    for name, color in zip(groups, colors):
        runs = data.get((arm, name), [])
        if not runs:
            ax.plot([], [], color=color, label=f'{name}: waiting (0/5)')
            continue
        values = np.array([binned(r) for _, r in runs])
        for y in values: ax.plot(x, y, color=color, alpha=.22, lw=.85)
        valid = np.all(np.isfinite(values), axis=0)
        mean = np.full(len(x), np.nan)
        mean[valid] = values[:,valid].mean(axis=0)
        ax.plot(x, mean, color=color, lw=2.5, ls='--' if name.startswith('Historical') else '-',
                label=f'{name}: mean of {len(runs)}/5 seeds')
        summary.append(dict(arm=arm,group=name,seeds={str(s):r[-1]['episode'] for s,r in runs},
                            mean_through=int(edges[1:][valid][-1]) if valid.any() else 0))
    ax.set_title(title, loc='left', weight='bold')
    ax.set_ylim(0,.85); ax.set_xlim(0,200)
    ax.set_ylabel('Goal reach rate')
    ax.grid(alpha=.16)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
axes[-1].set_xlabel('Training episodes (thousands)')
fig.suptitle('Live JaxNav decay runs — sparse reward, 8×8 maps', fontsize=19, weight='bold', y=.99)
fig.text(.09,.951,'Shaded 100k–150k: shift to ranked TD sampling. Thin = each seed; thick = arithmetic mean.',fontsize=11)
fig.text(.09,.026, f'Log snapshot: {stamp} | Arrays 280874 / 280876 / 280899 | 5k-episode bins\n'
         'Mean stops at the shared end of the shown seeds; thin tails show faster seeds. No filled tails.\n'
         'Eval: 1,000 fixed maps, seed 20260923. Uniform/PER/gray are historical old-collector runs; hardware differs.\n'
         'PER baseline uses raw TD, not the ranked TD decay endpoint. Baselines show means only. No confidence bands.',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.08,1,.94))
fig.savefig(OUT/'decay-training-curves-with-baselines.png',dpi=180)
(OUT/'decay-training-curves-with-baselines-sources.json').write_text(json.dumps(dict(snapshot=stamp,banks=list(banks),summary=summary,sources=sources),indent=2))
print(json.dumps(summary,indent=2))
