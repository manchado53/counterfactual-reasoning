"""Plot real DoorKey logs; never fill unfinished tails or mix recipes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('uniform','per','cce_mul','cce_add','cce_only')
COMMON = ('environment','layout_name','slip_prob','random_start','max_episode_steps',
          'eval_max_steps','observation','timeout_bootstrap','explore_hold_steps',
          'gamma','alpha','hidden_dim','n_layers','batch_size','buffer_capacity',
          'target_update_freq','n_steps_per_update','double_dqn','n_envs','collect_steps',
          'epsilon_start','epsilon_end','epsilon_decay_episodes','fixed_eval_seed',
          'PER_parameters','blocks','eval_every_blocks')
CCE_EXPECTED = dict(cf_horizon=60, cf_n_rollouts=1, cf_gamma=.99,
    cf_rollout_temperature=0, cf_action_prob_temperature=.5,
    consequence_metric='wasserstein', consequence_aggregation='weighted_mean',
    cf_bootstrap=True, normalise_priorities='rank', score_on_insert=True,
    cf_score_chunk=1280, refresh_enabled=False, drift_probe_enabled=False)


def validate_recipe(p):
    arm, cfg = p['arm'], p['config']
    if arm not in ARMS:
        raise ValueError('unknown replay arm')
    if not isinstance(p.get('eval_bank_hash'), str) or not p['eval_bank_hash']:
        raise ValueError('missing eval_bank_hash')
    if not isinstance(p.get('versions'), dict) or not p['versions']:
        raise ValueError('missing package versions')
    if cfg.get('cce_enabled') is not arm.startswith('cce'):
        raise ValueError('CCE enabled flag differs from arm')
    if arm.startswith('cce'):
        expected = dict(CCE_EXPECTED, mu=1. if arm == 'cce_only' else .5,
            mu_c=1., mu_delta=1.,
            priority_mixing='multiplicative' if arm == 'cce_mul' else 'additive')
        for name, value in expected.items():
            if name not in cfg or cfg[name] != value:
                raise ValueError(f'{arm}: invalid or missing CCE setting {name}')


def load_runs(paths):
    runs, seen, contract = [], set(), None
    for path in paths:
        p = json.loads((path/'provenance.json').read_text())
        if p['environment'] != 'doorkey':
            raise ValueError('not a DoorKey run')
        validate_recipe(p)
        identity = (p['arm'],p['seed'])
        if identity in seen:
            raise ValueError('duplicate arm/seed: '+str(identity))
        seen.add(identity)
        current = (p['mode'],p['source_hashes'],p['eval_bank_hash'],p['versions'],
                   {k:p['config'][k] for k in COMMON})
        if contract is not None and current != contract:
            raise ValueError('mixed source, mode, eval bank, package versions or common config')
        contract = current
        rows = []
        if (path/'eval.jsonl').exists():
            for line in (path/'eval.jsonl').read_text().splitlines(keepends=True):
                if not line.endswith('\n'):
                    continue  # writer may be halfway through the final line
                rows.append(json.loads(line))
        steps = [r['env_steps'] for r in rows]
        if steps != sorted(set(steps)):
            raise ValueError('duplicate or nonmonotone eval steps')
        if any(r['arm'] != p['arm'] or r['seed'] != p['seed'] for r in rows):
            raise ValueError('eval identity differs from provenance')
        if any(r.get('eval_bank_hash') != p['eval_bank_hash'] for r in rows):
            raise ValueError('eval bank differs from provenance')
        if any(r['unique_tasks'] != 1 or r['eval_episodes'] != 1000 for r in rows):
            raise ValueError('evaluation task changed')
        runs.append((p,rows,path))
    return runs


def plot(paths, output):
    runs = load_runs(paths)
    output.mkdir(parents=True,exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for arm,color in zip(ARMS,plt.cm.tab10.colors):
        group = [(p,rows) for p,rows,_ in runs if p['arm']==arm]
        for p,rows in group:
            for ax,metric in zip(axes,('win_rate','avg_length')):
                ax.plot([r['env_steps'] for r in rows],[r[metric] for r in rows],
                        color=color,alpha=.75,lw=1.,marker='o',markersize=4,label=f'{arm} seed {p["seed"]}')
        if {p['seed'] for p,_ in group} == {0,1}:
            maps = [{r['env_steps']:r for r in rows} for _,rows in group]
            common = sorted(set(maps[0]) & set(maps[1]))
            if common:
                for ax,metric in zip(axes,('win_rate','avg_length')):
                    values = np.array([[m[s][metric] for s in common] for m in maps])
                    ax.plot(common,values.mean(axis=0),color=color,lw=2,label=arm+' mean')
                    ax.plot(common,np.median(values,axis=0),color=color,lw=1.5,ls='--',label=arm+' median')
    for ax,label in zip(axes,('Success within 50 actions','Episode length')):
        ax.set(xlabel='Real training transitions',ylabel=label)
        ax.grid(alpha=.2)
    axes[0].set_ylim(-.03,1.03)
    axes[1].legend(fontsize=6,ncol=2)
    phase = 'Smoke checks (not full training)' if runs[0][0]['mode'] == 'smoke' else 'Full training (live)'
    fig.suptitle(phase + '\nCustom DoorKey: 1000 repeats, one fixed task; no repeat-based confidence bands')
    fig.savefig(output/'doorkey_curves.png',dpi=160)
    plt.close(fig)
    endpoints = []
    for p,rows,path in runs:
        expected = [1850*8192,1900*8192,1950*8192,2000*8192]
        final = {r['env_steps']:r for r in rows}
        value = float(np.mean([final[s]['win_rate'] for s in expected])) if (
            p['mode']=='train' and all(s in final for s in expected)) else None
        endpoints.append(dict(arm=p['arm'],seed=p['seed'],endpoint=value,run=str(path),
                              status='complete endpoint' if value is not None else 'pending'))
    (output/'endpoints.json').write_text(json.dumps(endpoints,indent=2)+'\n')
    return output/'doorkey_curves.png'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    print(plot(args.runs,args.output))


if __name__ == '__main__':
    main()
