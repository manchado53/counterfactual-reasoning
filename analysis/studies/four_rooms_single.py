"""Matched seed-0 control plot: observed differences, no seed-level inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis.studies.four_rooms import read_jsonl

REPLAY_KEYS={'arm','normalise_priorities','priority_mixing','mu'}


def load_seed_zero(roots):
    runs={};configs={};statuses={};hashes=set()
    for root in roots:
        for file in sorted(root.glob('*/attempt.json')):
            rec=json.loads(file.read_text());c=rec['resolved_config'];arm=c['arm']
            if rec.get('smoke') or c['seed']!=0: continue
            if arm in runs: raise ValueError(f'Duplicate seed-0 arm: {arm}')
            rows=read_jsonl(file.parent/'eval.jsonl')
            runs[arm]={r['env_steps']:r for r in rows}
            configs[arm]={k:v for k,v in c.items() if k not in REPLAY_KEYS}
            statuses[arm]=rec['status']
            hashes.update(r['eval_bank_hash'] for r in rows)
    if len(hashes)>1: raise ValueError('Different evaluation banks')
    if len({json.dumps(c,sort_keys=True) for c in configs.values()})>1:
        raise ValueError('Non-replay settings differ')
    return runs,statuses


def report(control,reference,out):
    out.mkdir(parents=True,exist_ok=True)
    runs,statuses=load_seed_zero([reference,control])
    colors=dict(uniform='#2878b5',rank_per='#e38c22',cce='#25976a',proportional_per='#9b4eb2')
    for metric,label,filename in [('success_rate','Goal success rate','success'),
                                   ('successful_length','Successful episode length','successful-length')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for arm,values in runs.items():
            x=sorted(values)
            y=[values[s][metric] if values[s][metric] is not None else np.nan for s in x]
            ax.plot(x,y,label=arm,color=colors[arm],lw=2)
        ax.set(title='2× Sparse Four Rooms — matched seed 0 only',xlabel='Training environment steps',ylabel=label)
        if runs: ax.legend()
        ax.text(0,-.18,'One training seed per arm. No SEM or probability-of-superiority claim.\n'
                'Observed points only; missing tails stay missing.',transform=ax.transAxes,fontsize=9)
        fig.savefig(out/f'{filename}.png',dpi=160,bbox_inches='tight');plt.close(fig)
    common=sorted(set.intersection(*(set(runs[a]) for a in colors))) if set(colors)<=set(runs) else []
    table=[]
    if common:
        x=np.asarray(common)
        for arm in colors:
            y=np.array([runs[arm][s]['success_rate'] for s in x])
            table.append(dict(arm=arm,seed=0,common_horizon=int(x[-1]),success_at_horizon=float(y[-1]),
                normalized_auc=float(np.trapezoid(y,x)/(x[-1]-x[0])) if len(x)>1 else None,
                peak_observed=float(y.max())))
    with (out/'comparison.csv').open('w') as f:
        if table:
            writer=csv.DictWriter(f,fieldnames=table[0]);writer.writeheader();writer.writerows(table)
    (out/'status.json').write_text(json.dumps(dict(statuses=statuses,common_horizon=common[-1] if common else None,
        inference='One seed only; do not estimate across-seed superiority from this run'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.control,a.reference,a.out)
