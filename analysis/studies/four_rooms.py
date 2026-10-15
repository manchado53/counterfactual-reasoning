"""Observed-only Sparse Four Rooms report; paired-seed uncertainty."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('uniform', 'rank_per', 'cce')
COLORS = ('#2878b5', '#e38c22', '#25976a')


def read_jsonl(path):
    rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass  # A writer may be midway through last line.
    return rows


def paired_probability(differences):
    """Exact paired seed bootstrap; strict wins, not a posterior probability."""
    d=np.asarray(differences,dtype=float)
    if d.shape != (5,) or not np.isfinite(d).all():
        raise ValueError('Need five finite paired seed differences')
    indices=np.asarray(list(itertools.product(range(5),repeat=5)))
    return float(np.mean(d[indices].mean(axis=1)>0))


def common_steps(runs):
    if any((a,s) not in runs for a in ARMS for s in range(5)):
        return []
    return sorted(set.intersection(*(set(runs[a,s]) for a in ARMS for s in range(5))))


def load_runs(root):
    runs, statuses, hashes = {}, {}, set()
    for path in sorted(root.glob('*/attempt.json')):
        record=json.loads(path.read_text())
        if record.get('smoke'): continue
        c=record['resolved_config']; key=(c['arm'],c['seed'])
        if key in runs: raise ValueError(f'Duplicate attempt for {key}; select one cohort')
        if key[0] not in ARMS: continue
        runs[key]={r['env_steps']:r for r in read_jsonl(path.parent/'eval.jsonl')}
        statuses[key]=record['status']
        hashes |= {r['eval_bank_hash'] for r in runs[key].values()}
    if len(hashes)>1: raise ValueError('Evaluation bank mismatch')
    return runs,statuses


def draw_map(out):
    # Read-only source constants; loading this module imports JAX, never a device.
    from training.envs.four_rooms import LAYOUT, DOORS
    walls=np.array([[c=='#' for c in row] for row in LAYOUT])
    fig,ax=plt.subplots(figsize=(6,6))
    ax.imshow(walls,cmap='Greys',vmin=0,vmax=1)
    for r,c in DOORS:
        ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,facecolor='#f8cd62'))
        ax.text(c,r,'D',ha='center',va='center',fontsize=12)
    for r,c,label,color in [(1,1,'S','#2878b5'),(9,9,'G','#25976a')]:
        ax.scatter(c,r,s=390,color=color)
        ax.text(c,r,label,ha='center',va='center',color='white',weight='bold')
    ax.set_xticks(np.arange(-.5,13,1),minor=True);ax.set_yticks(np.arange(-.5,13,1),minor=True)
    ax.grid(which='minor',alpha=.2);ax.set_xticks([]);ax.set_yticks([])
    ax.set_title('Sparse Four Rooms\nS=start, G=goal, D=door; cap 200 steps')
    fig.savefig(out/'map.png',dpi=180,bbox_inches='tight');plt.close(fig)


def report(root,out):
    out.mkdir(parents=True,exist_ok=True)
    draw_map(out)
    runs,statuses=load_runs(root)
    for metric,title,name in [('success_rate','Goal success rate','success'),
                              ('successful_length','Successful episode length','successful-length')]:
        fig,ax=plt.subplots(figsize=(9,5))
        counts={}
        for arm,color in zip(ARMS,COLORS):
            steps=sorted(set().union(*(set(v) for (a,s),v in runs.items() if a==arm)))
            if not steps: continue
            data=np.full((5,len(steps)),np.nan)
            for s in range(5):
                for j,x in enumerate(steps):
                    value=runs.get((arm,s),{}).get(x,{}).get(metric)
                    if value is not None: data[s,j]=value
                ax.plot(steps,data[s],color=color,alpha=.16,lw=.8)
            n=np.isfinite(data).sum(axis=0)
            valid=n>0
            mean=np.full(len(steps),np.nan);median=mean.copy();sem=mean.copy()
            for j in np.flatnonzero(valid):
                values=data[:,j][np.isfinite(data[:,j])]
                mean[j]=values.mean();median[j]=np.median(values)
                if len(values)>1: sem[j]=values.std(ddof=1)/np.sqrt(len(values))
            ax.plot(steps,mean,color=color,lw=2.6,label=arm)
            ax.plot(steps,median,color=color,lw=1.6,ls='--')
            ax.fill_between(steps,mean-sem,mean+sem,color=color,alpha=.15)
            counts[arm]=dict(steps=steps,n=n.tolist())
        if not runs: ax.text(.5,.5,'No full-run observations yet',transform=ax.transAxes,ha='center')
        ax.set(title=title+' — observed points only',xlabel='Training environment steps',ylabel=title)
        if runs: ax.legend()
        ax.text(0,-.23,'Thick: mean | dashed: median | faint: seeds | band: descriptive SEM\n'
                'Missing tails stay missing. Counts saved in curve-counts JSON.',transform=ax.transAxes,fontsize=9)
        fig.savefig(out/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)
        (out/f'{name}-curve-counts.json').write_text(json.dumps(counts,indent=2))
    common=common_steps(runs)
    table=[];prob=[]
    if common:
        x=np.asarray(common)
        for arm in ARMS:
            for seed in range(5):
                y=np.array([runs[arm,seed][step]['success_rate'] for step in x])
                table.append(dict(arm=arm,seed=seed,horizon=int(x[-1]),final=float(y[-1]),
                    auc=float(np.trapezoid(y,x)/(x[-1]-x[0])) if len(x)>1 else None,
                    peak=float(y.max())))
        for other in ARMS[:2]:
            for step in common:
                d=np.array([runs['cce',s][step]['success_rate']-runs[other,s][step]['success_rate'] for s in range(5)])
                prob.append(dict(comparison='cce-'+other,env_steps=step,
                    paired_bootstrap_probability=paired_probability(d),seed_win_fraction=float((d>0).mean()),
                    mean_difference=float(d.mean())))
            for metric in ('final','auc','peak'):
                if metric=='auc' and len(common)<2: continue
                vals={a:[r[metric] for r in table if r['arm']==a] for a in ('cce',other)}
                d=np.array(vals['cce'])-np.array(vals[other])
                prob.append(dict(comparison='cce-'+other,env_steps=int(x[-1]),metric=metric,
                    paired_bootstrap_probability=paired_probability(d),seed_win_fraction=float((d>0).mean()),
                    mean_difference=float(d.mean())))
    summary=[]
    for arm in ARMS:
        rows=[r for r in table if r['arm']==arm]
        if rows:
            row=dict(arm=arm,n=5,horizon=common[-1])
            for metric in ('final','auc','peak'):
                values=[r[metric] for r in rows if r[metric] is not None]
                row[metric+'_mean']=float(np.mean(values)) if values else None
                row[metric+'_median']=float(np.median(values)) if values else None
            summary.append(row)
    for filename,rows in [('per-seed.csv',table),('comparison.csv',summary)]:
        with (out/filename).open('w') as f:
            if rows:
                writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (out/'probabilities.json').write_text(json.dumps(prob,indent=2))
    (out/'status.json').write_text(json.dumps(dict(statuses={f'{a}/{s}':v for (a,s),v in statuses.items()},
        common_horizon=common[-1] if common else None,
        completed=len(statuses)==15 and all(s=='completed' for s in statuses.values()),
        inference='exact paired seed bootstrap, not posterior probability'),indent=2))
    fig,ax=plt.subplots(figsize=(9,4))
    for other,color in zip(ARMS[:2],COLORS[:2]):
        rows=[r for r in prob if r['comparison']=='cce-'+other and 'metric' not in r]
        ax.plot([r['env_steps'] for r in rows],[r['paired_bootstrap_probability'] for r in rows],label='CCE > '+other,color=color)
    ax.set(ylim=(-.02,1.02),xlabel='Training steps (all five matched seeds)',ylabel='Paired bootstrap fraction > 0',
           title='Descriptive bootstrap support; not a posterior probability')
    ax.legend();fig.savefig(out/'probability.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.runs,a.out)
