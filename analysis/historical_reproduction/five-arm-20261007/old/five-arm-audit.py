"""Read-only run audit and derived figure; no training or repository changes."""
from pathlib import Path
import importlib.util, ast, re, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/ad.msoe.edu/manchadoa/UR-RL/counterfactual-reasoning')
OUT=Path(__file__).resolve().parent
p=ROOT/'.claude/worktrees/research+cce-robotics-transfer/analysis/claim2/fixed_eval_log_curves.py'
s=importlib.util.spec_from_file_location('original',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
records=[m._load_task(p) for a in ['277733','278035'] for p in sorted(m.DEFAULT_LOG_DIR.glob(a+'_*.out'))]
m._check_configs(records)
configs=[];seen=set();data={a:[] for a in m.ARM_ORDER}
for arm,seed,launch,rows,path in records:
    assert (arm,seed) not in seen;seen.add((arm,seed))
    assert rows[-1]['episode']>=250000
    assert np.all(np.diff([r['episode'] for r in rows])>0)
    cfg={}
    for line in (path/'metrics.log').read_text().splitlines():
        if not line.startswith('#'):break
        z=re.match(r'^# ((?:[a-z_][a-z_0-9]*|PER_parameters)): (.*)$',line)
        if z:
            try:cfg[z[1]]=ast.literal_eval(z[2])
            except (ValueError,SyntaxError):cfg[z[1]]=z[2]
    assert cfg['seed']==seed
    for k,v in launch.items():assert cfg[k]==v,(path,k)
    configs.append(dict(arm=arm,seed=seed,run=str(path),config=cfg,eval_sha256=hashlib.sha256((path/'eval.jsonl').read_bytes()).hexdigest()))
    vals=[]
    for lo in range(0,250000,5000):
        v=[r['win_rate'] for r in rows if lo<r['episode']<=lo+5000]
        assert v; vals.append(np.mean(v))
    data[arm].append(vals)
assert all({s for a,s in seen if a==arm}==set(range(5)) for arm in m.ARM_ORDER)
keys=set().union(*(r['config'] for r in configs))
shared={k:configs[0]['config'][k] for k in keys if all(k in r['config'] and r['config'][k]==configs[0]['config'].get(k) for r in configs)}
differences={k:{a:[r['config'].get(k,'NOT LOGGED') for r in configs if r['arm']==a] for a in m.ARM_ORDER} for k in keys if k not in shared}
allowed={'seed','algorithm','priority_mixing','mu','mu_c','mu_delta','normalise_priorities','score_on_insert','consequence_metric'}
assert all(k in allowed or k.startswith('cf_') for k in differences),differences.keys()
cce=[r for r in configs if r['arm'].startswith('CCE')]
for k in set().union(*(r['config'] for r in cce))-{'seed','priority_mixing','mu','mu_c','mu_delta'}:
    assert all(r['config'].get(k)==cce[0]['config'].get(k) for r in cce),k
fig,ax=plt.subplots(figsize=(12.8,7.6),dpi=170)
x=np.arange(2500,250000,5000)/1000
summary={}
for arm in m.ARM_ORDER:
    a=np.array(data[arm]);mean=a.mean(0);median=np.median(a,axis=0);sem=a.std(0,ddof=1)/np.sqrt(5)
    ax.fill_between(x,mean-sem,mean+sem,color=m.COLORS[arm],alpha=.08,lw=0)
    ax.plot(x,mean,color=m.COLORS[arm],lw=2.6,label=arm+' (5 seeds)')
    ax.plot(x,median,color=m.COLORS[arm],lw=1.2,ls='--',alpha=.8)
    summary[arm]={'mean':mean.tolist(),'median':median.tolist(),'sem':sem.tolist()}
    print(arm,'peak 5k-bin mean',float(mean.max()),'peak center',int(x[mean.argmax()]*1000),'last 5k mean',float(mean[-1]))
ax.axvline(62.5,color='#777',ls=':',lw=1)
ax.text(64,.65,'Epsilon floor',fontsize=10,color='#555')
ax.set(xlim=(0,250),ylim=(0,.8),xlabel='Logged training episodes (thousands)',ylabel='Goal reach rate')
ax.grid(alpha=.18);ax.legend(loc='upper left',ncol=2,frameon=False,fontsize=10)
fig.suptitle('JaxNav: all five replay arms',x=.085,ha='left',fontsize=18,fontweight='bold')
ax.set_title('8x8 | fill 0.3 | radius 0.8 | sparse: goal +1, crash 0 | fixed evaluation seed 20260923',loc='left',fontsize=11,pad=12)
fig.subplots_adjust(left=.085,right=.98,bottom=.26,top=.86)
fig.text(.085,.13,'Solid = mean; dashed = median; light band = descriptive ±SEM (not a 95% confidence interval).\nEach point averages evaluations within a 5,000-episode bin per seed, then across all five seeds.',fontsize=10)
fig.text(.085,.045,'Arrays 277733 + 278035 | 25 unique runs | shared logged settings checked | no filled tails\nReplay rules differ by design, including raw-TD PER vs ranked CCE signals. Historical code/package identity not certified.\nOriginal episode axis retained; these are historical runs, not a new fixed-collector rerun. Read on October 7, 2026.',fontsize=9,color='#444')
fig.savefig(OUT/'jaxnav-five-arms.png',facecolor='white')
(OUT/'jaxnav-five-arms-audit.json').write_text(json.dumps({'shared_logged_config':shared,'differing_config_fields':differences,'runs':configs,'bin_centers':(x*1000).tolist(),'summaries':summary,'limitations':['Historical source/package identity not certified','Original logged episode axis retained','Rank-normalized CCE priorities are not the raw-TD PER baseline']},indent=2))
print('SHARED',json.dumps(shared,sort_keys=True))
