"""Final PER scaling: tour vs gradient steps. Shows the ceiling."""
import glob, json, os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='src/counterfactual_rl/agents/cvrp_neural/runs'; OPT,TH=6.10,6.36
def curve(j,n):
    ep,t=[],[]
    for l in open(f'{R}/{j}/metrics.log'):
        if l.startswith('#'): continue
        p=l.split()
        if not p or p[0]=='episode': continue
        ep.append(int(p[0])); t.append(-float(p[5]))
    return np.array(ep)*24/n, np.array(t)

man=json.load(open(glob.glob(f'{R}/../experiments/2026-09/replay_ratio_*/manifest.json')[0]))
cells=sorted(man.items(), key=lambda kv:-kv[1]['n_steps_per_update'])
fig,ax=plt.subplots(figsize=(11,6.5))
cols=plt.cm.plasma(np.linspace(.05,.78,len(cells)))
pts=[]
for (j,c),col in zip(cells,cols):
    if not os.path.exists(f'{R}/{j}/metrics.log'): continue
    g,t=curve(j,c['n_steps_per_update'])
    ax.plot(g,t,lw=1.7,color=col,alpha=.85,
            label=f"nspu={c['n_steps_per_update']:<4} {c['n_episodes']//1000}k eps")
    ax.plot(g[-1],t[-1],'o',ms=9,color=col,mec='white',mew=1.4,zorder=5)
    pts.append((g[-1],t[-1],c['n_episodes']))
g,t=curve('274699',512); ax.plot(g,t,lw=1.5,ls='--',color='#BF3B30',alpha=.8,label='nspu=512  250k eps')
ax.plot(g[-1],t[-1],'s',ms=9,color='#BF3B30',mec='white',mew=1.4,zorder=5); pts.append((g[-1],t[-1],250000))

# the plateau band
band=[p for p in pts if p[0]>30000]
if band:
    lo,hi=min(p[1] for p in band), max(p[1] for p in band)
    ax.axhspan(lo-.02,hi+.02,xmin=.55,color='#BF3B30',alpha=.07)
    ax.annotate(f'PLATEAU  {lo:.2f} - {hi:.2f}\n2x the gradient steps changes nothing',
                xy=(50000,(lo+hi)/2), xytext=(14000,6.45), fontsize=10.5, color='#BF3B30',
                arrowprops=dict(arrowstyle='->',color='#BF3B30',lw=1.3))
ax.axhline(TH,ls='--',lw=1.8,color='#C06A08'); ax.axhline(OPT,lw=1.5,color='#157F6B')
ax.text(1.4e5,TH+.06,'RP-DQN greedy 6.36  (gap 4.26%)',ha='right',color='#C06A08',fontsize=10)
ax.text(1.4e5,OPT-.16,'optimal 6.10',ha='right',color='#157F6B',fontsize=10)
ax.set(xscale='log',xlim=(1e2,1.5e5),ylim=(6.0,11.5),
       xlabel='gradient steps (log)',ylabel='mean tour length  (lower is better)',
       title='PER on neural CVRP n=20 — the ceiling is not a budget problem')
ax.grid(alpha=.25,lw=.6,which='both'); ax.legend(frameon=False,fontsize=9.5,loc='upper right')
for sp in ('top','right'): ax.spines[sp].set_visible(False)
plt.tight_layout()
o='docs/figures/real/claim2/cvrp_neural/per_scaling_plateau.png'
plt.savefig(o,dpi=150,bbox_inches='tight',facecolor='white'); print('wrote',o)
for g,t,e in sorted(pts): print(f"  {g:>8,.0f} steps  {e:>7,} eps  {t:.3f}")
