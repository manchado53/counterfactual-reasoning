"""The same curve, split by replay METHOD, at both budgets."""
import os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

R='src/counterfactual_rl/agents/cvrp_neural/runs'
OPT, THEIRS, NOISE = 6.10, 6.36, 0.05

def read(j):
    m=f'{R}/{j}/metrics.log'
    if not os.path.exists(m): return None
    cfg,ep,t={},[],[]
    for l in open(m):
        if l.startswith('#'):
            k=l[1:].split(':',1)
            if len(k)==2: cfg[k[0].strip()]=k[1].strip()
            continue
        p=l.split()
        if not p or p[0]=='episode': continue
        ep.append(int(p[0])); t.append(-float(p[5]))
    if len(ep)<3: return None
    n=int(cfg.get('n_steps_per_update',512))
    return dict(gs=np.array(ep)*24/n, tour=np.array(t))

C = dict(uni='#8A94A0', per='#1D6FB8', h1='#9B7FB8', mul10='#B0308F',
         mul25='#157F6B', add10='#C06A08', add25='#8E6E1F')
BUDGETS = [
  ('A · nspu=512, 250k episodes  —  7 methods',
   [('274713','uniform',C['uni']),('274714','PER',C['per']),
    ('274719','CCE-mul h=1',C['h1']),('274716','CCE-mul h=10',C['mul10']),
    ('274717','CCE-mul h=25',C['mul25']),('274715','CCE-add h=10',C['add10']),
    ('274718','CCE-add h=25',C['add25'])]),
  ('B · nspu=32, 50k episodes  —  4 methods',
   [('274720','uniform',C['uni']),('274721','PER',C['per']),
    ('274723','CCE-mul h=10',C['mul10']),('274722','CCE-add h=10',C['add10'])]),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10.5),
                         gridspec_kw=dict(width_ratios=[1.55, 1], hspace=.30, wspace=.16))
for row,(title, arms) in enumerate(BUDGETS):
    ax, axz = axes[row]
    common = min(read(j)['gs'][-1] for j,_,_ in arms if read(j))
    finals = []
    for j,l,c in arms:
        d=read(j)
        if d is None: continue
        ax.plot(d['gs'], d['tour'], lw=1.6, color=c, label=l, alpha=.9)
        i=int(np.searchsorted(d['gs'], common))
        v=float(np.mean(d['tour'][max(0,i-5):i+1]))
        finals.append((l,v,c))
        ax.plot(d['gs'][-1], d['tour'][-1], 'o', ms=7, color=c, mec='white', mew=1.2, zorder=4)
    ax.axhline(THEIRS, ls='--', lw=1.4, color='#C06A08', alpha=.7)
    ax.set(xscale='log', ylim=(6.2, 11.4), xlabel='gradient steps (log)',
           ylabel='mean tour length', title=title)
    ax.grid(alpha=.25, lw=.6, which='both')
    ax.legend(frameon=False, fontsize=8.5, loc='lower left', ncol=2,
              handlelength=1.4, columnspacing=1.1)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)

    # zoom panel: the actual comparison, at matched steps
    finals.sort(key=lambda r: r[1])
    best = finals[0][1]
    y = np.arange(len(finals))[::-1]
    axz.axvspan(best, best+NOISE, color='#BF3B30', alpha=.10)
    for yy,(l,v,c) in zip(y, finals):
        axz.plot([best, v], [yy, yy], color=c, lw=1.2, alpha=.4)
        axz.plot(v, yy, 'o', ms=12, color=c, mec='white', mew=1.5)
        axz.text(v+.0022, yy, f' {v:.3f}', va='center', fontsize=10, fontfamily='monospace')
    axz.set_yticks(y); axz.set_yticklabels([r[0] for r in finals], fontsize=10)
    axz.set(xlim=(best-.006, best+.105),
            xlabel=f'mean tour @ {common:,.0f} matched steps',
            title='the comparison, zoomed')
    axz.text(best+.052, len(finals)-1.25,
             f'red band = {NOISE} noise floor\n(same config twice: 0.027)',
             fontsize=9, color='#BF3B30')
    axz.grid(alpha=.25, lw=.6, axis='x')
    for sp in ('top','right','left'): axz.spines[sp].set_visible(False)
    axz.tick_params(axis='y', length=0)

fig.suptitle('CVRP n=20 — every replay method, both budgets, single seed',
             fontsize=16, fontweight='600', y=.98, x=.075, ha='left')
plt.tight_layout(rect=[0,0,1,.965])
o='docs/figures/real/claim2/cvrp_neural/methods_curve.png'
plt.savefig(o, dpi=145, bbox_inches='tight', facecolor='white'); print('wrote', o)
