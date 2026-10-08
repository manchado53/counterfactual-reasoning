"""Current state of the CVRP/RP-DQN work: every arm, the ladder, the null, the gap."""
import glob, json, os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = 'src/counterfactual_rl/agents/cvrp_neural/runs'
E = f'{R}/../experiments/2026-09'
OPT, THEIRS = 6.10, 6.36
NOISE = 0.05

def load(j):
    m = f'{R}/{j}/metrics.log'
    if not os.path.exists(m): return None
    cfg, ep, t = {}, [], []
    for l in open(m):
        if l.startswith('#'):
            k = l[1:].split(':', 1)
            if len(k) == 2: cfg[k[0].strip()] = k[1].strip()
            continue
        p = l.split()
        if not p or p[0] == 'episode': continue
        ep.append(int(p[0])); t.append(-float(p[5]))
    if len(ep) < 3: return None
    n = int(cfg.get('n_steps_per_update', 512))
    return dict(cfg=cfg, gs=np.array(ep)*24/n, tour=np.array(t),
                target=int(cfg.get('n_episodes', 0)), ep=np.array(ep),
                done=os.popen(f'squeue -h -j {j}').read().strip() == '')

def at(d, steps, w=5):
    i = int(np.searchsorted(d['gs'], steps))
    return float(np.mean(d['tour'][max(0, i-w):i+1]))

C = dict(uni='#8A94A0', per='#1D6FB8', mul='#B0308F', add='#C06A08', h25='#157F6B', h1='#5A6875')
fig = plt.figure(figsize=(17, 10.5))
gs_ = fig.add_gridspec(2, 2, hspace=.32, wspace=.22)

# ---------- A: every arm at the 250k budget ----------
ax = fig.add_subplot(gs_[0, 0])
ARMS = [('274713','uniform',C['uni']),('274714','PER',C['per']),
        ('274716','CCE-mul h=10',C['mul']),('274715','CCE-add h=10',C['add']),
        ('274719','CCE-mul h=1',C['h1']),('274717','CCE-mul h=25',C['h25']),
        ('274718','CCE-add h=25','#8E6E1F')]
common = 1e18
for j,l,c in ARMS:
    d = load(j)
    if d is None: continue
    common = min(common, d['gs'][-1])
for j,l,c in ARMS:
    d = load(j)
    if d is None: continue
    ls = '-' if d['done'] else '--'
    ax.plot(d['gs'], d['tour'], lw=1.7, color=c, ls=ls, label=l + ('' if d['done'] else ' (running)'))
    ax.plot(d['gs'][-1], d['tour'][-1], 'o' if d['done'] else '>', ms=7, color=c, mec='white', mew=1.2)
ax.axvline(common, color='#BF3B30', ls=':', lw=1.4)
ax.text(common*.97, 10.9, 'matched\ncompare pt', ha='right', fontsize=8.5, color='#BF3B30')
ax.axhline(THEIRS, ls='--', lw=1.5, color='#C06A08')
ax.set(xscale='log', ylim=(6.0, 11.4), xlabel='gradient steps (log)',
       ylabel='mean tour length', title='A · every arm, nspu=512 / 250k episodes')
ax.legend(frameon=False, fontsize=8.5, loc='upper right', ncol=1)
ax.grid(alpha=.25, lw=.6, which='both')

# ---------- B: the ladder -- score quality vs outcome ----------
ax = fig.add_subplot(gs_[0, 1])
RHO = {'274719':0.626, '274716':0.773, '274717':1.000}      # multiplicative
RHO_A = {'274715':0.773, '274718':1.000}                     # additive
for grp, col, lab, mk in [(RHO, C['mul'], 'multiplicative', 'o'), (RHO_A, C['add'], 'additive', 's')]:
    xs, ys = [], []
    for j, r in sorted(grp.items(), key=lambda kv: kv[1]):
        d = load(j)
        if d is None: continue
        xs.append(r); ys.append(at(d, common))
    ax.plot(xs, ys, mk+'-', color=col, lw=2, ms=9, mec='white', mew=1.4, label=lab)
u = load('274713')
ax.axhline(at(u, common), color=C['uni'], lw=1.6)
ax.text(1.005, at(u, common), ' uniform', va='center', fontsize=9.5, color=C['uni'])
ax.axhspan(at(u, common)-NOISE, at(u, common)+NOISE, color=C['uni'], alpha=.13)
ax.set(xlabel='Gate-2 $\\rho$ of the score vs the exact consequence',
       ylabel=f'mean tour @ {common:,.0f} steps',
       title='B · does a better score train better?')
ax.legend(frameon=False, fontsize=9.5, loc='upper right', title='mixing', title_fontsize=8.5)
ax.grid(alpha=.25, lw=.6)
ax.text(.63, ax.get_ylim()[0]+.012, 'shaded = noise floor (±0.05)', fontsize=8.5, color=C['uni'])

# ---------- C: the null at the good budget ----------
ax = fig.add_subplot(gs_[1, 0])
NULL = [('uniform',6.705,C['uni']),('CCE-add h10',6.718,C['add']),
        ('CCE-mul h10',6.719,C['mul']),('PER',6.726,C['per']),('PER (indep)',6.680,C['per'])]
ys = np.arange(len(NULL))[::-1]
lo = min(v for _,v,_ in NULL)
ax.axvspan(lo, lo+NOISE, color='#BF3B30', alpha=.10)
for y,(l,v,c) in zip(ys, NULL):
    ax.plot([lo, v], [y, y], color=c, lw=1.2, alpha=.4)
    ax.plot(v, y, 'o', ms=12, color=c, mec='white', mew=1.5)
    ax.text(v+.004, y, f' {v:.3f}', va='center', fontsize=10, fontfamily='monospace')
ax.set_yticks(ys); ax.set_yticklabels([l for l,_,_ in NULL], fontsize=10)
ax.set(xlim=(lo-.008, lo+.075), xlabel='mean tour length @ 37,500 matched steps',
       title='C · Claim 2 at nspu=32 — everything inside the noise band')
ax.text(lo+.052, len(NULL)-1.35, 'red band = 0.05 noise floor\n(two identical configs differ by 0.027)',
        fontsize=9, color='#BF3B30')
ax.grid(alpha=.25, lw=.6, axis='x')

# ---------- D: distance to their baseline ----------
ax = fig.add_subplot(gs_[1, 1])
PTS = [(11722,250071,7.319),(37515,50021,6.680),(75045,25015,6.735),
       (88302,58868,6.613),(89098,118798,6.561)]
xs = [p[0] for p in PTS]; ys = [p[2] for p in PTS]
sc = ax.scatter(xs, ys, s=[40+p[1]/900 for p in PTS], c=[p[1] for p in PTS],
                cmap='viridis', zorder=4, edgecolor='white', linewidth=1.4)
ax.plot(xs, ys, color='#5A6875', lw=1, alpha=.5, zorder=2)
ax.scatter([1e6],[THEIRS], marker='*', s=340, color='#C06A08', zorder=5, edgecolor='white', linewidth=1.2)
ax.annotate('RP-DQN published\n6.36 @ ~1M steps', xy=(1e6,THEIRS), xytext=(1.7e5,6.30),
            fontsize=9.5, color='#C06A08',
            arrowprops=dict(arrowstyle='->', color='#C06A08', lw=1.2))
ax.axhline(OPT, color='#157F6B', lw=1.4)
ax.text(1.1e2, OPT-.045, 'optimal 6.10', fontsize=9, color='#157F6B')
ax.annotate('best so far 6.561\n(still running)', xy=(89098,6.561), xytext=(9e3,6.44),
            fontsize=9.5, color='#1D6FB8',
            arrowprops=dict(arrowstyle='->', color='#1D6FB8', lw=1.2))
plt.colorbar(sc, ax=ax, label='episodes', pad=.02)
ax.set(xscale='log', xlim=(9e3,2.2e6), ylim=(6.02,7.45),
       xlabel='gradient steps (log)', ylabel='mean tour length',
       title='D · how far we are from their number')
ax.grid(alpha=.25, lw=.6, which='both')

for a in fig.axes[:4]:
    for sp in ('top','right'): a.spines[sp].set_visible(False)
fig.suptitle('CCE on RP-DQN — current state', fontsize=17, fontweight='600', y=.975, x=.09, ha='left')
fig.text(.09,.945,f'single seed throughout · dashed lines and arrows = still running · compare point {common:,.0f} gradient steps',
         fontsize=10.5, color='#5A6875', ha='left')
o = 'docs/figures/real/claim2/cvrp_neural/dashboard.png'
plt.savefig(o, dpi=140, bbox_inches='tight', facecolor='white')
print('wrote', o, '| matched compare point:', f'{common:,.0f}')
