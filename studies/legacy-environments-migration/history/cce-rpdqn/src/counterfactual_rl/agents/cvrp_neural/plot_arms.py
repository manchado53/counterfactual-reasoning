"""All CVRP arms so far, in progress. Two budgets side by side, plus the horizon ladder.

Runs still in flight are drawn to wherever they have reached and marked. Nothing here
is a result yet -- single seed, and several curves are far from their target episode
count. It exists to show shape, not to be read off.
"""
import os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

RUNS = 'src/counterfactual_rl/agents/cvrp_neural/runs'
OPT, THEIRS = 6.10, 6.36

def load(j):
    m = f'{RUNS}/{j}/metrics.log'
    if not os.path.exists(m): return None
    cfg, ep, t = {}, [], []
    for l in open(m):
        if l.startswith('#'):
            p = l[1:].split(':', 1)
            if len(p) == 2: cfg[p[0].strip()] = p[1].strip()
            continue
        p = l.split()
        if not p or p[0] == 'episode': continue
        ep.append(int(p[0])); t.append(-float(p[5]))
    if not ep: return None
    n = int(cfg.get('n_steps_per_update', 512))
    return dict(cfg=cfg, ep=np.array(ep), tour=np.array(t),
                gs=np.array(ep) * 24 / n, nspu=n,
                target=int(cfg.get('n_episodes', 0)))

# job -> (panel, label, colour, linestyle)
C = dict(uni='#8A94A0', per='#1D6FB8', add='#C06A08', mul='#B0308F', h1='#5A6875', h25='#157F6B')
SPEC = {
    '274713': (0, 'uniform',        C['uni'], '-'),
    '274714': (0, 'PER',            C['per'], '-'),
    '274715': (0, 'CCE additive',   C['add'], '-'),
    '274716': (0, 'CCE multiplic.', C['mul'], '-'),
    '274720': (1, 'uniform',        C['uni'], '-'),
    '274721': (1, 'PER',            C['per'], '-'),
    '274722': (1, 'CCE additive',   C['add'], '-'),
    '274723': (1, 'CCE multiplic.', C['mul'], '-'),
    '274719': (2, 'CCE-mul h=1  (no rollout)', C['h1'],  '-'),
    '274716b':(2, 'CCE-mul h=10',    C['mul'], '-'),
    '274717': (2, 'CCE-mul h=25',    C['h25'], '-'),
    '274718': (2, 'CCE-add h=25',    C['add'], '--'),
}
TITLES = ['A · nspu=512, 250k eps\n(the budget where PER ties uniform)',
          'B · nspu=32, 50k eps\n(8x the gradient steps, cheaper)',
          'C · horizon ladder, CCE-mul\n(does a better score train better?)']

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
for jid, (panel, lab, col, ls) in SPEC.items():
    d = load(jid.rstrip('b'))
    if d is None: continue
    ax = axes[panel]
    ax.plot(d['gs'], d['tour'], lw=1.8, color=col, ls=ls, label=lab)
    done = d['target'] and d['ep'][-1] >= d['target'] * 0.98
    ax.plot(d['gs'][-1], d['tour'][-1], 'o' if done else '>', ms=8 if done else 9,
            color=col, mec='white', mew=1.3, zorder=5)
    if not done:
        ax.plot(d['gs'][-1], d['tour'][-1], marker='>', ms=9, color=col,
                mec='white', mew=1.3, zorder=5)

for i, ax in enumerate(axes):
    ax.axhline(THEIRS, ls='--', lw=1.7, color='#C06A08', alpha=.75)
    ax.axhline(OPT, lw=1.4, color='#157F6B', alpha=.75)
    ax.set(xscale='log', ylim=(6.0, 11.6), xlabel='gradient steps (log)',
           title=TITLES[i])
    ax.grid(alpha=.25, lw=.6, which='both')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.text(ax.get_xlim()[1]*.96, THEIRS+.10, 'RP-DQN greedy 6.36',
            ha='right', fontsize=8.5, color='#C06A08')
    ax.text(ax.get_xlim()[1]*.96, OPT-.26, 'optimal 6.10',
            ha='right', fontsize=8.5, color='#157F6B')
axes[0].set_ylabel('mean tour length  (lower is better)')
fig.text(.5, .005, '▶ = still running   ● = finished   |   single seed, in progress — shape only, not results',
         ha='center', fontsize=9.5, color='#5A6875')
plt.tight_layout(rect=[0, .035, 1, 1])
o = 'docs/figures/real/claim2/cvrp_neural/arms_in_progress.png'
plt.savefig(o, dpi=150, bbox_inches='tight', facecolor='white'); print('wrote', o)

print(f"\n{'job':<8}{'arm':<22}{'nspu':>5}{'episodes':>11}{'target':>9}{'steps':>9}{'tour':>8}")
for jid in ['274713','274714','274715','274716','274720','274721','274722','274723','274719','274716','274717','274718']:
    d = load(jid)
    if d is None: print(f"{jid:<8}(no evals yet)"); continue
    lab = SPEC.get(jid, (0,'?','',''))[1]
    h = d['cfg'].get('cf_horizon','-') if 'consequence' in d['cfg'].get('algorithm','') else '-'
    print(f"{jid:<8}{lab+(' h='+h if h!='-' else ''):<22}{d['nspu']:>5}"
          f"{d['ep'][-1]:>11,}{d['target']:>9,}{int(d['gs'][-1]):>9,}{d['tour'][-1]:>8.3f}")
