"""The CVRP scaling curve: every baseline run we have, against gradient steps."""
import glob, json, os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = 'src/counterfactual_rl/agents/cvrp_neural/runs'
OPT, THEIRS, THEIR_STEPS = 6.10, 6.36, 1_000_000

def read(j):
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
    if cfg.get('algorithm') == 'consequence-dqn': return None      # baselines only
    n = int(cfg.get('n_steps_per_update', 512))
    return dict(gs=np.array(ep)*24/n, tour=np.array(t), nspu=n,
                eps=int(cfg.get('n_episodes', 0)), alg=cfg.get('algorithm'))

runs = [(j, read(os.path.basename(j))) for j in sorted(glob.glob(f'{R}/2*'))]
runs = [(os.path.basename(j), d) for j, d in runs if d and d['gs'][-1] > 1500]

fig, ax = plt.subplots(figsize=(13, 7.5))
cmap = plt.cm.viridis
lo, hi = np.log10(6000), np.log10(260000)
for j, d in runs:
    c = cmap((np.log10(max(d['eps'], 6000)) - lo)/(hi - lo))
    ax.plot(d['gs'], d['tour'], lw=1.5, color=c, alpha=.85)
    ax.plot(d['gs'][-1], d['tour'][-1], 'o', ms=8, color=c, mec='white', mew=1.3, zorder=4)

# endpoints only, for the fit
ends = sorted([(d['gs'][-1], float(np.mean(d['tour'][-5:]))) for _, d in runs])
X = np.array([e[0] for e in ends]); Y = np.array([e[1] for e in ends])
m = X >= 30000
b, a = np.polyfit(np.log10(X[m]), Y[m], 1)
xf = np.logspace(np.log10(30000), np.log10(1.3e6), 60)
ax.plot(xf, a + b*np.log10(xf), ls=':', lw=2, color='#BF3B30', zorder=3)
pred = a + b*np.log10(THEIR_STEPS)
ax.plot(THEIR_STEPS, pred, 'D', ms=10, color='#BF3B30', mec='white', mew=1.5, zorder=5)
ax.annotate(f'our curve extrapolated\nto their budget: {pred:.2f}',
            xy=(THEIR_STEPS, pred), xytext=(2.0e5, 6.92), fontsize=10, color='#BF3B30',
            arrowprops=dict(arrowstyle='->', color='#BF3B30', lw=1.3))

ax.scatter([THEIR_STEPS], [THEIRS], marker='*', s=430, color='#C06A08',
           zorder=6, edgecolor='white', linewidth=1.4)
ax.annotate('RP-DQN published\n6.36 @ ~1M gradient steps',
            xy=(THEIR_STEPS, THEIRS), xytext=(2.4e5, 6.20), fontsize=10.5, color='#C06A08',
            arrowprops=dict(arrowstyle='->', color='#C06A08', lw=1.3))
ax.axhline(OPT, color='#157F6B', lw=1.6)
ax.text(1.6e3, OPT-.055, 'optimal 6.10 (Gurobi)', fontsize=10, color='#157F6B')
ax.axhline(THEIRS, ls='--', lw=1.3, color='#C06A08', alpha=.55)

ax.axvspan(X[m].max(), 1.3e6, color='#BF3B30', alpha=.05)
ax.text(3.3e5, 7.9, 'never run\n(≈38 GPU-h)', fontsize=10, color='#BF3B30', ha='center')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
cb = plt.colorbar(sm, ax=ax, pad=.015)
cb.set_label('episodes (log)', fontsize=10)
cb.set_ticks([np.log10(v) for v in (6250, 25000, 50000, 125000, 250000)])
cb.set_ticklabels(['6k', '25k', '50k', '125k', '250k'])

ax.set(xscale='log', xlim=(1.5e3, 1.4e6), ylim=(6.0, 9.6),
       xlabel='gradient steps (log)', ylabel='mean tour length  (lower is better)',
       title=f'CVRP n=20 — every baseline run, {len(runs)} in total')
ax.grid(alpha=.25, lw=.6, which='both')
for sp in ('top','right'): ax.spines[sp].set_visible(False)
ax.text(1.6e3, 9.35, f'fit over ≥30k steps:  tour ≈ {a:.2f} {b:+.3f}·log₁₀(steps)',
        fontsize=10, color='#BF3B30', fontfamily='monospace')
plt.tight_layout()
o = 'docs/figures/real/claim2/cvrp_neural/scaling_curve.png'
plt.savefig(o, dpi=150, bbox_inches='tight', facecolor='white')
print('wrote', o)
print(f'  {len(runs)} baseline runs | fit slope {b:+.3f}/decade | extrapolation at 1M steps: {pred:.3f}')
print(f'  best measured: {Y.min():.3f} at {X[Y.argmin()]:,.0f} steps')
