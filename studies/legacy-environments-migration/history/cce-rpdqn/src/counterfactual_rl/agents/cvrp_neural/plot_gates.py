"""Plot the actual CVRP/RP-DQN run data: Gate 1 curves, the replay-ratio probe, Gate 2."""

import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = 'src/counterfactual_rl/agents/cvrp_neural/runs'
OPT, THEIRS = 6.10, 6.36


def curve(job):
    """-> (episodes, mean tour length) from a run's metrics.log."""
    ep, tour = [], []
    for line in open(f'{RUNS}/{job}/metrics.log'):
        if line.startswith('#'):
            continue
        p = line.split()
        if not p or p[0] == 'episode':
            continue
        ep.append(int(p[0])); tour.append(-float(p[5]))
    return np.array(ep), np.array(tour)


fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

# ---- A: Gate 1, their real schedule ----------------------------------------
ax = axes[0]
for job, lab, c in [('274699', 'PER (their arm)', '#1D6FB8'), ('274700', 'uniform', '#8A94A0')]:
    e, t = curve(job)
    ax.plot(e / 1000, t, lw=1.5, color=c, label=f'{lab} → {t[-1]:.2f}')
ax.axhline(THEIRS, ls='--', lw=1.6, color='#C06A08')
ax.text(252, THEIRS + .05, 'RP-DQN greedy 6.36', color='#C06A08', fontsize=9, ha='right')
ax.axhline(OPT, lw=1.4, color='#157F6B')
ax.text(252, OPT - .16, 'optimal 6.10', color='#157F6B', fontsize=9, ha='right')
ax.axvline(125, color='#C8CFD6', ls=':', lw=1.2)
ax.text(127, 6.6, 'exploration ends', fontsize=8, color='#5A6875')
ax.set(xlabel='episodes (thousands)', ylabel='mean tour length  (lower is better)',
       title='A · Gate 1 — 250k episodes, ~12k gradient steps', ylim=(5.8, 11.8), xlim=(0, 255))
ax.legend(frameon=False, fontsize=9, loc='center right')
ax.grid(alpha=.25, lw=.6)

# ---- B: replay-ratio probe, plotted against GRADIENT STEPS ------------------
ax = axes[1]
mpath = glob.glob(f'{os.path.dirname(RUNS)}/experiments/2026-09/replay_ratio_*/manifest.json')
man = json.load(open(mpath[0])) if mpath else {}
cells = sorted(man.items(), key=lambda kv: -kv[1]['n_steps_per_update'])
cmap = plt.cm.viridis(np.linspace(.08, .88, max(len(cells), 1)))
for (jid, cfg), c in zip(cells, cmap):
    if not os.path.exists(f'{RUNS}/{jid}/metrics.log'):
        continue
    e, t = curve(jid)
    gs = e * 25 / cfg['n_steps_per_update']
    ax.plot(gs, t, lw=1.6, color=c,
            label=f"nspu={cfg['n_steps_per_update']:<4} ({e[-1]/1000:.0f}k eps)")
e, t = curve('274699')
ax.plot(e * 25 / 512, t, lw=1.4, ls='--', color='#BF3B30', label='Gate 1 (250k eps)')
ax.axhline(THEIRS, ls='--', lw=1.6, color='#C06A08')
ax.set(xscale='log', xlabel='gradient steps  (log)', ylabel='mean tour length',
       title='B · Same axis, wildly different data — do they collapse?',
       ylim=(5.8, 11.8))
ax.legend(frameon=False, fontsize=8.5, loc='upper right')
ax.grid(alpha=.25, lw=.6, which='both')

# ---- C: Gate 2 signal vs horizon -------------------------------------------
ax = axes[2]
log = os.path.expanduser('~/cvrp_rpdqn_logs/274711.out')
blocks, cur = [], None
if os.path.exists(log):
    for line in open(log):
        if 'CHECKPOINT' in line:
            cur = {}; blocks.append((line.split()[-1], cur))
        m = re.match(r'\s*horizon\s+(\d+)\s+rho\s+([-\d.]+)', line)
        if m and cur is not None:
            cur[int(m.group(1))] = float(m.group(2))
for (name, d), c in zip(blocks, ['#9BD0C4', '#3FA890', '#0F5D4E']):
    if not d:
        continue
    ks = sorted(d)
    ep = int(re.findall(r'\d+', name)[0])
    ax.plot(ks, [d[k] for k in ks], 'o-', lw=1.8, ms=5, color=c, label=f'{ep/1000:.0f}k episodes')
ax.axhline(1.0, color='#C8CFD6', lw=1)
ax.set(xlabel='cf_horizon  (1 = no rollout at all)',
       ylabel='Spearman ρ vs exact full-horizon score',
       title='C · Gate 2 — does the rollout beat no rollout?', ylim=(0, 1.06))
ax.legend(frameon=False, fontsize=9, loc='lower right', title='checkpoint', title_fontsize=8.5)
ax.grid(alpha=.25, lw=.6)

for a in axes:
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
plt.tight_layout()
out = 'docs/figures/real/claim2/cvrp_neural'
os.makedirs(out, exist_ok=True)
plt.savefig(f'{out}/gates_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f'wrote {out}/gates_overview.png')
