"""Win rate and return over training for every arm, plus the finished horizon ladder.

'win_rate' on this env is quality = optimal / mean_tour_length, so 1.000 is optimal and
RP-DQN's published greedy arm is 6.10/6.36 = 0.959. Return is -mean_tour_length. Both
are monotone transforms of tour length -- plotted because the claim-2 pipeline consumes
win_rate, and because the exploration phase is much easier to see on them.
"""
import os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = 'src/counterfactual_rl/agents/cvrp_neural/runs'
OPT, THEIRS = 6.10, 6.36
NOISE = 0.05

ARMS = [('274713','uniform','#8A94A0','-'),
        ('274714','PER','#1D6FB8','-'),
        ('274719','CCE-mul h=1','#5A6875','--'),
        ('274715','CCE-add h=10','#C06A08','--'),
        ('274716','CCE-mul h=10','#B0308F','-'),
        ('274718','CCE-add h=25','#8E6E1F','--'),
        ('274717','CCE-mul h=25','#157F6B','-')]

def load(j):
    m = f'{R}/{j}/metrics.log'
    if not os.path.exists(m): return None
    cfg, ep, wr, ret = {}, [], [], []
    for l in open(m):
        if l.startswith('#'):
            k = l[1:].split(':', 1)
            if len(k) == 2: cfg[k[0].strip()] = k[1].strip()
            continue
        p = l.split()
        if not p or p[0] == 'episode': continue
        ep.append(int(p[0])); wr.append(float(p[3].rstrip('%'))/100); ret.append(float(p[5]))
    if len(ep) < 3: return None
    n = int(cfg.get('n_steps_per_update', 512))
    return dict(cfg=cfg, ep=np.array(ep), gs=np.array(ep)*24/n,
                wr=np.array(wr), ret=np.array(ret), nspu=n)

def smooth(y, w=7):
    """Centred moving average with EDGE-SAFE ends.

    np.convolve(..., mode='same') zero-pads, which drags the first and last few
    points toward zero and draws a cliff at both ends of every curve. Reflect the
    signal instead so the ends are averaged against real data.
    """
    if len(y) < w: return y
    pad = w // 2
    yp = np.pad(y, pad, mode='reflect')
    return np.convolve(yp, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

# exploration ends at exploration_fraction * n_episodes -> convert to gradient steps
d0 = load('274713')
frac = float(d0['cfg'].get('exploration_fraction', 0.5))
expl_end = frac * int(d0['cfg']['n_episodes']) * 24 / d0['nspu']

# ---------- A: win rate (quality) ----------
ax = axes[0]
ax.axvspan(0, expl_end, color='#9AA5B1', alpha=.18, zorder=0)
ax.text(expl_end*.5, .335, 'EXPLORATION\nBoltzmann 1.0 → 0.02', ha='center', fontsize=9,
        color='#4A5560', fontweight='600', zorder=1)
for j,l,c,ls in ARMS:
    d = load(j)
    if d is None: continue
    ax.plot(d['gs'], smooth(d['wr']), lw=1.7, color=c, ls=ls, label=l)
ax.axhline(OPT/THEIRS, ls='--', lw=1.6, color='#C06A08')
ax.text(11800, OPT/THEIRS+.008, 'RP-DQN greedy 0.959', ha='right', fontsize=9, color='#C06A08')
ax.axhline(1.0, lw=1.4, color='#157F6B')
ax.text(11800, 1.006, 'optimal 1.000', ha='right', fontsize=9, color='#157F6B')
ax.set(xlabel='gradient steps', ylabel='win rate  =  optimal / mean tour   (higher is better)',
       title='A · win rate over training', ylim=(.30, 1.04), xlim=(0, 12000))
ax.legend(frameon=False, fontsize=9, loc='lower right', ncol=2)
ax.grid(alpha=.25, lw=.6)

# ---------- B: return ----------
ax = axes[1]
ax.axvspan(0, expl_end, color='#9AA5B1', alpha=.18, zorder=0)
ax.text(expl_end*.5, -19.0, 'EXPLORATION', ha='center', fontsize=9,
        color='#4A5560', fontweight='600', zorder=1)
for j,l,c,ls in ARMS:
    d = load(j)
    if d is None: continue
    ax.plot(d['gs'], smooth(d['ret']), lw=1.7, color=c, ls=ls, label=l)
ax.axhline(-THEIRS, ls='--', lw=1.6, color='#C06A08')
ax.text(11800, -THEIRS+.25, 'RP-DQN greedy −6.36', ha='right', fontsize=9, color='#C06A08')
ax.set(xlabel='gradient steps', ylabel='mean episode return  =  − tour length',
       title='B · return over training', ylim=(-20.5, -5.6), xlim=(0, 12000))
ax.grid(alpha=.25, lw=.6)

# ---------- C: the finished ladder ----------
ax = axes[2]
RHO = {0.626:'h=1', 0.773:'h=10', 1.000:'h=25'}
MUL = [(0.626, 7.429), (0.773, 7.397), (1.000, 7.352)]
ADD = [(0.773, 7.433), (1.000, 7.404)]
for pts, col, lab, mk in [(MUL,'#B0308F','multiplicative','o'), (ADD,'#C06A08','additive','s')]:
    ax.plot([p[0] for p in pts], [p[1] for p in pts], mk+'-', color=col, lw=2.2, ms=10,
            mec='white', mew=1.5, label=lab)
u = 7.364
ax.axhline(u, color='#8A94A0', lw=1.8)
ax.axhspan(u-NOISE, u+NOISE, color='#8A94A0', alpha=.16)
ax.text(1.005, u, ' uniform', va='center', fontsize=10, color='#5A6875')
for x, y in MUL:
    ax.annotate(RHO[x], xy=(x,y), xytext=(0,-16), textcoords='offset points',
                ha='center', fontsize=9.5, color='#B0308F', fontweight='600')
ax.set(xlabel='Gate-2 ρ  —  how well the score matches the exact consequence',
       ylabel='mean tour @ 11,718 matched steps',
       title='C · better score → better learning (but only back to par)',
       xlim=(.58, 1.06))
ax.legend(frameon=False, fontsize=10, loc='upper right', title='mixing', title_fontsize=9)
ax.grid(alpha=.25, lw=.6)
ax.text(.60, u+NOISE+.004, 'shaded = ±0.05 noise floor', fontsize=8.5, color='#5A6875')

for a in axes:
    for sp in ('top','right'): a.spines[sp].set_visible(False)
fig.suptitle('CCE on RP-DQN — all arms complete, nspu=512 / 250k episodes, single seed',
             fontsize=15, fontweight='600', y=1.02, x=.085, ha='left')
plt.tight_layout()
o = 'docs/figures/real/claim2/cvrp_neural/training_winrate_return.png'
plt.savefig(o, dpi=145, bbox_inches='tight', facecolor='white')
print('wrote', o)
print(f'exploration ends at {expl_end:,.0f} gradient steps ({frac:.0%} of episodes)')
print('\nfinal win rate / return per arm:')
for j,l,c,ls in ARMS:
    d = load(j)
    if d: print(f'  {l:<14} win_rate {d["wr"][-5:].mean():.4f}   return {d["ret"][-5:].mean():+.3f}')
