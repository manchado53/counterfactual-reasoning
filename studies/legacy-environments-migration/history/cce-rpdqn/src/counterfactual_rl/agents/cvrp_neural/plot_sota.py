"""Where our reproduction sits against published CVRP n=20 results, on THEIR test set."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

OPT = 6.10
# RP-DQN arXiv 2104.12226 Table 1, n=20, mean tour on Nazari's 10,000-instance test set.
PUB = [
    ('Gurobi (exact)',        6.10, 'solver'),
    ('Wu et al. 5000 steps',  6.12, 'search'),
    ('LKH3',                  6.14, 'solver'),
    ('RP-DQN sampling 1024',  6.24, 'ml-sample'),
    ('Kool sampling 1280',    6.25, 'ml-sample'),
    ('Falkner sampling 1280', 6.26, 'ml-sample'),
    ('RP-DQN greedy',         6.36, 'ml-greedy'),
    ('Kool greedy',           6.40, 'ml-greedy'),
    ('OR-Tools',              6.43, 'search'),
    ('Falkner greedy',        6.47, 'ml-greedy'),
    ('Nazari greedy',         6.59, 'ml-greedy'),
]
# ours, greedy, measured on the SAME test set (job 275450)
OURS = [('ours — PER (best)', 6.5819), ('ours — PER', 6.5948),
        ('ours — uniform', 6.6570), ('ours — PER (nspu32)', 6.7185),
        ('ours — CCE-mul', 6.7374)]

COL = dict(solver='#157F6B', search='#5A6875', **{'ml-sample':'#8A94A0', 'ml-greedy':'#1D6FB8'})
rows = [(n, v, COL[k]) for n, v, k in PUB] + [(n, v, '#BF3B30') for n, v in OURS]
rows.sort(key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(12.5, 8))
y = np.arange(len(rows))[::-1]
for yy, (n, v, c) in zip(y, rows):
    gap = 100*(v-OPT)/OPT
    ax.plot([OPT, v], [yy, yy], color=c, lw=1.3, alpha=.35, zorder=1)
    ax.plot(v, yy, 'o', ms=13 if c == '#BF3B30' else 10, color=c,
            mec='white', mew=1.6, zorder=4)
    w = 'bold' if c == '#BF3B30' else 'normal'
    ax.text(v+.012, yy, f'{v:.3f}   {gap:5.2f}%', va='center', fontsize=10.5,
            fontfamily='monospace', color=c if c == '#BF3B30' else '#121A23', fontweight=w)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=11)
# fontweight/color are per-Text, not list-valued on set_yticklabels
for tick, r in zip(ax.get_yticklabels(), rows):
    if r[2] == '#BF3B30':
        tick.set_color('#BF3B30'); tick.set_fontweight('bold')

# band: everything between RP-DQN greedy and us
rp = 6.36
ax.axvspan(rp, 6.5819, color='#BF3B30', alpha=.07, zorder=0)
ax.axvline(rp, color='#1D6FB8', ls='--', lw=1.6, zorder=2)
ax.axvline(OPT, color='#157F6B', lw=1.6, zorder=2)
ax.annotate('RP-DQN greedy\nthe arm we set out to beat', xy=(rp, 4.6),
            xytext=(6.20, 3.4), fontsize=9.5, color='#1D6FB8', ha='left',
            arrowprops=dict(arrowstyle='->', color='#1D6FB8', lw=1.2))
ax.annotate('our best is 3.6 points\nbehind the paper we\nare reproducing',
            xy=(6.47, 5.0), xytext=(6.655, 8.6), fontsize=10, color='#BF3B30', ha='center',
            arrowprops=dict(arrowstyle='->', color='#BF3B30', lw=1.2))

ax.set(xlim=(6.06, 6.90), xlabel='mean tour length on RP-DQN\'s 10,000-instance test set (lower is better)',
       title='CVRP n=20 — where our reproduction actually sits')
ax.grid(alpha=.25, lw=.6, axis='x')
for sp in ('top','right','left'): ax.spines[sp].set_visible(False)
ax.tick_params(axis='y', length=0)
h = [plt.Line2D([],[],marker='o',ls='',ms=9,mec='white',mew=1.3,color=c,label=l)
     for l,c in [('exact / heuristic solver','#157F6B'),('local search','#5A6875'),
                 ('ML, sampling','#8A94A0'),('ML, greedy','#1D6FB8'),('ours (greedy)','#BF3B30')]]
ax.legend(handles=h, frameon=False, fontsize=9.5, loc='upper right',
          bbox_to_anchor=(1.0, .78))
plt.tight_layout()
o = 'docs/figures/real/claim2/cvrp_neural/sota_comparison.png'
plt.savefig(o, dpi=150, bbox_inches='tight', facecolor='white'); print('wrote', o)
