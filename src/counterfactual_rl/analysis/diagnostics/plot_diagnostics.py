"""
Offline CCE diagnostics — plot stage (cheap, local). Env-agnostic.

Reads a diagnostics.npz (from compute_diagnostics.py [Connect Four] or
compute_diagnostics_fl.py [FrozenLake]) and emits figures D1-D9 answering:

  Q1  Is there structure?       D1 histogram, D2 Lorenz, D3 over-training
  Q2  Is CCE just TD?           D4 CCE-vs-|TD| scatter
  Q3  Who matches truth?        D5 CCE-vs-stakes, D6 |TD|-vs-stakes, D7 corr bars
  qualitative / summary         D8 worked examples, D9 verdict card

Ground-truth keys are generic ('truth_spread' / 'truth_qvalues' / 'truth_regret');
old Connect Four npz files using 'mcts_*' keys are still accepted. Set the
referee's name with --truth-label (e.g. "MCTS" or "optimal Q*").

Usage:
    python -m counterfactual_rl.analysis.diagnostics.plot_diagnostics \
        --npz <.../diagnostics.npz> --out <.../figs> [--truth-label "optimal Q*"]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

CCE_C = '#4CAF50'    # green
TD_C = '#2196F3'     # blue
TRUTH_C = '#FF9800'  # orange


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def spearman_boot(a, b, reps=2000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float('nan'), (float('nan'), float('nan'))
    rho = spearmanr(a, b).correlation
    rng = np.random.default_rng(seed)
    boots = [spearmanr(a[i], b[i]).correlation for i in (rng.integers(0, a.size, a.size) for _ in range(reps))]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(rho), (float(lo), float(hi))


def _norm(d):
    """Normalize ground-truth keys: accept generic 'truth_*' or legacy 'mcts_*'."""
    for g, m in (('truth_spread', 'mcts_spread'),
                 ('truth_qvalues', 'mcts_qvalues'),
                 ('truth_regret', 'mcts_regret')):
        if g not in d and m in d:
            d[g] = d[m]
    return d


def _scatter(ax, x, y, xlabel, ylabel):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ax.hexbin(x, y, gridsize=40, cmap='Greys', mincnt=1, bins='log')
    rho, (lo, hi) = spearman_boot(x, y)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f'ρ = {rho:.2f}  [{lo:.2f}, {hi:.2f}]')
    ax.grid(True, alpha=0.3)
    return rho


# ── Per-env board/grid renderers for D8 ───────────────────────────────────────

def _render_c4(ax, obs):
    grid = obs.reshape(6, 7, 2)
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 5.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for r in range(6):
        for c in range(7):
            col = '#E53935' if grid[r, c, 0] > 0.5 else '#FDD835' if grid[r, c, 1] > 0.5 else 'white'
            ax.add_patch(plt.Circle((c, 5 - r), 0.42, color=col, ec='#1565C0', lw=1.5))
    ax.set_facecolor('#1565C0')


def _render_fl(ax, fl_map, ncols, state):
    nrows = len(fl_map)
    ax.set_xlim(-0.5, ncols - 0.5); ax.set_ylim(-0.5, nrows - 0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    ax.invert_yaxis()
    colors = {'H': '#37474F', 'G': '#43A047', 'S': '#90CAF9', 'F': '#E1F5FE'}
    for r in range(nrows):
        for c in range(ncols):
            tile = fl_map[r][c]
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=colors.get(tile, 'white'), edgecolor='#B0BEC5'))
    ar, ac = divmod(int(state), ncols)
    ax.add_patch(plt.Circle((ac, ar), 0.3, color='#E53935', zorder=3))  # agent = red dot


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_d1(d, out):
    cce = d['cce_score'][d['chunk'] == d['chunk'].max()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cce, bins=50, color=CCE_C, alpha=0.85)
    ax.axvline(cce.mean(), color='k', ls='--', lw=1, label=f'mean={cce.mean():.3f}')
    ax.set_xlabel('CCE consequence score'); ax.set_ylabel('count')
    ax.set_title(f'D1 — CCE score distribution (final)   Gini={gini(cce):.2f}')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")


def fig_d2(d, out):
    cce = np.sort(d['cce_score'][d['chunk'] == d['chunk'].max()])
    cum = np.cumsum(cce) / cce.sum() if cce.sum() > 0 else np.linspace(0, 1, cce.size)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(np.linspace(0, 1, cce.size), cum, color=CCE_C, lw=2, label='CCE mass')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='uniform (no structure)')
    ax.set_xlabel('fraction of transitions (sorted)')
    ax.set_ylabel('cumulative share of total consequence')
    ax.set_title(f'D2 — concentration of consequence   Gini={gini(cce):.2f}')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")


def fig_d3(d, out):
    chunks = np.unique(d['chunk'])
    data = [d['cce_score'][d['chunk'] == c] for c in chunks]
    width = max(1, (chunks.max() - chunks.min()) / max(len(chunks), 1) * 0.6)
    fig, ax = plt.subplots(figsize=(8, 4))
    parts = ax.violinplot(data, positions=chunks, widths=width, showmeans=True)
    for b in parts['bodies']:
        b.set_facecolor(CCE_C); b.set_alpha(0.6)
    ax.set_xlabel('training checkpoint (episodes/chunks)'); ax.set_ylabel('CCE consequence score')
    ax.set_title('D3 — does pivotal structure emerge over training?')
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")


def fig_d4(d, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, np.abs(d['td_error']), d['cce_score'], '|TD error|', 'CCE score')
    ax.set_title('D4 — Is CCE just TD error?   ' + ax.get_title())
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")


def fig_d5_d6(d, label, out5, out6):
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, d['truth_spread'], d['cce_score'], f'{label} value spread (ground-truth stakes)', 'CCE score')
    ax.set_title('D5 — CCE vs ground truth   ' + ax.get_title())
    fig.savefig(out5, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved {out5}")

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, d['truth_spread'], np.abs(d['td_error']), f'{label} value spread (ground-truth stakes)', '|TD error|')
    ax.set_title('D6 — TD error vs ground truth   ' + ax.get_title())
    fig.savefig(out6, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved {out6}")


def fig_d7(d, out):
    pairs = [
        ('corr(CCE, truth)', d['truth_spread'], d['cce_score'], CCE_C),
        ('corr(|TD|, truth)', d['truth_spread'], np.abs(d['td_error']), TD_C),
        ('corr(CCE, |TD|)', np.abs(d['td_error']), d['cce_score'], '#9E9E9E'),
    ]
    labels, rhos, los, his, cols = [], [], [], [], []
    for name, a, b, c in pairs:
        rho, (lo, hi) = spearman_boot(a, b)
        labels.append(name); rhos.append(rho); los.append(rho - lo); his.append(hi - rho); cols.append(c)
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(labels))
    ax.barh(y, rhos, xerr=[los, his], color=cols, alpha=0.85, capsize=5)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('Spearman ρ (95% bootstrap CI)')
    ax.set_title('D7 — which signal tracks ground truth?')
    ax.grid(True, axis='x', alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")
    return {'cce_truth': rhos[0], 'td_truth': rhos[1], 'cce_td': rhos[2]}


def fig_d8(d, out):
    final = np.where(d['chunk'] == d['chunk'].max())[0]
    cce = d['cce_score'][final]
    picks = {
        'highest CCE': final[np.argmax(cce)],
        'lowest CCE': final[np.argmin(cce)],
        'highest stakes': final[np.argmax(d['truth_spread'][final])],
    }
    is_c4 = 'obs' in d
    act_labels = list('0123') if is_c4 else ['L', 'D', 'R', 'U']
    fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 7))
    for col, (label, i) in enumerate(picks.items()):
        if is_c4:
            _render_c4(axes[0, col], d['obs'][i])
        else:
            _render_fl(axes[0, col], [str(r) for r in d['fl_map']], int(d['fl_ncols']), d['state'][i])
        axes[0, col].set_title(f"{label}\nCCE={d['cce_score'][i]:.3f}  "
                               f"|TD|={abs(d['td_error'][i]):.3f}\nstakes={d['truth_spread'][i]:.2f}", fontsize=9)
        q = d['truth_qvalues'][i].astype(float).copy()
        bars = axes[1, col].bar(np.arange(len(q)), q, color=TRUTH_C, alpha=0.85)
        bars[int(d['taken_action'][i])].set_color('#E53935')
        axes[1, col].set_xlabel('action'); axes[1, col].set_ylabel('ground-truth Q-value')
        axes[1, col].set_xticks(range(len(q))); axes[1, col].set_xticklabels(act_labels)
        axes[1, col].grid(True, axis='y', alpha=0.3)
    fig.suptitle('D8 — worked examples (red = taken action)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")


def fig_d9(d, corr, label, out):
    g = gini(d['cce_score'][d['chunk'] == d['chunk'].max()])
    q1 = "structure EXISTS — some moves matter far more" if g > 0.2 else "FLAT — little exploitable structure"
    q2 = "REDUNDANT with TD error" if corr['cce_td'] > 0.7 else "DISTINCT from TD error"
    q3 = (f"CCE tracks {label} truth BETTER than TD" if corr['cce_truth'] > corr['td_truth']
          else f"TD tracks {label} truth better than CCE")
    lines = [
        "CCE DIAGNOSTIC — VERDICT",
        "",
        f"Q1  Is there structure?     Gini={g:.2f}  ->  {q1}",
        f"Q2  Just TD in disguise?    rho(CCE,|TD|)={corr['cce_td']:.2f}  ->  {q2}",
        f"Q3  Who matches truth?      rho(CCE,truth)={corr['cce_truth']:.2f} vs "
        f"rho(|TD|,truth)={corr['td_truth']:.2f}",
        f"                            ->  {q3}",
    ]
    fig, ax = plt.subplots(figsize=(9, 4)); ax.axis('off')
    ax.text(0.02, 0.95, "\n".join(lines), va='top', ha='left', family='monospace', fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved {out}")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--truth-label', default='MCTS')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = _norm(dict(np.load(args.npz, allow_pickle=False)))
    print(f"Loaded {d['cce_score'].shape[0]} rows, chunks {sorted(np.unique(d['chunk']))}")

    p = lambda n: os.path.join(args.out, n)
    fig_d1(d, p('d1_cce_histogram.png'))
    fig_d2(d, p('d2_lorenz.png'))
    fig_d3(d, p('d3_over_training.png'))
    fig_d4(d, p('d4_cce_vs_td.png'))
    fig_d5_d6(d, args.truth_label, p('d5_cce_vs_truth.png'), p('d6_td_vs_truth.png'))
    corr = fig_d7(d, p('d7_corr_bars.png'))
    fig_d8(d, p('d8_examples.png'))
    fig_d9(d, corr, args.truth_label, p('d9_verdict.png'))


if __name__ == '__main__':
    main()
