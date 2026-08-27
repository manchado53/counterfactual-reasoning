"""
What routing-hard actually is, in one picture.

  (a) the best anyone can do — 9 of 10 served, one abandoned, shift nearly full
  (b) a real trained policy that stranded itself — everything delivered is wiped
  (c) the reward the network sees in each case: eleven zeros, then one number

Panels (a) and (b) are real routes: the exact oracle plan, and a seed from the
pre-registered sweep that ended stranded.

Run:
    python -m counterfactual_rl.analysis.claim2.routing_hard_explainer \
        --runs-dir src/counterfactual_rl/agents/cvrp/runs --out docs/figures/real/claim2
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from counterfactual_rl.envs.routing_budget import BudgetRoutingEnv, DEPOT
from counterfactual_rl.analysis.claim1.cvrp.budget_oracle import optimal_plan, optimal_served

GREEN, RED, BLUE, AMBER, INK = '#2C6E49', '#9E2F27', '#1B5FA8', '#B4600F', '#122236'


def draw_map(ax, env, tour, stranded, title, sub):
    xy = env.node_xy
    served = [p for p in tour if p != DEPOT]
    skipped = sorted(set(range(1, env.n_nodes)) - set(served))
    for i in range(len(tour) - 1):
        a, b = xy[tour[i]], xy[tour[i + 1]]
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(
            arrowstyle='-|>', lw=1.9, shrinkA=10, shrinkB=10,
            color=(RED if stranded else BLUE), alpha=.85))
    col = '#9aa5b1' if stranded else GREEN
    if served:
        ax.scatter(xy[served, 0], xy[served, 1], s=230, color=col, zorder=3)
    if skipped:
        ax.scatter(xy[skipped, 0], xy[skipped, 1], s=230, facecolors='none',
                   edgecolors=RED, linewidths=2, zorder=3)
        ax.scatter(xy[skipped, 0], xy[skipped, 1], marker='x', s=130,
                   color=RED, linewidths=2.4, zorder=4)
    for j in range(1, env.n_nodes):
        ax.text(xy[j, 0], xy[j, 1] - 0.065, str(j), ha='center', fontsize=8,
                color=(RED if j in skipped else col), weight='bold')
    ax.scatter(*xy[DEPOT], marker='s', s=230, color=INK, zorder=5)
    ax.text(xy[DEPOT, 0], xy[DEPOT, 1] + 0.055, 'depot', ha='center', fontsize=8.5)
    if stranded:
        ax.scatter(*xy[tour[-1]], s=420, facecolors='none', edgecolors=RED,
                   linewidths=2.6, zorder=6)
        ax.text(xy[tour[-1], 0], xy[tour[-1], 1] + 0.10, 'STUCK HERE\nno time to get back',
                ha='center', fontsize=9, color=RED, weight='bold')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.12, 1.16); ax.axis('off')
    ax.set_title(f'{title}\n{sub}', fontsize=11, loc='left')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    env = BudgetRoutingEnv(budget_mult=0.95, capacity=10, dist_scale=10,
                           allow_stranding=True, reward_shape='terminal')
    best_tour, best_served = optimal_plan(env)

    # a real seed that stranded itself
    from counterfactual_rl.agents.cvrp.dqn import CVRPDQN
    d = Path(args.runs_dir) / 'oa_per_s09'
    cfg = pickle.load(open(d / 'last.pkl', 'rb'))['config']
    agent = CVRPDQN(dict(cfg)); agent.load(str(d / 'last.pkl'))
    bad_tour, _ = agent.rollout_greedy()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    draw_map(ax, env, best_tour, False, '(a) The best anyone can do',
             f'{best_served} of {env.n_customers} served · '
             f'{env.tour_units(best_tour)} of {env.budget} units used')

    ax = fig.add_subplot(gs[0, 1])
    n_bad = len([p for p in bad_tour if p != DEPOT])
    draw_map(ax, env, bad_tour, True, '(b) A real trained policy that stranded',
             f'delivered to {n_bad} · never got home · SCORED 0')

    # (c) reward timeline
    ax = fig.add_subplot(gs[0, 2])
    for row, (lab, n_steps, payout, col) in enumerate((
            ('good run', len(best_tour) - 1, best_served, GREEN),
            ('stranded', len(bad_tour) - 1, 0, RED))):
        y = 1 - row
        for t in range(n_steps):
            last = (t == n_steps - 1)
            val = payout if last else 0
            c = col if (last and payout) else '#E8ECF1'
            ax.add_patch(plt.Rectangle((t, y - 0.28), 0.86, 0.56, color=c))
            ax.text(t + 0.43, y, str(val), ha='center', va='center', fontsize=10,
                    color=('white' if (last and payout) else '#7C8A9B'),
                    weight=('bold' if last else 'normal'))
        ax.text(-0.4, y, lab, ha='right', va='center', fontsize=10, weight='bold')
        ax.text(n_steps + 0.5, y, f'total {payout}', va='center', fontsize=10, color=col)
    ax.set_xlim(-3.4, 16); ax.set_ylim(-0.9, 1.9); ax.axis('off')
    ax.set_title('(c) What the network is told, move by move', fontsize=11, loc='left')
    ax.text(-3.3, -0.62, 'Zero all the way. One number at the end.\n'
                         'Strand the truck and everything delivered is wiped.',
            fontsize=9.5, color=INK)

    fig.suptitle('Routing-hard — 10 customers, a 33-unit shift, and the truck is allowed '
                 'to strand itself', fontsize=13, y=1.0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_routing_hard_explainer.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"wrote {p}")
    print(f"  best: {best_served} served, {env.tour_units(best_tour)}/{env.budget} units")
    print(f"  stranded seed: delivered {n_bad}, ended at node {bad_tour[-1]}, scored 0")


if __name__ == '__main__':
    main()
