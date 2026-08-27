"""
What "the map" actually is in our setup, and how it differs from the routing RL literature.

  (a) what we do: ONE fixed map, identical every episode, same depot start
  (b) the four fixed maps we have tried, all separate experiments -- never mixed
  (c) what Kool/Nazari do: a NEW random map every episode
  (d) the trade, stated plainly

Run:
    python -m counterfactual_rl.analysis.claim2.instance_setup_explainer --out <dir>
"""

import argparse
from pathlib import Path

import numpy as np

from counterfactual_rl.envs.routing_budget import INSTANCES, DEPOT

INK, BLUE, GREEN, RED, AMBER = '#122236', '#1B5FA8', '#2C6E49', '#9E2F27', '#B4600F'


def mini_map(ax, xy, title=None, colour=BLUE, star_depot=True, s=44):
    ax.scatter(xy[1:, 0], xy[1:, 1], s=s, color=colour, zorder=3)
    if star_depot:
        ax.scatter(*xy[DEPOT], marker='s', s=s * 1.7, color=INK, zorder=4)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#C9D2DC')
    if title:
        ax.set_title(title, fontsize=9)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.5, 7.4))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.75], hspace=0.55, wspace=0.35)

    default_xy = INSTANCES['default']['xy']

    # ---- row 1: what WE do — same map every episode ------------------------
    fig.text(0.012, 0.955, 'WHAT WE DO NOW  —  one fixed map, reused every single episode',
             fontsize=12, weight='bold', color=INK)
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        mini_map(ax, default_xy, f'episode {i + 1}', BLUE)
    ax = fig.add_subplot(gs[0, 5]); ax.axis('off')
    ax.text(0.0, 0.55, '…for all\n5,500 episodes\n\nidentical stops\nidentical depot\nidentical demands',
            fontsize=9.5, color=BLUE, va='center')

    # ---- row 2: what the literature does — new map every episode -----------
    fig.text(0.012, 0.635, 'WHAT THE ROUTING RL LITERATURE DOES  —  a NEW random map every episode',
             fontsize=12, weight='bold', color=INK)
    rng = np.random.default_rng(3)
    for i in range(5):
        ax = fig.add_subplot(gs[1, i])
        xy = np.vstack([[[0.5, 0.5]], rng.random((10, 2)) * 0.9 + 0.05]).astype(np.float32)
        mini_map(ax, xy, f'episode {i + 1}', GREEN)
    ax = fig.add_subplot(gs[1, 5]); ax.axis('off')
    ax.text(0.0, 0.55, '…never the same\nmap twice\n\nthe agent must\nLEARN A RULE,\nnot memorise\none answer',
            fontsize=9.5, color=GREEN, va='center')

    # ---- row 3: the four fixed maps we have, + the trade -------------------
    fig.text(0.012, 0.30, 'THE FOUR MAPS WE HAVE TRIED  —  each a separate experiment, never mixed',
             fontsize=12, weight='bold', color=INK)
    for i, name in enumerate(('default', 'clustered', 'hub_outliers', 'two_lobes')):
        ax = fig.add_subplot(gs[2, i])
        mini_map(ax, INSTANCES[name]['xy'], name, AMBER, s=36)

    ax = fig.add_subplot(gs[2, 4:]); ax.axis('off')
    ax.text(0.0, 1.0,
            'THE TRADE\n\n'
            'fixed map    → every state can be enumerated\n'
            '             → EXACT oracle → Claim 1 works\n'
            '             → but the agent memorises one answer\n\n'
            'random maps  → real generalisation, matches the field\n'
            '             → but no exact per-state ground truth\n'
            '             → and our tabular env cannot do it',
            fontsize=9.5, color=INK, va='top', family='monospace')

    fig.suptitle('Right now the agent solves the SAME delivery problem 5,500 times — '
                 'it is memorising one map, not learning to route',
                 fontsize=13, y=1.005)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / 'fig_instance_setup.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"wrote {p}")


if __name__ == '__main__':
    main()
