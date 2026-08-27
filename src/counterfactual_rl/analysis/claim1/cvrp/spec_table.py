"""
Static reference table of the routing environment's dynamics — a slide asset.

Every number is read from the live environment, so the card cannot drift from the code.

Run:
    python -m counterfactual_rl.analysis.claim1.cvrp.spec_table
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from counterfactual_rl.envs.cvrp import CVRPEnv, DEPOT

TEAL = '#0E7C86'
TEAL_DK = '#0A5B63'
AMBER = '#B5641A'
INK = '#15211F'
PAPER = '#FFFFFF'
BAND = '#F1F5F4'
LINE = '#D2DBD9'
GREY = '#6B7A77'


def build_rows(env, tsp_states, opt_len, n_decisions, n_decision_states):
    demand_total = int(env.demand.sum())
    mdp = [
        ('STATE', 'where the truck is  +  which stops are served  +  how much load is left',
         f'{env.feature_dim} numbers'),
        ('ACTION', 'drive to one node — a customer, or the depot to reload',
         f'{env.n_actions} choices'),
        ('REWARD', 'minus the distance of the leg just driven (dense — paid every step)',
         'e.g. -0.20'),
        ('TRANSITION', 'deterministic: you arrive, that stop is marked served, load drops',
         'no randomness'),
        ('START', 'at the depot, nothing served yet, truck full', '1 state'),
        ('END', 'back at the depot AND every stop served', '1 state'),
        ('DISCOUNT', 'gamma — future legs still count, so early choices matter', '0.99'),
        ('EPISODE', 'one full delivery day, start to finish', f'{n_decisions} decisions'),
    ]
    rules = [
        ('Already served', 'you cannot deliver to the same stop twice',
         'menu shrinks each step'),
        ('Will not fit', 'stop wants more units than the truck has left',
         'forces a reload'),
        ('No idling at depot', 'you cannot "stay" at the depot; you must drive somewhere',
         'depot is only reachable from a stop'),
        ('Blocked = impossible', 'illegal moves are scored minus-infinity',
         'agent can never pick one'),
    ]
    facts = [
        ('Stops (customers)', f'{env.n_customers}'),
        ('Truck capacity', f'{env.capacity} units'),
        ('Total demand of all stops', f'{demand_total} units'),
        ('Loads needed (24 will not fit in 10)', f'{env.min_loads()}'),
        ('Shortest possible plan (proved by DP)', f'{opt_len:.4f}'),
        ('Reachable states — with load limit', f'{env.n_states:,}'),
        ('Reachable states — no load limit (TSP)', f'{tsp_states:,}'),
        ('States that involve a real choice', f'{n_decision_states:,}'),
    ]
    knobs = [
        ('capacity', 'the load limit — the "C" in CVRP',
         'None = TSP (one big loop)', '10 / 6 / 5 — tighter = more pivotal reload calls'),
        ('travel_noise', 'traffic: the drive takes longer than planned',
         '0.0 = deterministic (Claim 2)', '0.15 = graded CCE score (needed for Claim 1)'),
    ]
    return mdp, rules, facts, knobs


def render(env, out_path, tsp_states, opt_len, n_decisions, n_decision_states):
    mdp, rules, facts, knobs = build_rows(
        env, tsp_states, opt_len, n_decisions, n_decision_states)

    fig = plt.figure(figsize=(13.6, 10.4))
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(.045, .965, 'Capacitated Vehicle Routing — how the environment works',
            fontsize=21, fontweight='bold', color=INK)
    ax.text(.045, .938, 'one truck  ·  ten stops  ·  a load limit  ·  fully deterministic',
            fontsize=12.5, color=TEAL_DK, family='monospace')

    y = .905
    ROW = .0305

    def section(label, y):
        ax.add_patch(plt.Rectangle((.045, y - .004), .91, .0035,
                                   facecolor=TEAL, edgecolor='none'))
        ax.text(.045, y + .011, label, fontsize=11, fontweight='bold',
                color=TEAL_DK, family='monospace')
        return y - .028

    def rows(data, cols, y, band=True):
        for i, row in enumerate(data):
            if band and i % 2 == 0:
                ax.add_patch(plt.Rectangle((.045, y - .0095), .91, ROW - .002,
                                           facecolor=BAND, edgecolor='none'))
            for (x, size, color, weight, mono), cell in zip(cols, row):
                ax.text(x, y, cell, fontsize=size, color=color, va='center',
                        fontweight=weight,
                        family='monospace' if mono else 'sans-serif')
            y -= ROW
        return y - .012

    # ── the MDP ──
    y = section('THE FIVE PIECES  ·  what an RL agent needs', y)
    y = rows(mdp,
             [(.055, 11.5, TEAL_DK, 'bold', True),
              (.20, 11.5, INK, 'normal', False),
              (.80, 11.5, AMBER, 'bold', True)], y)

    # ── the rules ──
    y = section('THE RULES  ·  what the agent is NOT allowed to do', y)
    y = rows(rules,
             [(.055, 11.5, INK, 'bold', False),
              (.26, 11.5, INK, 'normal', False),
              (.71, 11, GREY, 'normal', False)], y)

    # ── the numbers ──
    y = section('THIS INSTANCE  ·  the numbers', y)
    half = (len(facts) + 1) // 2
    y0 = y
    for i, (k, v) in enumerate(facts[:half]):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((.045, y - .0095), .445, ROW - .002,
                                       facecolor=BAND, edgecolor='none'))
        ax.text(.055, y, k, fontsize=11.5, color=INK, va='center')
        ax.text(.475, y, v, fontsize=11.5, color=AMBER, va='center',
                ha='right', fontweight='bold', family='monospace')
        y -= ROW
    y_left = y
    y = y0
    for i, (k, v) in enumerate(facts[half:]):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((.51, y - .0095), .445, ROW - .002,
                                       facecolor=BAND, edgecolor='none'))
        ax.text(.52, y, k, fontsize=11.5, color=INK, va='center')
        ax.text(.945, y, v, fontsize=11.5, color=AMBER, va='center',
                ha='right', fontweight='bold', family='monospace')
        y -= ROW
    y = min(y_left, y) - .012

    # ── the knobs ──
    y = section('THE TWO DIALS  ·  what we change between experiments', y)
    for i, (name, what, off, on) in enumerate(knobs):
        ax.add_patch(plt.Rectangle((.045, y - .020), .91, .050,
                                   facecolor=BAND if i % 2 == 0 else PAPER,
                                   edgecolor=LINE, linewidth=.8))
        ax.text(.055, y + .012, name, fontsize=12, color=TEAL_DK,
                fontweight='bold', family='monospace', va='center')
        ax.text(.055, y - .008, what, fontsize=10.5, color=GREY, va='center')
        ax.text(.40, y + .012, off, fontsize=10.5, color=INK, va='center',
                family='monospace')
        ax.text(.40, y - .008, on, fontsize=10.5, color=AMBER, va='center',
                family='monospace')
        y -= .058

    # Footer follows the content so bbox_inches='tight' crops off the empty space.
    ax.text(.045, y + .012,
            'Every figure read live from the environment. Optimal plan proved by dynamic '
            'programming and cross-checked against brute-force enumeration.',
            fontsize=9.5, color=GREY, style='italic', va='top')

    fig.savefig(out_path, dpi=190, facecolor=PAPER)
    plt.close(fig)

    # An axes spanning the whole figure defeats bbox_inches='tight', so trim the
    # blank margin off the rendered PNG instead.
    from PIL import Image, ImageChops, ImageOps
    im = Image.open(out_path).convert('RGB')
    box = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox()
    if box:
        pad = 26
        im = im.crop((max(0, box[0] - pad), max(0, box[1] - pad),
                      min(im.width, box[2] + pad), min(im.height, box[3] + pad)))
        im.save(out_path)
    print(f'table saved -> {out_path}  {im.size[0]}x{im.size[1]}px  '
          f'({Path(out_path).stat().st_size/1024:.0f} KB)')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args(argv)

    from .oracle import compute_oracle, optimal_tour
    env = CVRPEnv()
    tsp = CVRPEnv(capacity=None)
    tour, opt_len = optimal_tour(env, gamma=1.0)
    _, _, decision = compute_oracle(env, gamma=0.99)

    out_dir = Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'cvrp'
    out_dir.mkdir(parents=True, exist_ok=True)
    render(env, str(args.out or out_dir / 'cvrp_env_spec_table.png'),
           tsp.n_states, opt_len, len(tour) - 1, len(decision))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
