"""
Animate trained budget-routing policies side by side, against the exact optimum.

Each panel replays one arm's GREEDY policy on the same instance: the truck drives, the
fuel bar fills, customers light up as they are served. The leftmost panel is the exact
oracle plan, so you can see what was achievable.

The point of the side-by-side is the question the summary tables cannot answer: when the
budget runs out, WHICH customers does each policy give up on, and does CCE make a visibly
different call than PER?

Usage:
    python -m counterfactual_rl.analysis.claim1.cvrp.budget_gif \
        --runs-dir src/counterfactual_rl/agents/cvrp/runs \
        --budget 0.90 --capacity 10 --seed 0 \
        --arms per ccemul cceadd --out docs/figures/real/claim2/cvrp_budget_policies.gif
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from counterfactual_rl.envs.routing_budget import DEPOT
from counterfactual_rl.analysis.claim1.cvrp.budget_oracle import optimal_plan, optimal_served

FRAMES_PER_LEG = 5
HOLD_FRAMES = 14
FPS = 14          # 71 ms/frame -> one leg takes ~0.36 s

ARM_LABEL = {
    'uniform': 'DQN-Uniform', 'per': 'DQN+PER', 'cceonly': 'DQN+CCE-only',
    'cceadd': 'CCE+TD (add)', 'ccemul': 'CCE+TD (mul)',
}


def load_policy_tour(run_dir: Path):
    """(node path, config) for a run's final greedy policy."""
    ckpt = run_dir / 'last.pkl'
    with open(ckpt, 'rb') as f:
        config = pickle.load(f)['config']

    from counterfactual_rl.agents.cvrp.dqn import CVRPDQN
    agent = CVRPDQN(dict(config))
    agent.load(str(ckpt))
    path, _ = agent.rollout_greedy()
    return path, config, agent.env


def tour_frames(env, tour):
    """One entry per animation frame: truck position, served set, fuel spent."""
    xy, D = env.node_xy, env.D
    out, served, spent = [], set(), 0
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        leg = int(D[a, b])
        for f in range(FRAMES_PER_LEG):
            t = (f + 1) / FRAMES_PER_LEG
            out.append(dict(pos=xy[a] * (1 - t) + xy[b] * t,
                            served=set(served), spent=spent + leg * t,
                            legs=tour[:i + 1] + [b], frac=t))
        spent += leg
        if b != DEPOT:
            served.add(b)
        out[-1]['served'] = set(served)
        out[-1]['spent'] = spent
    if not out:
        out = [dict(pos=xy[DEPOT], served=set(), spent=0, legs=[DEPOT], frac=0.0)]
    return out


def render(env, panels, out_path, suptitle, ncols=None):
    """panels = [(label, tour, frames)]; all animate together, short ones hold at the end."""
    n = len(panels)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    n_frames = max(len(f) for _, _, f in panels) + HOLD_FRAMES
    xy = env.node_xy
    B = env.budget

    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 5.5 * nrows),
                                  squeeze=False)
    axes = [ax for row in axes_grid for ax in row]
    for ax in axes[n:]:
        ax.set_axis_off()
    artists = []

    for ax, (label, tour, frames) in zip(axes, panels):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.30, 1.12)
        ax.set_aspect('equal')
        ax.axis('off')
        # customers: hollow until served
        ax.scatter(xy[1:, 0], xy[1:, 1], s=190, facecolors='none',
                   edgecolors='#bbbbbb', linewidths=1.6, zorder=2)
        for j in range(1, env.n_nodes):
            ax.text(xy[j, 0], xy[j, 1], str(j), ha='center', va='center',
                    fontsize=7, color='#888888', zorder=3)
        ax.scatter([xy[DEPOT, 0]], [xy[DEPOT, 1]], marker='s', s=210,
                   color='#222222', zorder=4)
        ax.text(xy[DEPOT, 0], xy[DEPOT, 1] + 0.075, 'depot', ha='center',
                fontsize=8, color='#222222')

        trail, = ax.plot([], [], '-', color='#1f77b4', lw=1.8, alpha=.85, zorder=1)
        truck, = ax.plot([], [], 'o', color='#d62728', ms=11, zorder=6)
        done = ax.scatter([], [], s=190, color='#2ca02c', zorder=5)
        # fuel gauge under the map
        ax.add_patch(plt.Rectangle((0.02, -0.13), 0.96, 0.055, fill=False,
                                   edgecolor='#666666', lw=1.1))
        gauge = ax.add_patch(plt.Rectangle((0.02, -0.13), 0.0, 0.055,
                                           color='#4c9f70', lw=0))
        cap = ax.text(0.5, 1.045, '', ha='center', fontsize=9.5, weight='bold')
        ax.set_title(label, fontsize=11)
        artists.append(dict(trail=trail, truck=truck, done=done, gauge=gauge,
                            cap=cap, frames=frames, tour=tour))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    key = [
        Line2D([], [], marker='s', color='none', markerfacecolor='#222222',
               markersize=10, label='depot (start / reload / finish)'),
        Line2D([], [], marker='o', color='none', markeredgecolor='#bbbbbb',
               markerfacecolor='none', markersize=11, label='customer not yet served'),
        Line2D([], [], marker='o', color='none', markerfacecolor='#2ca02c',
               markersize=11, label='customer served'),
        Line2D([], [], marker='o', color='none', markerfacecolor='#d62728',
               markersize=10, label='the vehicle'),
        Line2D([], [], color='#1f77b4', lw=2, label='route driven so far'),
        Patch(facecolor='#4c9f70', edgecolor='#666666', label='fuel used / budget B'),
    ]
    fig.legend(handles=key, loc='lower center', ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0.11 if nrows == 1 else 0.06, 1, 0.95])
    if nrows > 1:
        fig.subplots_adjust(hspace=0.30)

    def update(k):
        out = []
        for a in artists:
            fr = a['frames'][min(k, len(a['frames']) - 1)]
            pts = xy[fr['legs']]
            head = np.vstack([pts, fr['pos'][None, :]])
            a['trail'].set_data(head[:, 0], head[:, 1])
            a['truck'].set_data([fr['pos'][0]], [fr['pos'][1]])
            srv = sorted(fr['served'])
            a['done'].set_offsets(xy[srv] if srv else np.empty((0, 2)))
            used = min(1.0, fr['spent'] / B)
            a['gauge'].set_width(0.96 * used)
            a['gauge'].set_color('#4c9f70' if used < 0.8 else '#e08a1e' if used < 0.97 else '#c0392b')
            a['cap'].set_text(
                f"served {len(fr['served'])}/{env.n_customers}"
                f"     fuel {fr['spent']:.0f}/{B} ({100 * used:.0f}%)")
            out += [a['trail'], a['truck'], a['done'], a['gauge'], a['cap']]
        return out

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=80)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)

    from counterfactual_rl.analysis.claim1.cvrp.make_gif import _shrink
    # explicit duration: never let the end-hold frame set the speed of the whole loop
    _shrink(str(out_path), duration=int(round(1000 / FPS)))
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def progression_panels(env, run_dir: Path, stages):
    """One panel per training stage, replaying that checkpoint's greedy policy."""
    from counterfactual_rl.agents.cvrp.dqn import CVRPDQN
    ckpts = sorted((run_dir / 'checkpoints').glob('ckpt_*.pkl'))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run_dir}/checkpoints")
    pick = [ckpts[min(len(ckpts) - 1, round(i * (len(ckpts) - 1) / max(1, stages - 1)))]
            for i in range(stages)]

    out = []
    for c in pick:
        with open(c, 'rb') as f:
            config = pickle.load(f)['config']
        agent = CVRPDQN(dict(config))
        agent.load(str(c))
        tour, _ = agent.rollout_greedy()
        ep = int(c.stem.split('_')[1])
        served = len([p for p in tour if p != DEPOT])
        out.append((f"episode {ep} — served {served}", tour, tour_frames(env, tour)))
    return out


def render_static(env, panels, out_path, suptitle):
    """
    One still frame per policy: the whole route at once, with the abandoned customers
    marked. For a TIE this reads far better than an animation — three trucks driving
    different loops to the same score is motion that hides the answer.
    """
    n = len(panels)
    xy = env.node_xy
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 5.0), squeeze=False)
    allc = set(range(1, env.n_nodes))

    for ax, (label, tour, _) in zip(axes[0], panels):
        served = [p for p in tour if p != DEPOT]
        skipped = sorted(allc - set(served))
        ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.08, 1.12)
        ax.set_aspect('equal'); ax.axis('off')

        # the route, drawn as arrows in travel order
        for i in range(len(tour) - 1):
            a, b = xy[tour[i]], xy[tour[i + 1]]
            ax.annotate('', xy=b, xytext=a,
                        arrowprops=dict(arrowstyle='-|>', color='#1f77b4',
                                        lw=1.9, alpha=.85, shrinkA=11, shrinkB=11))
        ax.scatter(xy[served, 0], xy[served, 1], s=260, color='#2ca02c', zorder=3)
        if skipped:
            ax.scatter(xy[skipped, 0], xy[skipped, 1], s=260, facecolors='none',
                       edgecolors='#c0392b', linewidths=2.0, zorder=3)
            ax.scatter(xy[skipped, 0], xy[skipped, 1], marker='x', s=150,
                       color='#c0392b', linewidths=2.4, zorder=4)
        for j in range(1, env.n_nodes):
            ax.text(xy[j, 0], xy[j, 1] - 0.062, str(j), ha='center', fontsize=8.5,
                    color=('#c0392b' if j in skipped else '#2ca02c'), weight='bold')
        ax.scatter([xy[DEPOT, 0]], [xy[DEPOT, 1]], marker='s', s=250,
                   color='#222222', zorder=5)
        ax.text(xy[DEPOT, 0], xy[DEPOT, 1] + 0.06, 'depot', ha='center', fontsize=8.5)

        ax.set_title(f"{label}\nserved {len(served)}   ·   gave up on "
                     f"{', '.join(str(k) for k in skipped) if skipped else 'nobody'}"
                     f"\nfuel {env.tour_units(tour)}/{env.budget}", fontsize=10.5)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], marker='s', color='none', markerfacecolor='#222222',
               markersize=10, label='depot'),
        Line2D([], [], marker='o', color='none', markerfacecolor='#2ca02c',
               markersize=11, label='served'),
        Line2D([], [], marker='x', color='#c0392b', lw=0, markersize=10,
               markeredgewidth=2.4, label='abandoned (no fuel left for it)'),
        Line2D([], [], color='#1f77b4', lw=2, label='route, in travel order'),
    ], loc='lower center', ncol=4, frameon=False, fontsize=10)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--prefix', default='bdm', help='run-name prefix (bdm = mean aggregation)')
    ap.add_argument('--arms', nargs='+', default=['per', 'ccemul'])
    ap.add_argument('--budget', type=float, required=True)
    ap.add_argument('--capacity', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--no-oracle', action='store_true')
    ap.add_argument('--static', action='store_true',
                    help='write a still PNG comparing whole routes instead of an animation')
    ap.add_argument('--runs', nargs='+', default=None, metavar='RUNDIR',
                    help='explicit run directory names to replay side by side, instead of '
                         'building names from --arms/--budget/--seed. Use this to show several '
                         'seeds of the SAME arm that ended differently.')
    ap.add_argument('--progression', nargs='+', default=None, metavar='RUN',
                    help='run dir names to show as a training progression (one row each)')
    ap.add_argument('--stages', type=int, default=4)
    args = ap.parse_args(argv)

    if args.progression:
        rows, env = [], None
        for name in args.progression:
            d = runs_dir_arg = Path(args.runs_dir) / name
            _, _, env = load_policy_tour(d)
            rows.append((name, d))
        panels = []
        for name, d in rows:
            for lbl, tour, fr in progression_panels(env, d, args.stages):
                panels.append((f"{name.replace('gifck_', '')} · {lbl}", tour, fr))
        opt_n = optimal_served(env)
        sup = (f"Policy improving during training · {env.n_customers} customers · "
               f"capacity {env.capacity} · B={env.budget}u · optimal = {opt_n} served")
        render(env, panels, args.out, sup, ncols=args.stages)
        return

    runs = Path(args.runs_dir)
    panels, env = [], None

    if args.runs:
        # Explicit run dirs: one panel each, labelled by how the run actually ended.
        built = []
        for name in args.runs:
            d = runs / name
            if not (d / 'last.pkl').exists():
                print(f"  skip {name}: no checkpoint")
                continue
            tour, _, env = load_policy_tour(d)
            served = len([p for p in tour if p != DEPOT])
            stranded = tour[-1] != DEPOT
            if stranded:
                lab = f"{name.split('_')[-1]}: STRANDED\nscored 0"
            else:
                lab = f"{name.split('_')[-1]}: served {served}\nscored {served}/{optimal_served(env)}"
            built.append((lab, tour, tour_frames(env, tour)))
        if env is None:
            raise SystemExit('none of the named runs had a checkpoint')
        sup = (f"Same method, same map, {len(built)} different seeds — "
               f"routing does not end in just two ways")
        (render_static if args.static else render)(env, built, args.out, sup)
        return

    for arm in args.arms:
        name = f"{args.prefix}_{arm}_b{int(round(args.budget * 100)):03d}_c{args.capacity}_s{args.seed}"
        d = runs / name
        if not (d / 'last.pkl').exists():
            print(f"  skip {name}: no checkpoint")
            continue
        tour, _, env = load_policy_tour(d)
        panels.append((arm, tour, None))

    if env is None:
        raise SystemExit("no runs found for that cell")

    opt_n = optimal_served(env)
    built = []
    if not args.no_oracle:
        opt_tour, _ = optimal_plan(env)
        built.append((f"ORACLE (optimal, {opt_n})", opt_tour, tour_frames(env, opt_tour)))
    for arm, tour, _ in panels:
        served = len([p for p in tour if p != DEPOT])
        built.append((f"{ARM_LABEL.get(arm, arm)}  —  {served}/{opt_n}",
                      tour, tour_frames(env, tour)))

    sup = (f"Tank holds {env.budget} units — only {opt_n} of {env.n_customers} stops can be "
           f"reached. Which {env.n_customers - opt_n} does each method give up on?")
    if args.static:
        render_static(env, built, args.out, sup)
    else:
        render(env, built, args.out, sup)


if __name__ == '__main__':
    main()
