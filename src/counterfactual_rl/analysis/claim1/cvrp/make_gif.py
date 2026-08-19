"""
Animate a delivery plan in the routing env — a GIF for talks/slides.

Shows the truck driving its route: stops light up as they are served, the load gauge
drains and refills at each depot reload, and the running distance ticks up.

Run:
    # the exact optimal plan (no checkpoint needed)
    python -m counterfactual_rl.analysis.claim1.cvrp.make_gif

    # a trained policy's plan, side-by-side comparison against optimal
    python -m counterfactual_rl.analysis.claim1.cvrp.make_gif --ckpt <path/to/best.pkl>
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from counterfactual_rl.envs.cvrp import CVRPEnv, INSTANCES, DEPOT

# Palette (matches the environment brief)
TEAL = '#0E7C86'
TEAL_DK = '#0A5B63'
AMBER = '#B5641A'
INK = '#15211F'
PAPER = '#F4F7F6'
LINE = '#D2DBD9'
GREEN = '#1F7A4D'

FRAMES_PER_LEG = 8
HOLD_FRAMES = 10          # pause on the finished route before looping


def _shrink(path):
    """Palette-quantize the GIF in place — keeps it small enough to preview inline."""
    from PIL import Image
    im = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(im.convert('RGB').convert(
                'P', palette=Image.ADAPTIVE, colors=64))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   loop=0, duration=im.info.get('duration', 70), optimize=True)


def plan_from_policy(env, ckpt_path, config):
    """Greedy plan from a trained checkpoint."""
    from counterfactual_rl.agents.cvrp.dqn import CVRPDQN
    agent = CVRPDQN(config)
    agent.load(ckpt_path)
    path, _ = agent.rollout_greedy()
    return path


def build_frames(env, tour):
    """Interpolate truck positions along the tour; one entry per animation frame."""
    xy = env.node_xy
    frames = []
    served = set()
    load = env.capacity if env.is_capacitated else None
    dist_so_far = 0.0

    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        leg = float(env.dist[a, b])
        for f in range(FRAMES_PER_LEG):
            t = (f + 1) / FRAMES_PER_LEG
            pos = xy[a] * (1 - t) + xy[b] * t
            frames.append(dict(
                pos=pos, served=set(served), load=load,
                dist=dist_so_far + leg * t,
                leg_from=a, leg_to=b, reloading=False, done=False,
            ))
        dist_so_far += leg
        # arrival effects
        if b == DEPOT:
            if env.is_capacitated:
                load = env.capacity
            frames[-1]['reloading'] = True
        else:
            served.add(b)
            if env.is_capacitated:
                load -= int(env.demand[b])
        frames[-1]['served'] = set(served)
        frames[-1]['load'] = load

    for _ in range(HOLD_FRAMES):
        f = dict(frames[-1])
        f['done'] = True
        frames.append(f)
    return frames


def render(env, tour, out_path, title, subtitle=''):
    xy = env.node_xy
    frames = build_frames(env, tour)
    total = env.tour_length(tour)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    pad = 0.12
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad + 0.10)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE)

    # static: faint full route
    route = np.array([xy[i] for i in tour])
    ax.plot(route[:, 0], route[:, 1], '-', color=LINE, lw=1.4, zorder=1)

    # stops
    demand = env.demand
    stop_sc = ax.scatter(xy[1:, 0], xy[1:, 1],
                         s=np.asarray(demand[1:]) * 30 + 90,
                         c='white', edgecolors=INK, linewidths=1.8, zorder=3)
    for i in range(1, env.n_nodes):
        ax.annotate(f'C{i}', xy[i], textcoords='offset points', xytext=(0, -27),
                    ha='center', fontsize=8.5, color=INK, family='monospace')
        if env.is_capacitated:
            ax.annotate(f'{int(demand[i])}', xy[i], ha='center', va='center',
                        fontsize=8.5, color=INK, family='monospace', zorder=4)

    ax.scatter(*xy[DEPOT], marker='s', s=300, c=INK, zorder=5)
    ax.annotate('DEPOT', xy[DEPOT], textcoords='offset points', xytext=(0, -24),
                ha='center', fontsize=9, fontweight='bold', color=INK, family='monospace')

    trail, = ax.plot([], [], '-', color=TEAL, lw=3.0, solid_capstyle='round', zorder=2)
    truck = ax.scatter([], [], marker='o', s=210, c=AMBER, edgecolors='white',
                       linewidths=2.0, zorder=6)

    ax.set_title(title, fontsize=13, fontweight='bold', color=INK, pad=16)
    sub = ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha='center',
                  fontsize=9.5, color=TEAL_DK, family='monospace')
    hud = ax.text(0.02, 0.975, '', transform=ax.transAxes, va='top', ha='left',
                  fontsize=10, family='monospace', color=INK,
                  bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                            edgecolor=LINE, alpha=.92))
    flash = ax.text(0.5, 0.045, '', transform=ax.transAxes, ha='center',
                    fontsize=12, fontweight='bold', color=AMBER, family='monospace')

    # load gauge
    gx, gy, gw, gh = 0.62, 0.955, 0.34, 0.028
    gauge_bg = plt.Rectangle((gx, gy), gw, gh, transform=ax.transAxes,
                             facecolor='white', edgecolor=LINE, zorder=10)
    gauge_fg = plt.Rectangle((gx, gy), gw, gh, transform=ax.transAxes,
                             facecolor=TEAL, edgecolor='none', zorder=11)
    if env.is_capacitated:
        ax.add_patch(gauge_bg); ax.add_patch(gauge_fg)
    gauge_txt = ax.text(gx + gw, gy - 0.030, '', transform=ax.transAxes, ha='right',
                        va='top', fontsize=9, family='monospace', color=INK)

    trail_x, trail_y = [], []

    def update(k):
        fr = frames[k]
        trail_x.append(fr['pos'][0]); trail_y.append(fr['pos'][1])
        trail.set_data(trail_x, trail_y)
        truck.set_offsets([fr['pos']])

        cols = ['white'] * (env.n_nodes - 1)
        edges = [INK] * (env.n_nodes - 1)
        for s in fr['served']:
            cols[s - 1] = TEAL
            edges[s - 1] = TEAL_DK
        stop_sc.set_facecolors(cols)
        stop_sc.set_edgecolors(edges)

        n_served = len(fr['served'])
        hud.set_text(f"stops served  {n_served}/{env.n_customers}\n"
                     f"distance      {fr['dist']:.2f}")

        if env.is_capacitated:
            frac = max(0.0, fr['load'] / env.capacity)
            gauge_fg.set_width(gw * frac)
            gauge_fg.set_facecolor(AMBER if frac <= 0.25 else TEAL)
            gauge_txt.set_text(f"load {fr['load']}/{env.capacity}")

        if fr['done']:
            flash.set_text(f'PLAN COMPLETE  ·  total {total:.2f}')
            flash.set_color(GREEN)
        elif fr['reloading']:
            flash.set_text('RELOAD AT DEPOT')
            flash.set_color(AMBER)
        else:
            flash.set_text('')
        return trail, truck, stop_sc, hud, flash, gauge_fg, gauge_txt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=12), dpi=85,
              savefig_kwargs={'facecolor': PAPER})
    plt.close(fig)
    _shrink(out_path)
    size_kb = Path(out_path).stat().st_size / 1024
    print(f'GIF saved -> {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', default='default', choices=sorted(INSTANCES))
    ap.add_argument('--capacity', type=int, default=None,
                    help='load limit; -1 for TSP mode; omit for the instance default')
    ap.add_argument('--ckpt', default=None,
                    help='checkpoint to animate; omit to animate the exact optimal plan')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args(argv)

    spec = INSTANCES[args.instance]
    capacity = spec['capacity'] if args.capacity is None else (
        None if args.capacity < 0 else args.capacity)
    env = CVRPEnv(node_xy=spec['xy'], demand=spec['demand'], capacity=capacity)

    from .oracle import optimal_tour
    opt_tour, opt_len = optimal_tour(env, gamma=1.0)

    out_dir = (Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'cvrp')
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt:
        from counterfactual_rl.agents.cvrp.config import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG.copy()
        cfg.update({'instance': args.instance, 'capacity': capacity, 'travel_noise': 0.0})
        tour = plan_from_policy(env, args.ckpt, cfg)
        length = env.tour_length(tour)
        title = 'Learned delivery plan (DQN)'
        sub = (f'{length:.2f} vs optimal {opt_len:.2f}  '
               f'({100*opt_len/length:.1f}% of optimal)')
        out = args.out or out_dir / 'cvrp_learned_plan.gif'
    else:
        tour = opt_tour
        loads = tour.count(DEPOT) - 1
        title = 'Capacitated Vehicle Routing — optimal plan'
        sub = (f'{env.n_customers} stops · capacity {env.capacity} · '
               f'{loads} loads · distance {opt_len:.2f}') if env.is_capacitated else \
              f'{env.n_customers} stops · TSP · distance {opt_len:.2f}'
        out = args.out or out_dir / 'cvrp_optimal_plan.gif'

    print(f'plan: {tour}')
    render(env, tour, str(out), title, sub)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
