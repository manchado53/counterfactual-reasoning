"""
Teaching animation: how the agent actually interacts with the routing environment.

Three panels, in sync:
    LEFT   the world      — the map, the truck, which stops are served
    TOP R  what it SEES   — the 22-number observation vector, grouped and labelled
    BOT R  what it DOES   — a Q-value per action, illegal ones struck out, choice highlighted

Every number shown is read live from the env and a trained checkpoint, so the animation
matches the real system rather than an illustration of it.

Run:
    python -m counterfactual_rl.analysis.claim1.cvrp.explain_gif --ckpt <path/to/ckpt.pkl>
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
import numpy as np
import jax.numpy as jnp

from counterfactual_rl.envs.cvrp import CVRPEnv, INSTANCES, DEPOT

TEAL = '#0E7C86'
TEAL_DK = '#0A5B63'
AMBER = '#B5641A'
INK = '#15211F'
PAPER = '#F4F7F6'
LINE = '#C8D4D2'
GREY = '#9AA8A5'

THINK_FRAMES = 9     # frames spent showing the decision
MOVE_FRAMES = 7      # frames spent driving to the chosen stop


def collect_trace(env, agent):
    """Replay the greedy policy, recording everything the agent saw and did."""
    masks = np.asarray(env.action_masks)
    next_s = np.asarray(env.next_states)[:, :, 0]
    rew = np.asarray(env.rewards)[:, :, 0]

    s = env.start_states[0]
    steps, served = [], set()
    for _ in range(4 * env.n_nodes):
        legal = np.flatnonzero(masks[s])
        if legal.size == 0:
            break
        q = np.asarray(agent._masked_q(agent.params, jnp.int32(s)))
        a = int(np.argmax(q))
        steps.append(dict(
            state=s, node=int(env.state_current_np[s]), load=int(env.state_cap[s]),
            obs=np.asarray(env.state_features[s]).copy(),
            legal=legal.copy(), q=q.copy(), action=a,
            reward=float(rew[s, a]), served=set(served),
        ))
        if a != DEPOT:
            served.add(a)
        s = int(next_s[s, a])
    return steps


def render(env, steps, out_path):
    xy = env.node_xy
    n_nodes, n_cust = env.n_nodes, env.n_customers

    fig = plt.figure(figsize=(11.4, 6.0))
    fig.patch.set_facecolor(PAPER)
    gs = GridSpec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[0.58, 1.0],
                  hspace=0.42, wspace=0.16,
                  left=0.03, right=0.975, top=0.86, bottom=0.09)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_obs = fig.add_subplot(gs[0, 1])
    ax_q = fig.add_subplot(gs[1, 1])

    fig.suptitle('How the agent interacts with the routing environment',
                 fontsize=14, fontweight='bold', color=INK, y=0.965)
    step_txt = fig.text(0.5, 0.915, '', ha='center', fontsize=10,
                        family='monospace', color=TEAL_DK)

    # ── map ──────────────────────────────────────────────────────────────
    ax_map.set_facecolor(PAPER)
    pad = 0.13
    ax_map.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax_map.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax_map.set_aspect('equal'); ax_map.set_xticks([]); ax_map.set_yticks([])
    for sp in ax_map.spines.values():
        sp.set_color(LINE)
    ax_map.set_title('THE WORLD', fontsize=9.5, color=GREY, family='monospace',
                     pad=8, loc='left')

    stop_sc = ax_map.scatter(xy[1:, 0], xy[1:, 1], s=np.asarray(env.demand[1:]) * 26 + 95,
                             c='white', edgecolors=INK, linewidths=1.7, zorder=3)
    for i in range(1, n_nodes):
        ax_map.annotate(f'C{i}', xy[i], textcoords='offset points', xytext=(0, -25),
                        ha='center', fontsize=8, color=INK, family='monospace')
        ax_map.annotate(f'{int(env.demand[i])}', xy[i], ha='center', va='center',
                        fontsize=8, color=INK, family='monospace', zorder=4)
    ax_map.scatter(*xy[DEPOT], marker='s', s=250, c=INK, zorder=5)
    ax_map.annotate('DEPOT', xy[DEPOT], textcoords='offset points', xytext=(0, -24),
                    ha='center', fontsize=8.5, fontweight='bold', color=INK,
                    family='monospace')
    trail, = ax_map.plot([], [], '-', color=TEAL, lw=2.6, solid_capstyle='round', zorder=2)
    truck = ax_map.scatter([], [], marker='o', s=190, c=AMBER, edgecolors='white',
                           linewidths=2.0, zorder=6)
    aim, = ax_map.plot([], [], '--', color=AMBER, lw=1.8, zorder=2, alpha=.9)
    hud = ax_map.text(0.02, 0.98, '', transform=ax_map.transAxes, va='top', fontsize=9,
                      family='monospace', color=INK,
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=LINE, alpha=.93))

    # ── observation strip ────────────────────────────────────────────────
    ax_obs.set_facecolor(PAPER)
    ax_obs.set_xlim(-0.5, 21.5); ax_obs.set_ylim(-1.55, 1.15)
    ax_obs.set_xticks([]); ax_obs.set_yticks([])
    for sp in ax_obs.spines.values():
        sp.set_visible(False)
    ax_obs.set_title('WHAT IT SEES   ·   22 numbers', fontsize=9.5, color=GREY,
                     family='monospace', pad=8, loc='left')

    cells = []
    for i in range(22):
        r = plt.Rectangle((i - 0.42, 0.0), 0.84, 0.78, facecolor='white',
                          edgecolor=LINE, lw=0.8)
        ax_obs.add_patch(r); cells.append(r)
    cell_txt = [ax_obs.text(i, 0.39, '', ha='center', va='center', fontsize=6.4,
                            family='monospace', color=INK) for i in range(22)]

    def group(x0, x1, label, y=-0.34):
        ax_obs.plot([x0 - 0.42, x1 + 0.42], [y, y], color=GREY, lw=1.1)
        ax_obs.text((x0 + x1) / 2, y - 0.30, label, ha='center', va='top',
                    fontsize=7.6, family='monospace', color=GREY)
    group(0, n_nodes - 1, 'where I am now\n(depot + 10 stops)')
    group(n_nodes, n_nodes + n_cust - 1, 'who I already served\n(1 = done)')
    group(21, 21, 'how full\nthe truck is')

    # ── Q-value bars ─────────────────────────────────────────────────────
    ax_q.set_facecolor(PAPER)
    ax_q.set_title('WHAT IT DOES   ·   score for each possible next stop',
                   fontsize=9.5, color=GREY, family='monospace', pad=8, loc='left')
    bars = ax_q.bar(np.arange(n_nodes), np.zeros(n_nodes), color=TEAL, width=0.66)
    ax_q.set_xticks(np.arange(n_nodes))
    ax_q.set_xticklabels(['DEPOT'] + [f'C{i}' for i in range(1, n_nodes)],
                         fontsize=7.6, family='monospace')
    ax_q.set_yticks([])
    ax_q.set_ylim(0, 1.30)
    for sp in ('top', 'right', 'left'):
        ax_q.spines[sp].set_visible(False)
    ax_q.spines['bottom'].set_color(LINE)
    bar_lbl = [ax_q.text(i, 0, '', ha='center', va='bottom', fontsize=6.8,
                         family='monospace', color=INK) for i in range(n_nodes)]
    verdict = ax_q.text(0.985, 0.965, '', transform=ax_q.transAxes, ha='right',
                        va='top', fontsize=10, family='monospace', fontweight='bold',
                        color=AMBER,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                  edgecolor=LINE, alpha=.93))

    # ── frame plan ───────────────────────────────────────────────────────
    plan = []
    for k, st in enumerate(steps):
        for f in range(THINK_FRAMES):
            plan.append((k, 'think', f))
        for f in range(MOVE_FRAMES):
            plan.append((k, 'move', f))
    plan += [(len(steps) - 1, 'done', f) for f in range(12)]

    trail_x, trail_y = [], []
    total_dist = [0.0]

    def update(idx):
        k, phase, f = plan[idx]
        st = steps[k]
        a = st['action']
        here, there = xy[st['node']], xy[a]

        # map
        if phase == 'think':
            pos = here
            aim.set_data([here[0], there[0]], [here[1], there[1]])
        else:
            t = (f + 1) / MOVE_FRAMES
            pos = here * (1 - t) + there * t
            aim.set_data([], [])
        trail_x.append(pos[0]); trail_y.append(pos[1])
        trail.set_data(trail_x, trail_y)
        truck.set_offsets([pos])

        shown = set(st['served']) | ({a} if (phase != 'think' and a != DEPOT) else set())
        cols = ['white'] * n_cust
        edges = [INK] * n_cust
        for s_ in shown:
            if s_ != DEPOT:
                cols[s_ - 1] = TEAL; edges[s_ - 1] = TEAL_DK
        stop_sc.set_facecolors(cols); stop_sc.set_edgecolors(edges)

        hud.set_text(f"served {len(shown)}/{n_cust}\nload   {st['load']}/{env.capacity}")
        where = 'DEPOT' if st['node'] == DEPOT else f"C{st['node']}"
        step_txt.set_text(f"decision {k + 1} of {len(steps)}   ·   standing at {where}")

        # observation
        obs = st['obs']
        for i, (r, t_) in enumerate(zip(cells, cell_txt)):
            v = float(obs[i])
            r.set_facecolor(TEAL if v > 0.99 else (AMBER if 0 < v < 0.99 else 'white'))
            t_.set_text(f'{v:.1f}' if (i == 21 and 0 < v < 1) else str(int(round(v))))
            t_.set_color('white' if v > 0.5 else INK)

        # Q-values: shift so the worst legal option sits at zero (bars are comparable)
        q, legal = st['q'], st['legal']
        lo, hi = q[legal].min(), q[legal].max()
        span = max(hi - lo, 1e-6)
        for i, b in enumerate(bars):
            if i in legal:
                b.set_height(0.15 + 0.85 * (q[i] - lo) / span)
                b.set_color(AMBER if (i == a and phase != 'think') or
                            (i == a and f > THINK_FRAMES // 2) else TEAL)
                b.set_hatch('')
                bar_lbl[i].set_text(f'{q[i]:.2f}')
                bar_lbl[i].set_position((i, b.get_height() + 0.02))
                bar_lbl[i].set_color(INK)
            else:
                b.set_height(0.05)
                b.set_color('#E3E9E8'); b.set_hatch('///')
                bar_lbl[i].set_text('x')
                bar_lbl[i].set_position((i, 0.07))
                bar_lbl[i].set_color(GREY)

        name = 'DEPOT (reload)' if a == DEPOT else f'C{a}'
        if phase == 'think' and f <= THINK_FRAMES // 2:
            verdict.set_text('picking the highest legal score...')
            verdict.set_color(GREY)
        elif phase == 'done':
            verdict.set_text('ALL STOPS SERVED — back at depot')
            verdict.set_color('#1F7A4D')
        else:
            verdict.set_text(f'GO TO {name}   reward {st["reward"]:+.2f}')
            verdict.set_color(AMBER)
        return ()

    anim = FuncAnimation(fig, update, frames=len(plan), interval=95, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=10), dpi=80,
              savefig_kwargs={'facecolor': PAPER})
    plt.close(fig)

    from PIL import Image
    im = Image.open(out_path)
    frames = []
    try:
        while True:
            frames.append(im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=64))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0,
                   duration=95, optimize=True)
    print(f'GIF saved -> {out_path}  ({len(plan)} frames, '
          f'{Path(out_path).stat().st_size / 1024:.0f} KB)')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', default='default', choices=sorted(INSTANCES))
    ap.add_argument('--capacity', type=int, default=None)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args(argv)

    from counterfactual_rl.agents.cvrp.dqn import CVRPDQN
    from counterfactual_rl.agents.cvrp.config import DEFAULT_CONFIG

    spec = INSTANCES[args.instance]
    capacity = spec['capacity'] if args.capacity is None else (
        None if args.capacity < 0 else args.capacity)
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({'instance': args.instance, 'capacity': capacity, 'travel_noise': 0.0})

    agent = CVRPDQN(cfg)
    agent.load(args.ckpt)
    env = agent.env

    steps = collect_trace(env, agent)
    print(f'trace: {len(steps)} decisions')

    out_dir = Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'cvrp'
    out_dir.mkdir(parents=True, exist_ok=True)
    render(env, steps, str(args.out or out_dir / 'cvrp_how_it_works.gif'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
