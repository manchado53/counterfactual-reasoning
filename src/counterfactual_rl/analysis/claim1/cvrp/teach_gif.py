"""
Teaching animation: STATE, ACTION, REWARD — one idea at a time, big and slow.

Built for a talk. Each chapter holds on screen long enough to say a sentence over it:

    1. THE WORLD    what the problem is
    2. STATE        everything that describes "right now"
    3. ACTION       the choices available, and the blocked ones
    4. REWARD       what you get back
    5. REPEAT       the loop running at speed

Run:
    python -m counterfactual_rl.analysis.claim1.cvrp.teach_gif --ckpt <path/to/ckpt.pkl>
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
import jax.numpy as jnp

from counterfactual_rl.envs.cvrp import CVRPEnv, INSTANCES, DEPOT

TEAL = '#0E7C86'
TEAL_DK = '#0A5B63'
AMBER = '#B5641A'
RED = '#C1362F'
INK = '#15211F'
PAPER = '#F4F7F6'
LINE = '#C8D4D2'
GREY = '#8C9A97'
GREEN = '#1F7A4D'

HOLD = 26          # frames each chapter holds (~2.6s at 10fps)


def make_trace(env, agent):
    masks = np.asarray(env.action_masks)
    next_s = np.asarray(env.next_states)[:, :, 0]
    rew = np.asarray(env.rewards)[:, :, 0]
    s = env.start_states[0]
    out, served = [], set()
    for _ in range(4 * env.n_nodes):
        legal = np.flatnonzero(masks[s])
        if legal.size == 0:
            break
        q = np.asarray(agent._masked_q(agent.params, jnp.int32(s)))
        a = int(np.argmax(q))
        out.append(dict(state=s, node=int(env.state_current_np[s]),
                        load=int(env.state_cap[s]), legal=legal.copy(),
                        action=a, reward=float(rew[s, a]), served=set(served)))
        if a != DEPOT:
            served.add(a)
        s = int(next_s[s, a])
    return out


def render(env, trace, out_path):
    xy = env.node_xy
    # Teaching moment: a step where the capacity rule blocks a stop.
    focus = next((i for i, st in enumerate(trace)
                  if st['node'] != DEPOT and len(st['legal']) < env.n_nodes - len(st['served'])),
                 3)
    st = trace[focus]

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_position([0.02, 0.02, 0.66, 0.83])
    pad = 0.14
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    title = fig.text(0.03, 0.93, '', fontsize=27, fontweight='bold',
                     color=INK, family='monospace')
    subtitle = fig.text(0.03, 0.885, '', fontsize=13, color=TEAL_DK, family='monospace')

    # right-hand explanation panel
    panel = fig.text(0.70, 0.80, '', fontsize=12.5, color=INK, va='top',
                     family='monospace', linespacing=1.75)
    bignum = fig.text(0.845, 0.235, '', fontsize=42, fontweight='bold',
                      color=AMBER, ha='center', family='monospace')
    bignum_lbl = fig.text(0.845, 0.175, '', fontsize=11.5, color=GREY,
                          ha='center', va='top', family='monospace')

    base_route = np.array([xy[t['node']] for t in trace] + [xy[DEPOT]])
    faint, = ax.plot([], [], '-', color=LINE, lw=1.3, zorder=1)
    trail, = ax.plot([], [], '-', color=TEAL, lw=3.2, solid_capstyle='round', zorder=2)

    stop_sc = ax.scatter(xy[1:, 0], xy[1:, 1], s=np.asarray(env.demand[1:]) * 34 + 190,
                         c='white', edgecolors=INK, linewidths=2.0, zorder=4)
    demand_txt = [ax.text(*xy[i], f'{int(env.demand[i])}', ha='center', va='center',
                          fontsize=11, family='monospace', color=INK, zorder=5)
                  for i in range(1, env.n_nodes)]
    name_txt = [ax.annotate(f'C{i}', xy[i], textcoords='offset points', xytext=(0, -30),
                            ha='center', fontsize=9.5, color=GREY, family='monospace')
                for i in range(1, env.n_nodes)]
    ax.scatter(*xy[DEPOT], marker='s', s=430, c=INK, zorder=6)
    ax.annotate('DEPOT', xy[DEPOT], textcoords='offset points', xytext=(0, -32),
                ha='center', fontsize=10.5, fontweight='bold', color=INK, family='monospace')

    truck = ax.scatter([], [], marker='o', s=330, c=AMBER, edgecolors='white',
                       linewidths=2.6, zorder=9)
    you_lbl = ax.annotate('', (0, 0), textcoords='offset points', xytext=(0, 30),
                          ha='center', fontsize=12, fontweight='bold', color=AMBER,
                          family='monospace', zorder=10)

    arrows, crosses, dist_lbl = [], [], []

    def clear_overlay():
        for c in arrows + crosses + dist_lbl:
            c.remove()
        arrows.clear(); crosses.clear(); dist_lbl.clear()

    # gauge
    gx, gy, gw, gh = 0.705, 0.085, 0.26, 0.032
    gbg = plt.Rectangle((gx, gy), gw, gh, transform=fig.transFigure, facecolor='white',
                        edgecolor=LINE, zorder=3)
    gfg = plt.Rectangle((gx, gy), gw, gh, transform=fig.transFigure, facecolor=TEAL,
                        edgecolor='none', zorder=4)
    fig.patches.extend([gbg, gfg])
    gtxt = fig.text(gx, gy + gh + 0.018, '', fontsize=11.5, color=INK, family='monospace')
    for art in (gbg, gfg):
        art.set_visible(False)

    # ── chapter plan: (chapter, index, how long to hold it in ms) ────────
    # One frame per scene with an explicit duration, rather than many identical
    # frames — GIF optimizers dedupe repeated frames, which silently destroys the
    # hold time and makes each chapter flash past.
    plan = [('world', 0, 3200), ('state', 0, 4000),
            ('action', 0, 4500), ('reward', 0, 4000)]
    plan += [('run', k, 420) for k in range(len(trace))]
    plan += [('end', 0, 4500)]

    trail_pts = []

    def set_stops(served, highlight=None, dim=False):
        cols, edges, lws = [], [], []
        for i in range(1, env.n_nodes):
            if i in served:
                cols.append(TEAL); edges.append(TEAL_DK); lws.append(2.0)
            elif highlight is not None and i == highlight:
                cols.append('white'); edges.append(AMBER); lws.append(3.4)
            else:
                cols.append('white'); edges.append(GREY if dim else INK); lws.append(2.0)
        stop_sc.set_facecolors(cols); stop_sc.set_edgecolors(edges)
        stop_sc.set_linewidths(lws)

    def update(idx):
        ch, f, _dur = plan[idx]
        clear_overlay()
        for art in (gbg, gfg):
            art.set_visible(ch in ('state', 'action', 'reward', 'run', 'end'))
        bignum.set_text(''); bignum_lbl.set_text(''); gtxt.set_text('')
        you_lbl.set_text('')

        # ---------- CHAPTER 1 : the world ----------
        if ch == 'world':
            title.set_text('THE WORLD')
            subtitle.set_text('one truck  ·  ten stops  ·  a load limit')
            faint.set_data([], []); trail.set_data([], [])
            set_stops(set())
            truck.set_offsets([xy[DEPOT]])
            panel.set_text(
                'Truck starts at the DEPOT.\n\n'
                'Each circle is a stop.\n'
                'The number inside is\n'
                'how much they want.\n\n'
                'Truck holds 10.\n'
                'Stops want 24 total.\n\n'
                '-> it CANNOT do it\n'
                '   in one trip.')
            return ()

        # ---------- CHAPTER 2 : state ----------
        if ch == 'state':
            title.set_text('STATE')
            subtitle.set_text('everything that describes RIGHT NOW')
            set_stops(st['served'])
            truck.set_offsets([xy[st['node']]])
            you_lbl.set_position((0, 34))
            you_lbl.set_text('YOU ARE HERE')
            you_lbl.xy = xy[st['node']]
            trail.set_data([], []); faint.set_data([], [])
            panel.set_text(
                'Three things:\n\n'
                f'1  WHERE I AM\n     at C{st["node"]}\n\n'
                f'2  WHO IS DONE\n     {len(st["served"])} of 10 filled in\n\n'
                f'3  HOW FULL I AM\n     {st["load"]} of 10 left')
            gfg.set_width(gw * st['load'] / env.capacity)
            gtxt.set_text(f"truck load  {st['load']}/10")
            return ()

        # ---------- CHAPTER 3 : action ----------
        if ch == 'action':
            title.set_text('ACTION')
            subtitle.set_text('pick ONE stop to drive to next')
            set_stops(st['served'])
            here = xy[st['node']]
            truck.set_offsets([here])
            gfg.set_width(gw * st['load'] / env.capacity)
            gtxt.set_text(f"truck load  {st['load']}/10")

            blocked_full, blocked_cap = [], []
            for j in range(1, env.n_nodes):
                if j in st['legal']:
                    a = ax.annotate('', xy[j], xytext=here, zorder=3,
                                    arrowprops=dict(arrowstyle='-|>', color=TEAL,
                                                    lw=2.0, alpha=.85,
                                                    shrinkA=16, shrinkB=18))
                    arrows.append(a)
                else:
                    c = ax.scatter(*xy[j], marker='x', s=260, c=RED,
                                   linewidths=3.4, zorder=8)
                    crosses.append(c)
                    (blocked_full if j in st['served'] else blocked_cap).append(j)
            if DEPOT in st['legal']:
                a = ax.annotate('', xy[DEPOT], xytext=here, zorder=3,
                                arrowprops=dict(arrowstyle='-|>', color=TEAL, lw=2.0,
                                                alpha=.85, shrinkA=16, shrinkB=20))
                arrows.append(a)

            cap_names = ', '.join(f'C{j}' for j in blocked_cap) or 'none'
            panel.set_text(
                f'{len(st["legal"])} arrows = legal moves\n'
                '(one is "go reload")\n\n'
                'RED X = not allowed:\n\n'
                f'  already served\n     {len(blocked_full)} stops\n\n'
                f'  WILL NOT FIT\n     {cap_names}\n'
                f'     wants more than {st["load"]}')
            return ()

        # ---------- CHAPTER 4 : reward ----------
        if ch == 'reward':
            title.set_text('REWARD')
            subtitle.set_text('you pay for the distance you drive')
            a_ = st['action']
            set_stops(st['served'], highlight=a_ if a_ != DEPOT else None)
            here, there = xy[st['node']], xy[a_]
            truck.set_offsets([here])
            arrows.append(ax.annotate('', there, xytext=here, zorder=3,
                                      arrowprops=dict(arrowstyle='-|>', color=AMBER,
                                                      lw=4.0, shrinkA=16, shrinkB=18)))
            mid = (here + there) / 2
            dist_lbl.append(ax.text(mid[0], mid[1] + .035,
                                    f'{abs(st["reward"]):.2f} of driving',
                                    ha='center', fontsize=12, color=AMBER,
                                    fontweight='bold', family='monospace',
                                    bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                              ec=AMBER, alpha=.95), zorder=11))
            gfg.set_width(gw * st['load'] / env.capacity)
            gtxt.set_text(f"truck load  {st['load']}/10")
            panel.set_text(
                'Drive there ->\n'
                'you get a NEGATIVE\n'
                'reward, the size of\n'
                'the distance.\n\n'
                'No prize for finishing.\n\n'
                'So "most reward"\n'
                'means "least driving".')
            bignum.set_text(f'{st["reward"]:+.2f}')
            bignum_lbl.set_text('the reward')
            return ()

        # ---------- CHAPTER 5 : run it ----------
        if ch in ('run', 'end'):
            k = f if ch == 'run' else len(trace) - 1
            s_ = trace[k]
            title.set_text('REPEAT')
            subtitle.set_text('state -> action -> reward, thirteen times')
            shown = set(s_['served']) | ({s_['action']} if s_['action'] != DEPOT else set())
            set_stops(shown)
            here = xy[s_['node']]; there = xy[s_['action']]
            truck.set_offsets([there])
            if not trail_pts or trail_pts[-1] is not here:
                trail_pts.append(here)
            trail_pts.append(there)
            arr = np.array(trail_pts)
            trail.set_data(arr[:, 0], arr[:, 1])
            gfg.set_width(gw * max(s_['load'], 0) / env.capacity)
            gtxt.set_text(f"truck load  {s_['load']}/10")
            done_dist = sum(abs(t['reward']) for t in trace[:k + 1])
            panel.set_text(
                f'decision {k + 1} of {len(trace)}\n\n'
                f'served   {len(shown)} of 10\n'
                f'driven   {done_dist:.2f}')
            if ch == 'end':
                title.set_text('DONE')
                subtitle.set_text('every stop served, truck back home')
                bignum.set_text(f'{sum(abs(t["reward"]) for t in trace):.2f}')
                bignum_lbl.set_text('total driving\n= the BEST possible')
                bignum.set_color(GREEN)
            return ()
        return ()

    # Render each scene once, keeping its own on-screen duration.
    import io
    from PIL import Image

    images, durations = [], []
    for i in range(len(plan)):
        update(i)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=78, facecolor=PAPER)
        buf.seek(0)
        images.append(Image.open(buf).convert('RGB').convert(
            'P', palette=Image.ADAPTIVE, colors=96))
        durations.append(plan[i][2])
    plt.close(fig)

    images[0].save(out_path, save_all=True, append_images=images[1:], loop=0,
                   duration=durations, optimize=False, disposal=2)
    total_s = sum(durations) / 1000
    print(f'GIF saved -> {out_path}  ({len(images)} scenes, {total_s:.0f}s, '
          f'{Path(out_path).stat().st_size / 1024:.0f} KB)')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--instance', default='default', choices=sorted(INSTANCES))
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args(argv)

    from counterfactual_rl.agents.cvrp.dqn import CVRPDQN
    from counterfactual_rl.agents.cvrp.config import DEFAULT_CONFIG

    spec = INSTANCES[args.instance]
    cfg = DEFAULT_CONFIG.copy()
    cfg.update({'instance': args.instance, 'capacity': spec['capacity'], 'travel_noise': 0.0})
    agent = CVRPDQN(cfg)
    agent.load(args.ckpt)

    trace = make_trace(agent.env, agent)
    print(f'trace: {len(trace)} decisions')
    out_dir = Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'cvrp'
    out_dir.mkdir(parents=True, exist_ok=True)
    render(agent.env, trace, str(args.out or out_dir / 'cvrp_teach_sar.gif'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
