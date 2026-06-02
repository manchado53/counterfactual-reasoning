"""
Generate presentation diagrams for CCE paper.

Saves to docs/figures/diagrams/:
  - replay_buffer.png        — what a replay buffer is + uniform vs priority sampling
  - cce_rollout.png          — counterfactual branching from a single state
  - priority_mixing.png      — comparing uniform / PER / CCE-only / additive mixing
  - pipeline.png             — full Algorithm 2 training pipeline
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patches import Arc
import matplotlib.gridspec as gridspec

OUT = Path(__file__).parent
STYLE = {
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
}
plt.rcParams.update(STYLE)

C_BLUE   = '#2196F3'
C_ORANGE = '#FF9800'
C_GREEN  = '#4CAF50'
C_RED    = '#F44336'
C_PURPLE = '#9C27B0'
C_PINK   = '#E91E63'
C_GREY   = '#9E9E9E'
C_DARK   = '#212121'
C_LIGHT  = '#F5F5F5'

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 1: Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

def draw_replay_buffer():
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # ── Left panel: what the buffer is ───────────────────────────────────────
    ax_buf = fig.add_axes([0.02, 0.05, 0.40, 0.90])
    ax_buf.set_xlim(0, 10)
    ax_buf.set_ylim(0, 10)
    ax_buf.axis('off')
    ax_buf.set_title('Experience Replay Buffer', fontsize=14, fontweight='bold',
                     pad=10, color=C_DARK)

    # Buffer box
    buf_rect = FancyBboxPatch((0.5, 1.5), 9, 6.5, boxstyle='round,pad=0.15',
                               linewidth=2, edgecolor=C_DARK, facecolor=C_LIGHT)
    ax_buf.add_patch(buf_rect)

    # Individual transitions (s, a, r, s')
    transitions = [
        ('s₀, a₀, r₁, s₁', C_GREY,   'old'),
        ('s₁, a₁, r₂, s₂', C_GREY,   'old'),
        ('s₂, a₂, r₃, s₃', C_BLUE,   ''),
        ('s₃, a₃, r₄, s₄', C_ORANGE, ''),
        ('s₄, a₄, r₅, s₅', C_GREEN,  ''),
        ('s₅, a₅, r₆, s₆', C_PURPLE, 'new ←'),
    ]

    slot_h = 0.85
    y_start = 7.8
    for i, (label, color, tag) in enumerate(transitions):
        y = y_start - i * slot_h
        rect = FancyBboxPatch((1.0, y - 0.35), 7.5, 0.70,
                               boxstyle='round,pad=0.05',
                               linewidth=1.5, edgecolor=color,
                               facecolor=color + '22')
        ax_buf.add_patch(rect)
        ax_buf.text(4.75, y, label, ha='center', va='center',
                    fontsize=10, color=C_DARK, fontfamily='monospace')
        if tag:
            col = C_PURPLE if 'new' in tag else C_GREY
            ax_buf.text(8.9, y, tag, ha='left', va='center',
                        fontsize=9, color=col, style='italic')

    # "evicted" label at bottom
    ax_buf.annotate('oldest — evicted when full', xy=(5, 1.1), ha='center',
                    fontsize=9, color=C_GREY, style='italic')

    # Arrow: agent → buffer
    ax_buf.annotate('', xy=(8.65, 8.45), xytext=(11.5, 9.5),
                    xycoords='data', textcoords='data',
                    arrowprops=dict(arrowstyle='->', color=C_PURPLE, lw=2))
    ax_buf.text(9.7, 9.2, 'agent adds\nnew experience', ha='center',
                fontsize=9, color=C_PURPLE)

    # ── Right panel: uniform vs priority sampling ────────────────────────────
    # Two small bar-chart subplots with explicit gap
    ax_u = fig.add_axes([0.48, 0.56, 0.48, 0.34])
    ax_p = fig.add_axes([0.48, 0.08, 0.48, 0.34])

    n = 6
    labels = [f'T{i}' for i in range(n)]
    uniform_p = np.ones(n) / n

    # Fake TD errors: T3 and T5 are high surprise
    td_errors = np.array([0.1, 0.3, 0.15, 0.9, 0.2, 0.8])
    beta = 0.6
    raw = (td_errors + 0.01) ** beta
    per_p = raw / raw.sum()

    colors_u = [C_BLUE] * n
    colors_p = [C_RED if e > 0.5 else C_BLUE for e in td_errors]

    ax_u.bar(labels, uniform_p, color=colors_u, edgecolor='white', linewidth=0.5)
    ax_u.set_ylim(0, 0.32)
    ax_u.set_title('Uniform Sampling — every transition equally likely',
                   fontsize=10, color=C_DARK)
    ax_u.set_ylabel('p(sample)', fontsize=9)
    ax_u.tick_params(labelsize=9)
    ax_u.axhline(1/n, color=C_GREY, linestyle='--', linewidth=0.8)
    ax_u.set_yticks([0, 1/n, 2/n])
    ax_u.set_yticklabels(['0', '1/6', '2/6'], fontsize=8)

    bars = ax_p.bar(labels, per_p, color=colors_p, edgecolor='white', linewidth=0.5)
    ax_p.set_ylim(0, 0.40)
    ax_p.set_title('Prioritized Sampling (PER) — high TD error sampled more',
                   fontsize=10, color=C_DARK)
    ax_p.set_ylabel('p(sample)', fontsize=9)
    ax_p.tick_params(labelsize=9)

    # Annotate high-TD transitions
    for i, (bar, err) in enumerate(zip(bars, td_errors)):
        if err > 0.5:
            ax_p.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'δ={err}', ha='center', fontsize=8, color=C_RED,
                      fontweight='bold')

    red_patch  = mpatches.Patch(color=C_RED,  label='High TD error (surprising)')
    blue_patch = mpatches.Patch(color=C_BLUE, label='Low TD error')
    ax_p.legend(handles=[red_patch, blue_patch], fontsize=8, loc='upper left')

    plt.savefig(OUT / 'replay_buffer.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('replay_buffer.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 2: CCE Counterfactual Rollout
# ─────────────────────────────────────────────────────────────────────────────

def draw_cce_rollout():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.set_title(
        'CCE: Counterfactual Consequence Estimation — "What would have happened?"',
        fontsize=14, fontweight='bold', color=C_DARK, pad=12
    )

    # ── State node ────────────────────────────────────────────────────────────
    state_x, state_y = 2.5, 5.0
    state_circ = plt.Circle((state_x, state_y), 0.75, color=C_DARK,
                              zorder=5, linewidth=2)
    ax.add_patch(state_circ)
    ax.text(state_x, state_y, 'sₜ', ha='center', va='center',
            fontsize=14, color='white', fontweight='bold', zorder=6)
    ax.text(state_x, state_y - 1.2, 'current state\n(stored in buffer)',
            ha='center', fontsize=9, color=C_DARK, style='italic')

    # ── Three branches ────────────────────────────────────────────────────────
    # Spread branches further apart to avoid label crowding
    branches = [
        # (y_branch, action_label, color, taken, return_vals, dist_label)
        (8.5, 'a_taken\n(actual)',        C_BLUE,   True,
         [0.7, 0.8, 0.75, 0.72],          'returns clustered\nnear +0.75'),
        (5.0, 'a_alt 1\n(counterfactual)', C_ORANGE, False,
         [0.2, 0.8, 0.5, 0.6, 0.3],       'returns spread\nacross [0.2, 0.8]'),
        (1.5, 'a_alt 2\n(counterfactual)', C_RED,    False,
         [-0.9, -0.8, -0.85],              'returns clustered\nnear −0.85'),
    ]

    mid_x = 5.5
    end_x = 10.0

    for y_branch, action_label, color, taken, rets, dist_label in branches:
        # Arrow from state to branch point
        ax.annotate('', xy=(mid_x, y_branch), xytext=(state_x + 0.75, state_y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.2,
                                   connectionstyle='arc3,rad=0.0'))

        # Action label — positioned at 1/3 of the way from state to mid_x
        lx = state_x + 0.75 + (mid_x - state_x - 0.75) * 0.42
        ly_offset = 0.55 if y_branch > state_y else (-0.55 if y_branch < state_y else 0.55)
        ly = state_y + (y_branch - state_y) * 0.42 + ly_offset
        ax.text(lx, ly, action_label, ha='center', va='center', fontsize=9,
                color=color, fontweight='bold' if taken else 'normal',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=color, linewidth=1.2))

        # Multiple rollout lines from branch point
        n_rollouts = len(rets)
        spread = 0.6
        for j, ret in enumerate(rets):
            y_jitter = y_branch + spread * (j - (n_rollouts - 1) / 2) / max(n_rollouts - 1, 1) * 2
            ax.plot([mid_x, mid_x + 1.5, end_x],
                    [y_branch, y_jitter, y_jitter],
                    color=color, alpha=0.5, linewidth=1.3, zorder=3)
            ax.plot(end_x, y_jitter, 'o', color=color, markersize=5,
                    zorder=4, alpha=0.8)

        # Distribution mini-histogram — larger and clearer
        hist_x = 10.5
        hist_w = 0.5
        hist_h_scale = 1.0
        counts, bin_edges = np.histogram(rets, bins=5, range=(-1, 1))
        for k, (cnt, left) in enumerate(zip(counts, bin_edges[:-1])):
            if cnt > 0:
                rect = mpatches.Rectangle(
                    (hist_x + k * hist_w * 0.92, y_branch - 0.55),
                    hist_w * 0.88, cnt * hist_h_scale,
                    facecolor=color, alpha=0.75, edgecolor='white', linewidth=0.5
                )
                ax.add_patch(rect)

        # Distribution label — to the right of histogram
        ax.text(hist_x + 5 * hist_w + 0.3, y_branch, dist_label,
                ha='left', va='center', fontsize=8.5, color=color)

        # "Stored in replay" badge for the taken action
        if taken:
            ax.text(mid_x + 0.2, y_branch + 0.7, '✓ replay store',
                    fontsize=8, color=C_BLUE, style='italic')

    # ── Divergence double-arrow ───────────────────────────────────────────────
    ax.annotate('', xy=(11.5, 8.3), xytext=(11.5, 1.7),
                arrowprops=dict(arrowstyle='<->', color=C_DARK, lw=1.8))
    ax.text(11.7, 5.0, 'Total\nVariation\n= CCE score',
            ha='left', va='center', fontsize=10, color=C_DARK, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4',
                      edgecolor=C_DARK, linewidth=1.5))

    # Score interpretation
    ax.text(7.0, 0.5,
            'High divergence between return distributions → action choice mattered → HIGH consequence score',
            ha='center', fontsize=10, color=C_DARK, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                      edgecolor=C_GREEN, linewidth=1.5))

    plt.tight_layout()
    plt.savefig(OUT / 'cce_rollout.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('cce_rollout.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 3: Priority Mixing Comparison
# ─────────────────────────────────────────────────────────────────────────────

def draw_priority_mixing():
    fig, axes = plt.subplots(1, 5, figsize=(21, 5), sharey=False)
    fig.patch.set_facecolor('white')
    fig.suptitle('How Each Method Prioritizes the Same Replay Buffer',
                 fontsize=13, fontweight='bold', color=C_DARK, y=1.01)

    n = 8
    labels = [f'T{i}' for i in range(n)]

    # Ground truth: T2 and T5 are pivotal (truly consequential).
    # T2/T5 have moderate TD error so additive still boosts them above baseline;
    # multiplicative amplifies their dual signal and damps high-TD/low-CCE transitions.
    consequence = np.array([0.05, 0.2, 0.85, 0.15, 0.1,  0.90, 0.08, 0.3])
    td_errors   = np.array([0.80, 0.1, 0.40, 0.7,  0.9,  0.35, 0.75, 0.2])

    def to_prob(arr, beta=0.6):
        raw = (arr + 0.01) ** beta
        return raw / raw.sum()

    p_uniform = np.ones(n) / n
    p_per     = to_prob(td_errors)
    p_cce     = to_prob(consequence)
    p_add     = 0.25 * p_cce + 0.75 * p_per
    p_mul_raw = p_cce * p_per
    p_mul     = p_mul_raw / p_mul_raw.sum()

    # Colors: Uniform=grey, PER=blue, CCE-only=orange, Additive=purple, Mul=pink
    configs = [
        ('Uniform',
         p_uniform, C_GREY,
         'All transitions\nsampled equally',                      False),
        ('DQN + PER\n(TD error only)',
         p_per,     C_BLUE,
         'High surprise → high priority\n(T0,T3,T4,T6 dominate)', False),
        ('CCE Only\n(μ=1)',
         p_cce,     C_ORANGE,
         'High consequence → high priority\n(T2,T5 dominate)',    True),
        ('CCE + TD\nAdditive μ=0.25',
         p_add,     C_PURPLE,
         'Blend: consequential AND surprising\nboth get boosted',  True),
        ('CCE + TD\nMul. ★ winner',
         p_mul,     C_PINK,
         'Must score high on BOTH —\nT4 (high TD, low CCE) damped', True),
    ]

    truly_consequential = [2, 5]
    threshold = 1.0 / n  # above uniform baseline = correctly prioritized

    for ax, (title, probs, base_color, note, correct) in zip(axes, configs):
        bar_colors = []
        for i in range(n):
            if i in truly_consequential:
                bar_colors.append(C_GREEN if probs[i] >= threshold else C_RED)
            else:
                bar_colors.append(base_color + 'BB')

        bars = ax.bar(labels, probs, color=bar_colors, edgecolor='white',
                      linewidth=0.8, zorder=3)
        ax.axhline(threshold, color=C_GREY, linestyle='--', linewidth=0.8,
                   label='uniform baseline', zorder=2)

        # Star on truly consequential transitions
        for i in truly_consequential:
            star_color = C_GREEN if probs[i] >= threshold else C_RED
            ax.text(i, probs[i] + 0.006, '★', ha='center', fontsize=12,
                    color=star_color)

        ax.set_title(title, fontsize=11, fontweight='bold', color=C_DARK, pad=8)
        ax.set_ylim(0, 0.32)
        ax.set_ylabel('sampling probability' if ax is axes[0] else '', fontsize=9)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Note box below each panel
        fc = '#E8F5E9' if correct else '#FAFAFA'
        ec = C_GREEN   if correct else C_GREY
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha='center',
                fontsize=8, color=C_DARK,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=fc,
                          edgecolor=ec, linewidth=1))

    # Legend at bottom
    green_patch = mpatches.Patch(color=C_GREEN,
                                 label='★ Truly consequential (T2, T5) — correctly prioritized')
    red_patch   = mpatches.Patch(color=C_RED,
                                 label='★ Truly consequential — MISSED (below uniform baseline)')
    fig.legend(handles=[green_patch, red_patch], loc='lower center',
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.13))

    plt.tight_layout()
    plt.savefig(OUT / 'priority_mixing.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('priority_mixing.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 4: Full pipeline overview
# ─────────────────────────────────────────────────────────────────────────────

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('Algorithm 2 — Full Training Pipeline',
                 fontsize=13, fontweight='bold', color=C_DARK, pad=10)

    boxes = [
        (1.1,  2.25, 'Environment\n(step)',         C_DARK,   'white'),
        (3.7,  2.25, 'Replay\nBuffer D',            C_BLUE,   'white'),
        (6.3,  3.5,  'CCE Score\n(Algorithm 1)',    C_ORANGE, 'white'),
        (6.3,  1.0,  'TD Error\nδ',                 C_PURPLE, 'white'),
        (9.2,  2.25, 'Combined\nPriority p(j)',     C_GREEN,  'white'),
        (12.0, 2.25, 'Q-Network\nUpdate',           C_RED,    'white'),
    ]

    box_w, box_h = 1.85, 0.9

    short = {
        'Environment': (1.1,  2.25),
        'Replay':      (3.7,  2.25),
        'CCE Score':   (6.3,  3.5),
        'TD Error':    (6.3,  1.0),
        'Combined':    (9.2,  2.25),
        'Q-Network':   (12.0, 2.25),
    }

    for (x, y, label, ec, fc) in boxes:
        rect = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                               boxstyle='round,pad=0.12',
                               linewidth=2, edgecolor=ec,
                               facecolor=ec + '22')
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=9.5, color=C_DARK, fontweight='bold',
                multialignment='center')

    arrows = [
        # (src, dst, label, connectionstyle, color, label_offset_x, label_offset_y)
        ('Environment', 'Replay',   '(s,a,r,s\')\nadded',  'arc3,rad=0',    C_DARK,    0.0, -0.38),
        ('Replay',      'CCE Score','sample\nB_est',        'arc3,rad=-0.2', C_ORANGE,  0.0,  0.0),
        ('Replay',      'TD Error', 'sample\nbatch',        'arc3,rad=0.2',  C_PURPLE,  0.0,  0.0),
        ('CCE Score',   'Combined', 'p^C(j)',               'arc3,rad=0.1',  C_ORANGE,  0.0,  0.0),
        ('TD Error',    'Combined', 'p^δ(j)',               'arc3,rad=-0.1', C_PURPLE,  0.0,  0.0),
        ('Combined',    'Q-Network','weighted\nbatch',       'arc3,rad=0',    C_GREEN,   0.0, -0.38),
        ('Q-Network',   'Environment','new policy\nπ',      'arc3,rad=0.45', C_RED,     0.0,  0.55),
    ]

    for src, dst, lbl, conn, color, dx_off, dy_off in arrows:
        sx, sy = short[src]
        dx, dy = short[dst]

        # Compute arrow endpoints at box edges
        def edge(px, py, toward_x, toward_y, w=box_w, h=box_h):
            """Return point on box edge (px,py) facing toward (toward_x, toward_y)."""
            ddx = toward_x - px
            ddy = toward_y - py
            if abs(ddx) < 0.01:
                return px, py + np.sign(ddy) * h / 2
            if abs(ddy) < 0.01:
                return px + np.sign(ddx) * w / 2, py
            tx = np.sign(ddx) * w / 2
            ty = np.sign(ddy) * h / 2
            if abs(ddy / ddx) < h / w:
                ty = tx * ddy / ddx
            else:
                tx = ty * ddx / ddy
            return px + tx, py + ty

        ax_start = edge(sx, sy, dx, dy)
        ax_end   = edge(dx, dy, sx, sy)

        ax.annotate('', xy=ax_end, xytext=ax_start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                   connectionstyle=conn))

        # Label midpoint — use explicit offsets for horizontal arrows
        mx = (sx + dx) / 2 + dx_off
        if abs(dy - sy) < 0.1:
            # Horizontal arrow: use explicit dy_off to place label above or below
            my = (sy + dy) / 2 + dy_off
        else:
            # Diagonal/vertical arrow: midpoint + small push
            my = (sy + dy) / 2 + (0.28 if dy > sy else -0.28)

        ax.text(mx, my, lbl, ha='center', va='center', fontsize=7.5,
                color=color, style='italic')

    # Mixing formula (multiplicative winner)
    ax.text(9.2, 0.22, 'p(j) ∝ p^C(j) · p^δ(j)   [normalized]',
            ha='center', fontsize=9, color=C_GREEN,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                      edgecolor=C_GREEN, linewidth=1.2))

    plt.tight_layout()
    plt.savefig(OUT / 'pipeline.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('pipeline.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 5: Claim 1 Validation Concept — two independent paths → Spearman ρ
# ─────────────────────────────────────────────────────────────────────────────

def draw_claim1_concept():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('Claim 1: Validating CCE Against Two Independent Oracles',
                 fontsize=13, fontweight='bold', color=C_DARK, pad=10)

    def box(cx, cy, w, h, ec, label, sublabel=None, fontsize=11):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle='round,pad=0.12', linewidth=2,
                                    edgecolor=ec, facecolor=ec + '18'))
        ax.text(cx, cy + (0.2 if sublabel else 0), label, ha='center',
                va='center', fontsize=fontsize, fontweight='bold', color=ec)
        if sublabel:
            ax.text(cx, cy - 0.3, sublabel, ha='center', va='center',
                    fontsize=8, color=C_GREY, style='italic')

    def arrow(x0, y0, x1, y1, color, rad=0.0, label=None):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                   connectionstyle=f'arc3,rad={rad}'))
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx - 0.25, my, label, ha='right', va='center',
                    fontsize=8.5, color=color, style='italic')

    # Top: same state
    box(6.5, 9.0, 4.0, 0.9, C_DARK, 'Same states sₜ evaluated independently',
        sublabel='(FrozenLake: 64 states   |   Chess: 1,857 game positions)')

    # Left: CCE path
    cce_cx, cce_cy = 2.5, 5.8
    ax.add_patch(FancyBboxPatch((0.3, 3.9), 4.4, 3.7,
                                boxstyle='round,pad=0.12', linewidth=2,
                                edgecolor=C_BLUE, facecolor=C_BLUE + '12'))
    ax.text(cce_cx, 7.35, 'CCE Scoring', ha='center', fontsize=11,
            fontweight='bold', color=C_BLUE)
    ax.text(cce_cx, 6.85, '(no oracle — only rollouts)', ha='center',
            fontsize=8, color=C_BLUE, style='italic')
    for i, line in enumerate([
        '① Roll out n alternative actions from sₜ',
        '② Estimate return distribution per action',
        '③ Score = Total Variation across distributions',
    ]):
        ax.text(cce_cx, 6.25 - i * 0.58, line, ha='center', fontsize=8.5,
                color=C_DARK)
    ax.text(cce_cx, 4.2, 'Purely from rollouts.\nNo value function. No labels.',
            ha='center', fontsize=8.5, color=C_BLUE, style='italic',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=C_BLUE, linewidth=0.8))

    # Right: Oracle path
    orc_cx, orc_cy = 10.5, 5.8
    ax.add_patch(FancyBboxPatch((8.3, 3.9), 4.4, 3.7,
                                boxstyle='round,pad=0.12', linewidth=2,
                                edgecolor=C_ORANGE, facecolor=C_ORANGE + '12'))
    ax.text(orc_cx, 7.35, 'Oracle Scoring', ha='center', fontsize=11,
            fontweight='bold', color=C_ORANGE)
    ax.text(orc_cx, 6.85, '(completely independent of CCE)', ha='center',
            fontsize=8, color=C_ORANGE, style='italic')
    ax.text(orc_cx, 6.25, 'FrozenLake:', ha='center', fontsize=9,
            color=C_DARK, fontweight='bold')
    ax.text(orc_cx, 5.75, 'ΔQ* = Q*(s,a*) − Q*(s,a)  via value iteration', ha='center',
            fontsize=8.5, color=C_DARK)
    ax.text(orc_cx, 5.15, 'Chess:', ha='center', fontsize=9,
            color=C_DARK, fontweight='bold')
    ax.text(orc_cx, 4.65, 'Δv via AlphaZero value head', ha='center',
            fontsize=8.5, color=C_DARK)
    ax.text(orc_cx, 4.1, 'Ground-truth importance.\nNo CCE knowledge used.',
            ha='center', fontsize=8.5, color=C_ORANGE, style='italic')

    # Bottom: Spearman ρ
    ax.add_patch(FancyBboxPatch((3.0, 0.6), 7.0, 1.65,
                                boxstyle='round,pad=0.12', linewidth=2.5,
                                edgecolor=C_GREEN, facecolor=C_GREEN + '15'))
    ax.text(6.5, 1.9, 'Spearman ρ — rank correlation of the two scores',
            ha='center', fontsize=11, fontweight='bold', color=C_GREEN)
    ax.text(6.5, 1.35, 'ρ > 0  →  CCE identifies real consequence without access to any oracle',
            ha='center', fontsize=9.5, color=C_DARK)
    ax.text(6.5, 0.85, 'Validated: FL 10 seeds  ·  Chess 100 independent games',
            ha='center', fontsize=8.5, color=C_GREY, style='italic')

    # Arrows
    arrow(5.2, 8.55, 3.8, 7.6, C_BLUE,  rad=0.1)
    arrow(7.8, 8.55, 9.2, 7.6, C_ORANGE, rad=-0.1)
    arrow(2.5, 3.9, 4.2, 2.25, C_GREEN, rad=0.15)
    arrow(10.5, 3.9, 8.8, 2.25, C_GREEN, rad=-0.15)

    plt.tight_layout()
    plt.savefig(OUT / 'claim1_concept.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('claim1_concept.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 6: FrozenLake Oracle — what ΔQ means spatially
# ─────────────────────────────────────────────────────────────────────────────

def draw_frozen_lake_oracle():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_title('FrozenLake Oracle — What ΔQ Measures',
                 fontsize=13, fontweight='bold', color=C_DARK, pad=10)

    # ── Left: mini FrozenLake grid ───────────────────────────────────────────
    # 4×4 grid with holes at (1,1) and (3,2) (row, col from top-left)
    grid_x0, grid_y0 = 0.3, 1.2
    cell = 1.3
    # layout: 0=safe, 1=hole, 2=goal, 3=agent (near hole)
    layout = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 3, 1],
        [0, 0, 0, 2],
    ]
    # ΔQ importance (high near holes, zero at safe distance)
    importance = [
        [0.05, 0.10, 0.02, 0.01],
        [0.30, 0.0,  0.15, 0.05],
        [0.10, 0.40, 0.0,  0.0],
        [0.05, 0.10, 0.35, 0.0],
    ]
    hole_positions = set()
    for r in range(4):
        for c in range(4):
            cx = grid_x0 + c * cell + cell / 2
            cy = grid_y0 + (3 - r) * cell + cell / 2
            t = layout[r][c]
            if t == 1:
                hole_positions.add((r, c))
                fc = '#212121'
                ec = '#212121'
            elif t == 2:
                fc = '#FFF176'
                ec = C_ORANGE
            elif t == 3:
                imp = importance[r][c]
                red_intensity = int(imp * 400)
                fc = f'#{min(255, red_intensity+180):02X}E0E0'
                ec = C_BLUE
            else:
                imp = importance[r][c]
                r_val = min(255, int(imp * 500 + 160))
                fc = f'#{r_val:02X}{max(200, 255 - int(imp*300)):02X}{max(200, 255 - int(imp*300)):02X}'
                ec = C_GREY

            rect = FancyBboxPatch((grid_x0 + c * cell, grid_y0 + (3 - r) * cell),
                                   cell - 0.08, cell - 0.08,
                                   boxstyle='round,pad=0.06',
                                   linewidth=1.5, edgecolor=ec, facecolor=fc)
            ax.add_patch(rect)

            if t == 1:
                ax.text(cx, cy, 'H', ha='center', va='center',
                        fontsize=14, color='white', fontweight='bold')
            elif t == 2:
                ax.text(cx, cy, 'G', ha='center', va='center',
                        fontsize=14, color=C_ORANGE, fontweight='bold')
            elif t == 3:
                agent_circ = plt.Circle((cx, cy + 0.1), 0.28, color=C_BLUE,
                                        zorder=5)
                ax.add_patch(agent_circ)
                ax.text(cx, cy + 0.1, 'A', ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold', zorder=6)
                ax.text(cx, cy - 0.45, 'agent', ha='center', fontsize=7,
                        color=C_BLUE)
            else:
                imp = importance[r][c]
                if imp > 0.05:
                    ax.text(cx, cy, f'ΔQ\n{imp:.2f}', ha='center', va='center',
                            fontsize=7, color='#8B0000', fontweight='bold')

    # Grid border
    grid_total = 4 * cell
    ax.add_patch(FancyBboxPatch((grid_x0 - 0.05, grid_y0 - 0.05),
                                grid_total + 0.05, grid_total + 0.05,
                                boxstyle='round,pad=0.05',
                                linewidth=2, edgecolor=C_DARK, facecolor='none'))

    ax.text(grid_x0 + grid_total / 2, grid_y0 - 0.55,
            'ΔQ heatmap — darker red = more consequential',
            ha='center', fontsize=9, color=C_DARK, style='italic')

    # Legend
    for label, color in [('Hole (H)', '#212121'), ('Goal (G)', C_ORANGE),
                         ('High ΔQ', '#CC0000'), ('Low ΔQ', '#E0E0E0')]:
        pass  # will draw inline

    # ── Right: ΔQ concept (all content starts after x=5.8) ──────────────────
    rx = 9.2   # center x for right panel text
    ax.text(rx, 9.3, 'What ΔQ(s, a) measures:',
            ha='center', fontsize=11, fontweight='bold', color=C_DARK)
    ax.text(rx, 8.8, 'ΔQ(s,a) = Q*(s, a*) − Q*(s, a)',
            ha='center', fontsize=11, color=C_DARK,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                      edgecolor=C_ORANGE, linewidth=1.5))
    ax.text(rx, 8.25, 'How much worse is acting suboptimally at this state?',
            ha='center', fontsize=8.5, color=C_DARK, style='italic')

    # Example 1: near hole — high ΔQ
    ax.add_patch(FancyBboxPatch((6.0, 5.9), 6.0, 1.9,
                                boxstyle='round,pad=0.1', linewidth=1.5,
                                edgecolor=C_RED, facecolor='#FFEBEE'))
    ax.text(rx, 7.55, 'Near a hole — HIGH ΔQ', ha='center', fontsize=10,
            fontweight='bold', color=C_RED)
    ax.text(rx, 7.05, 'Q*(s, step-into-hole) ≈ 0.0', ha='center',
            fontsize=8.5, color=C_DARK)
    ax.text(rx, 6.6, 'Q*(s, a*) = 0.75  →  ΔQ = 0.75  ★', ha='center',
            fontsize=8.5, color=C_RED, fontweight='bold')
    ax.text(rx, 6.15, 'Action choice here matters a lot', ha='center',
            fontsize=8, color=C_RED, style='italic')

    # Example 2: safe — low ΔQ
    ax.add_patch(FancyBboxPatch((6.0, 3.7), 6.0, 1.9,
                                boxstyle='round,pad=0.1', linewidth=1.5,
                                edgecolor=C_BLUE, facecolor='#E3F2FD'))
    ax.text(rx, 5.35, 'Far from danger — LOW ΔQ', ha='center', fontsize=10,
            fontweight='bold', color=C_BLUE)
    ax.text(rx, 4.85, 'Q*(s, any action) ≈ 0.72', ha='center',
            fontsize=8.5, color=C_DARK)
    ax.text(rx, 4.4, 'Q*(s, a*) = 0.74  →  ΔQ = 0.02', ha='center',
            fontsize=8.5, color=C_BLUE, fontweight='bold')
    ax.text(rx, 3.95, 'Action choice here barely matters', ha='center',
            fontsize=8, color=C_BLUE, style='italic')

    # Bottom summary
    ax.add_patch(FancyBboxPatch((6.0, 1.0), 6.5, 2.2,
                                boxstyle='round,pad=0.1', linewidth=1.5,
                                edgecolor=C_GREEN, facecolor='#E8F5E9'))
    ax.text(rx, 2.85, 'Oracle ranks by ΔQ* (true cost of wrong action).',
            ha='center', fontsize=8.5, color=C_DARK)
    ax.text(rx, 2.35, 'Claim 1: do CCE rollout scores correlate with ΔQ*?',
            ha='center', fontsize=10, color=C_GREEN, fontweight='bold')
    ax.text(rx, 1.85, 'Result: yes — Spearman ρ > 0, p < 0.001',
            ha='center', fontsize=8.5, color=C_DARK)
    ax.text(rx, 1.3, 'CCE finds dangerous cells without being told where they are.',
            ha='center', fontsize=8.5, color=C_DARK, style='italic')

    plt.tight_layout()
    plt.savefig(OUT / 'frozen_lake_oracle.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('frozen_lake_oracle.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 7: TD Error — Bellman backup + high vs low δ examples
# ─────────────────────────────────────────────────────────────────────────────

def draw_td_error():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.3, 1.3)
    ax.axis('off')

    q, tgt = 0.20, 0.85
    bar_w  = 1.1

    # baseline
    ax.plot([0.4, 4.6], [0, 0], color=C_GREY, linewidth=1.0)

    # Q(s,a) bar — agent's current prediction
    ax.add_patch(FancyBboxPatch((0.7, 0), bar_w, q,
                                boxstyle='round,pad=0.02', linewidth=0.8,
                                edgecolor='white', facecolor=C_BLUE + 'CC'))
    ax.text(1.25, -0.12, 'Q(s, a)', ha='center', fontsize=11,
            color=C_BLUE, fontweight='bold')
    ax.text(1.25, q + 0.05, f'{q:.2f}', ha='center', fontsize=11,
            color=C_BLUE, fontweight='bold')
    ax.text(1.25, -0.24, "agent's prediction", ha='center', fontsize=8.5,
            color=C_GREY, style='italic')

    # target bar — what actually happened
    ax.add_patch(FancyBboxPatch((3.2, 0), bar_w, tgt,
                                boxstyle='round,pad=0.02', linewidth=0.8,
                                edgecolor='white', facecolor=C_GREEN + 'CC'))
    ax.text(3.75, -0.12, 'r + γ·max Q(s′)', ha='center', fontsize=11,
            color=C_GREEN, fontweight='bold')
    ax.text(3.75, tgt + 0.05, f'{tgt:.2f}', ha='center', fontsize=11,
            color=C_GREEN, fontweight='bold')
    ax.text(3.75, -0.24, 'what actually happened', ha='center', fontsize=8.5,
            color=C_GREY, style='italic')

    # δ bracket between the two bars
    ax.annotate('', xy=(2.95, q), xytext=(2.95, tgt),
                arrowprops=dict(arrowstyle='<->', color=C_RED, lw=2.5))
    ax.text(2.88, (q + tgt) / 2, 'δ', ha='right', va='center',
            fontsize=18, color=C_RED, fontweight='bold')

    # title
    ax.set_title("TD Error  =  target  −  prediction",
                 fontsize=13, fontweight='bold', color=C_DARK, pad=12)

    plt.tight_layout()
    plt.savefig(OUT / 'td_error.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('td_error.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 8: JAX Parallelism — Sequential vs vmap
# ─────────────────────────────────────────────────────────────────────────────

def draw_jax_parallelism():
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Why JAX Makes CCE Feasible — vmap Parallelism',
                 fontsize=13, fontweight='bold', color=C_DARK, y=1.02)

    ax_left  = fig.add_axes([0.03, 0.12, 0.43, 0.82])
    ax_right = fig.add_axes([0.53, 0.12, 0.45, 0.82])
    for ax in [ax_left, ax_right]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

    fig.text(0.485, 0.5, 'vs', ha='center', va='center',
             fontsize=18, fontweight='bold', color=C_GREY)

    # ── LEFT: Sequential (Without JAX) ───────────────────────────────────────
    ax_left.set_title('Without JAX', fontsize=12, fontweight='bold',
                      color=C_RED, pad=10)
    ax_left.text(5, 9.4, 'Python loops — one rollout at a time',
                 ha='center', fontsize=9, color=C_DARK, style='italic')

    code = [
        'for j in range(B_est):    # 256 transitions',
        '  for a in A(sⱼ):        # |A| actions',
        '    for k in range(n):    # 16 rollouts',
        '      rollout(sⱼ, a, k)  ← H steps',
    ]
    for i, line in enumerate(code):
        ax_left.text(0.4, 8.6 - i * 0.50, line, fontsize=7.8,
                     color=C_DARK, fontfamily='monospace')

    palette_seq = [C_BLUE, C_ORANGE, C_BLUE, C_PURPLE, C_ORANGE,
                   C_BLUE, C_PURPLE, C_BLUE, C_ORANGE]
    box_h, y_top = 0.42, 6.5
    for i in range(9):
        y = y_top - i * (box_h + 0.06)
        rect = FancyBboxPatch((1.8, y), 5.8, box_h, boxstyle='round,pad=0.03',
                               linewidth=0.8, edgecolor='white',
                               facecolor=palette_seq[i] + 'AA')
        ax_left.add_patch(rect)
        label = f'rollout {i+1}' if i < 8 else '...'
        ax_left.text(4.7, y + box_h / 2, label, ha='center', va='center',
                     fontsize=7.5 if i < 8 else 11,
                     color='white', fontweight='bold')

    ax_left.text(8.2, 4.7, '×16,384\ntotal', ha='center', fontsize=10,
                 color=C_RED, fontweight='bold')
    ax_left.annotate('', xy=(0.9, 2.8), xytext=(0.9, 6.6),
                     arrowprops=dict(arrowstyle='->', color=C_RED, lw=2.2))
    ax_left.text(0.9, 4.7, 'time', ha='center', fontsize=8.5,
                 color=C_RED, rotation=90)
    ax_left.text(4.7, 1.5, '256 × |A| × 16 × H env steps\n(sequential)',
                 ha='center', fontsize=9, color=C_RED, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFEBEE',
                           edgecolor=C_RED, linewidth=1.2))

    # ── RIGHT: Parallel (With JAX) ────────────────────────────────────────────
    ax_right.set_title('With JAX  (vmap + lax.scan + jit)', fontsize=12,
                       fontweight='bold', color=C_GREEN, pad=10)
    ax_right.text(5, 9.4, 'All rollouts execute simultaneously on GPU',
                  ha='center', fontsize=9, color=C_DARK, style='italic')

    n_rows, n_cols = 6, 9
    grid_x0, grid_y0 = 0.5, 2.8
    cell_w = (8.2 - grid_x0) / n_cols - 0.06
    cell_h, gap_y = 0.55, 0.07
    pal = [C_BLUE + 'CC', C_ORANGE + 'CC', C_PURPLE + 'CC',
           C_GREEN + 'CC', C_PINK + 'CC']
    for r in range(n_rows):
        for c in range(n_cols):
            cx = grid_x0 + c * (cell_w + 0.06)
            cy = grid_y0 + r * (cell_h + gap_y)
            col = pal[(r + c * 2) % len(pal)]
            rect = FancyBboxPatch((cx, cy), cell_w, cell_h,
                                   boxstyle='round,pad=0.02',
                                   linewidth=0.4, edgecolor='white',
                                   facecolor=col)
            ax_right.add_patch(rect)

    ax_right.annotate('', xy=(0.15, 2.8), xytext=(0.15, 7.1),
                      arrowprops=dict(arrowstyle='<->', color=C_GREEN, lw=2.2))
    ax_right.text(0.15, 4.95, 'H\nsteps', ha='center', fontsize=7.5,
                  color=C_GREEN, fontweight='bold', va='center')

    ax_right.text(4.35, 2.35, 'B_est × |A| × n rollouts — all parallel',
                  ha='center', fontsize=8.5, color=C_DARK)

    badges = [
        (9.0, 6.8, 'jit\ncompile once',  C_DARK),
        (9.0, 5.5, 'lax.scan\nstep loop', C_BLUE),
        (9.0, 4.2, 'vmap\nenvs ∥',        C_ORANGE),
    ]
    for bx, by, label, color in badges:
        ax_right.text(bx, by, label, ha='center', va='center', fontsize=7.5,
                      color=color, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=color, linewidth=1.2))

    ax_right.text(4.35, 1.5, 'H steps wall-clock  (same as running 1 rollout)',
                  ha='center', fontsize=9, color=C_GREEN, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='#E8F5E9',
                            edgecolor=C_GREEN, linewidth=1.2))

    plt.savefig(OUT / 'jax_parallelism.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('jax_parallelism.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 9: CCE Score — distribution divergence between action outcomes
# ─────────────────────────────────────────────────────────────────────────────

def draw_cce_score():
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        "C(s, a)  =  mean$_{a' \\neq a}$  D( G$^a_s$ ,  G$^{a'}_s$ )     ←  Consequence Score",
        fontsize=12, fontweight='bold', color=C_PURPLE, y=1.03
    )

    rng = np.random.default_rng(0)

    configs = [
        dict(
            title='High consequence — action choice matters',
            color=C_RED,
            label='HIGH priority → sampled more',
            mu_a=0.7, sig_a=0.08,
            mu_alt=0.15, sig_alt=0.09,
        ),
        dict(
            title='Low consequence — actions lead to same outcome',
            color=C_BLUE,
            label='LOW priority → sampled less',
            mu_a=0.45, sig_a=0.12,
            mu_alt=0.50, sig_alt=0.12,
        ),
    ]

    x = np.linspace(-0.2, 1.2, 300)

    for ax, cfg in zip(axes, configs):
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.8, 6.2)
        ax.axis('off')
        ax.set_title(cfg['title'], fontsize=10.5, fontweight='bold',
                     color=cfg['color'], pad=8)

        from scipy.stats import norm
        ya = norm.pdf(x, cfg['mu_a'],   cfg['sig_a'])
        yb = norm.pdf(x, cfg['mu_alt'], cfg['sig_alt'])

        ax.fill_between(x, ya, alpha=0.45, color=C_BLUE,   label='G$^a_s$  (taken action)')
        ax.fill_between(x, yb, alpha=0.45, color=C_ORANGE, label="G$^{a'}_s$  (alt action)")
        ax.plot(x, ya, color=C_BLUE,   linewidth=1.8)
        ax.plot(x, yb, color=C_ORANGE, linewidth=1.8)

        # axis line
        ax.plot([-0.15, 1.15], [0, 0], color=C_GREY, linewidth=0.8)

        # x-axis label
        ax.text(0.5, -0.55, 'discounted return G', ha='center', fontsize=9,
                color=C_DARK, style='italic')

        # D bracket: draw between the two peaks
        x_a   = cfg['mu_a']
        x_alt = cfg['mu_alt']
        y_bracket = max(norm.pdf(x_a, cfg['mu_a'], cfg['sig_a']),
                        norm.pdf(x_alt, cfg['mu_alt'], cfg['sig_alt'])) + 0.4
        ax.annotate('', xy=(x_a, y_bracket), xytext=(x_alt, y_bracket),
                    arrowprops=dict(arrowstyle='<->', color=cfg['color'], lw=2.0))
        mid_x = (x_a + x_alt) / 2
        ax.text(mid_x, y_bracket + 0.18, 'D', ha='center', fontsize=13,
                color=cfg['color'], fontweight='bold')

        # priority badge
        ax.text(0.5, -0.78, cfg['label'], ha='center', fontsize=8.5,
                color=cfg['color'],
                bbox=dict(boxstyle='round,pad=0.25', facecolor=cfg['color'] + '18',
                          edgecolor=cfg['color'], linewidth=1))

    # shared legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=C_BLUE   + 'AA', edgecolor=C_BLUE,   label='$G^a_s$  — taken action'),
        mpatches.Patch(facecolor=C_ORANGE + 'AA', edgecolor=C_ORANGE, label="$G^{a'}_s$  — alt action"),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(OUT / 'cce_score.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('cce_score.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 10: Dynamic Programming — Bellman backup on a small grid
# ─────────────────────────────────────────────────────────────────────────────

def draw_dynamic_programming():
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.2, 4.5)

    # Title: Bellman equation
    ax.set_title(
        r"$V(s)\ =\ \max_a \sum_{s'} p(s'|s,a)\,[r + \gamma\,V(s')]$"
        "     ←  Bellman Equation",
        fontsize=12, fontweight='bold', color=C_DARK, pad=14
    )

    # ── 4×4 grid of states ────────────────────────────────────────────────
    # Layout: row 0 = bottom, row 3 = top (matching FrozenLake visual convention)
    CELL = 1.0
    GRID_X0 = 0.1
    GRID_Y0 = 0.2

    # Precomputed V* values (qualitative, diminishing from goal)
    # grid[row][col] — row 0 = bottom
    values = [
        [0.50, 0.60, 0.72, 0.85],   # row 0
        [0.40,  None, 0.80,  None],  # row 1 — holes
        [0.30, 0.45, 0.70, 0.90],   # row 2
        [0.20, 0.30, 0.55, 1.00],   # row 3 — top-right is goal
    ]

    def cell_center(row, col):
        cx = GRID_X0 + col * CELL + CELL / 2
        cy = GRID_Y0 + row * CELL + CELL / 2
        return cx, cy

    # Draw cells
    for row in range(4):
        for col in range(4):
            cx, cy = cell_center(row, col)
            v = values[row][col]
            x0 = GRID_X0 + col * CELL
            y0 = GRID_Y0 + row * CELL

            if v is None:
                # Hole
                ax.add_patch(plt.Rectangle((x0, y0), CELL, CELL,
                                           facecolor='#455A64', edgecolor='white', lw=1.5))
                ax.text(cx, cy, 'H', ha='center', va='center',
                        fontsize=12, color='white', fontweight='bold')
            elif row == 3 and col == 3:
                # Goal
                ax.add_patch(plt.Rectangle((x0, y0), CELL, CELL,
                                           facecolor=C_GREEN, edgecolor='white', lw=1.5))
                ax.text(cx, cy + 0.12, 'G', ha='center', va='center',
                        fontsize=12, color='white', fontweight='bold')
                ax.text(cx, cy - 0.18, 'V=1.0', ha='center', va='center',
                        fontsize=8, color='white')
            else:
                # Normal cell: color intensity by value
                alpha = 0.15 + 0.55 * v
                ax.add_patch(plt.Rectangle((x0, y0), CELL, CELL,
                                           facecolor=C_BLUE, alpha=alpha,
                                           edgecolor='white', lw=1.5))
                ax.text(cx, cy, f'V={v:.2f}', ha='center', va='center',
                        fontsize=8.5, color=C_DARK, fontweight='bold')

    # ── Backup arrow: highlight one state and its neighbor ────────────────
    # Show state (2,2) backed up from (3,2) and (2,3)
    focus_row, focus_col = 2, 2
    neighbors = [(3, 2), (2, 3)]
    fx, fy = cell_center(focus_row, focus_col)

    # Highlight focus cell with red border
    ax.add_patch(plt.Rectangle(
        (GRID_X0 + focus_col * CELL, GRID_Y0 + focus_row * CELL),
        CELL, CELL,
        facecolor='none', edgecolor=C_RED, lw=3, zorder=5
    ))

    for nr, nc in neighbors:
        nx, ny = cell_center(nr, nc)
        ax.annotate('', xy=(fx + 0.05 * (nx - fx), fy + 0.05 * (ny - fy)),
                    xytext=(nx - 0.1 * (nx - fx), ny - 0.1 * (ny - fy)),
                    arrowprops=dict(arrowstyle='<-', color=C_RED, lw=2.0))

    ax.text(fx - 0.38, fy + 0.55, 'update\nthis state', ha='center',
            fontsize=7.5, color=C_RED, style='italic')

    # ── Right panel: iteration timeline ───────────────────────────────────
    rx = 5.0
    panel_w = 5.2

    ax.add_patch(FancyBboxPatch((rx, 0.0), panel_w, 3.8,
                                boxstyle='round,pad=0.15', linewidth=1.2,
                                edgecolor=C_GREY, facecolor=C_LIGHT))

    ax.text(rx + panel_w / 2, 3.55, 'How Value Iteration Works',
            ha='center', fontsize=10, fontweight='bold', color=C_DARK)

    steps = [
        (C_GREY,   'Start:  V(s) = 0  for all states'),
        (C_BLUE,   'Sweep:  apply Bellman to every state'),
        (C_BLUE,   'Repeat until V stops changing  (δ < ε)'),
        (C_GREEN,  'Done:  V = V*  (exact optimal values)'),
        (C_PURPLE, 'Then:  Q*(s,a) = Σ p(s\'|s,a)[r + γV*(s\')]'),
    ]
    for i, (color, txt) in enumerate(steps):
        y = 3.0 - i * 0.58
        ax.plot(rx + 0.25, y, 'o', color=color, markersize=7, zorder=5)
        ax.text(rx + 0.52, y, txt, va='center', fontsize=8.8, color=C_DARK)

    # Connector line
    for i in range(len(steps) - 1):
        y_top = 3.0 - i * 0.58 - 0.07
        y_bot = 3.0 - (i + 1) * 0.58 + 0.07
        ax.plot([rx + 0.25, rx + 0.25], [y_top, y_bot],
                color=C_GREY, lw=1.2, zorder=4)

    # Bottom label
    ax.text(rx + panel_w / 2, -0.75,
            'Only works when we know p(s′|s,a) — the full transition table.',
            ha='center', fontsize=8.5, color=C_GREY, style='italic')

    plt.tight_layout()
    plt.savefig(OUT / 'dynamic_programming.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('dynamic_programming.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 11: FrozenLake 8×8 grid — plain map
# ─────────────────────────────────────────────────────────────────────────────

def draw_frozen_lake_grid():
    import numpy as np
    import matplotlib.colors as mcolors

    MAP = [
        'SFFFFFFF',
        'FFFFFFFF',
        'FFFHFFFF',
        'FFFFFHFF',
        'FFFHFFFF',
        'FHHFFFHF',
        'FHFFHFHF',
        'FFFHFFFG',
    ]

    # V* from value iteration (gamma=0.99, slippery), row-major
    V_STAR = np.array([
        [0.415, 0.427, 0.446, 0.468, 0.492, 0.517, 0.535, 0.541],
        [0.412, 0.421, 0.437, 0.458, 0.483, 0.514, 0.546, 0.557],
        [0.397, 0.394, 0.375, 0.000, 0.422, 0.494, 0.561, 0.586],
        [0.369, 0.353, 0.307, 0.200, 0.301, 0.000, 0.569, 0.628],
        [0.333, 0.291, 0.197, 0.000, 0.289, 0.362, 0.535, 0.690],
        [0.306, 0.000, 0.000, 0.086, 0.214, 0.273, 0.000, 0.772],
        [0.289, 0.000, 0.058, 0.048, 0.000, 0.251, 0.000, 0.878],
        [0.280, 0.201, 0.127, 0.000, 0.240, 0.486, 0.737, 0.000],
    ])

    n = 8
    cell = 1.0
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.patch.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(0, n * cell)
    ax.set_ylim(0, n * cell)
    ax.set_title("FrozenLake 8×8 — V* from Value Iteration",
                 fontsize=12, fontweight='bold', color=C_DARK, pad=10)

    cmap = plt.cm.Blues
    v_max = V_STAR.max()

    for r, row_str in enumerate(MAP):
        for c, ch in enumerate(row_str):
            x = c * cell
            y = (n - 1 - r) * cell

            if ch == 'H':
                ax.add_patch(plt.Rectangle((x, y), cell, cell,
                                           facecolor='#455A64', edgecolor='white', lw=1.5))
                ax.text(x + cell/2, y + cell/2, 'H', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')
            elif ch == 'G':
                ax.add_patch(plt.Rectangle((x, y), cell, cell,
                                           facecolor=C_GREEN, edgecolor='white', lw=1.5))
                ax.text(x + cell/2, y + cell/2 + 0.12, 'G', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')
                ax.text(x + cell/2, y + cell/2 - 0.18, 'V=1.0', ha='center', va='center',
                        fontsize=7.5, color='white')
            else:
                v = V_STAR[r, c]
                intensity = 0.12 + 0.72 * (v / v_max)
                fc = cmap(intensity)
                ax.add_patch(plt.Rectangle((x, y), cell, cell,
                                           facecolor=fc, edgecolor='white', lw=1.5))
                label = 'S' if ch == 'S' else ''
                if label:
                    ax.text(x + cell/2, y + cell/2 + 0.14, label, ha='center', va='center',
                            fontsize=10, fontweight='bold', color=C_DARK)
                    ax.text(x + cell/2, y + cell/2 - 0.16, f'{v:.2f}', ha='center', va='center',
                            fontsize=7.5, color=C_DARK)
                else:
                    ax.text(x + cell/2, y + cell/2, f'{v:.2f}', ha='center', va='center',
                            fontsize=8.5, color=C_DARK, fontweight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=v_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('V*(s)', fontsize=9)

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT / 'frozen_lake_grid.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('frozen_lake_grid.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 12: Chess Claim 1 — Spearman ρ across 3 seeds
# ─────────────────────────────────────────────────────────────────────────────

def draw_chess_seed_rho():
    import numpy as np

    seed_rhos = [0.306, 0.384, 0.390]
    seeds     = [0, 1, 2]
    mean_rho  = float(np.mean(seed_rhos))
    std_rho   = float(np.std(seed_rhos))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor('white')

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.15, 0.65)
    ax.set_xticks(seeds)
    ax.set_xticklabels([f'Seed {s}' for s in seeds], fontsize=11)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Random baseline
    ax.axhline(0, color=C_GREY, linewidth=1.2, linestyle='--', zorder=1)
    ax.text(2.48, 0.01, 'random\nbaseline', ha='right', fontsize=8,
            color=C_GREY, style='italic')

    # Mean band
    ax.axhline(mean_rho, color=C_BLUE, linewidth=1.8, linestyle='-', zorder=2, alpha=0.6)
    ax.fill_between([-0.5, 2.5], mean_rho - std_rho, mean_rho + std_rho,
                    color=C_BLUE, alpha=0.10, zorder=1)
    ax.text(-0.48, mean_rho + 0.025,
            f'mean = {mean_rho:.3f} ± {std_rho:.3f}',
            fontsize=8.5, color=C_BLUE)

    # Dots
    colors = [C_PURPLE, C_ORANGE, C_GREEN]
    for i, (s, rho, c) in enumerate(zip(seeds, seed_rhos, colors)):
        ax.plot(s, rho, 'o', color=c, markersize=14, zorder=5)
        ax.text(s, rho + 0.03, f'ρ={rho:.3f}', ha='center', fontsize=9,
                color=c, fontweight='bold')

    # p-value badge
    ax.text(1.0, 0.56, 'p < 0.001  (all seeds)', ha='center', fontsize=9,
            color=C_GREEN, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C_GREEN + '18',
                      edgecolor=C_GREEN, linewidth=1))

    ax.set_title('Chess Claim 1 — Spearman ρ (3 seeds × 100 games)',
                 fontsize=10, fontweight='bold', color=C_DARK, pad=10)

    plt.tight_layout()
    out = OUT.parent / 'real' / 'claim1' / 'chess' / 'fig_chess_seed_rho.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('fig_chess_seed_rho.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 13: AlphaZero Value Head — oracle for chess Claim 1
# ─────────────────────────────────────────────────────────────────────────────

def draw_alphazero_value_head():
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('AlphaZero Value Head — Used as Oracle for Chess Claim 1',
                 fontsize=13, fontweight='bold', color=C_DARK, pad=12)

    # ── LEFT: 5×5 board ──────────────────────────────────────────────────────
    board_x0, board_y0 = 0.2, 2.0
    cell = 0.72
    # Simple mid-game position (row 0 = rank 1, col 0 = file a)
    pieces = {
        (0, 0): ('R', 'w'), (0, 2): ('K', 'w'), (0, 4): ('R', 'w'),
        (1, 1): ('P', 'w'), (1, 3): ('P', 'w'),
        (3, 1): ('p', 'b'), (3, 3): ('p', 'b'),
        (4, 0): ('r', 'b'), (4, 2): ('k', 'b'), (4, 4): ('r', 'b'),
    }
    for r in range(5):
        for c in range(5):
            x = board_x0 + c * cell
            y = board_y0 + r * cell
            light = (r + c) % 2 == 0
            fc = '#F0D9B5' if light else '#B58863'
            ax.add_patch(plt.Rectangle((x, y), cell, cell,
                                       facecolor=fc, edgecolor='none'))
            if (r, c) in pieces:
                sym, side = pieces[(r, c)]
                color = 'white' if side == 'w' else '#1A1A1A'
                ec    = '#333333' if side == 'w' else 'white'
                circ  = plt.Circle((x + cell/2, y + cell/2), cell*0.32,
                                   facecolor=color, edgecolor=ec,
                                   linewidth=1.2, zorder=4)
                ax.add_patch(circ)
                ax.text(x + cell/2, y + cell/2, sym, ha='center', va='center',
                        fontsize=8.5, fontweight='bold', color=ec, zorder=5)

    # Board border
    bw = 5 * cell
    ax.add_patch(plt.Rectangle((board_x0, board_y0), bw, bw,
                                facecolor='none', edgecolor=C_DARK, lw=2.0))

    ax.text(board_x0 + bw/2, board_y0 - 0.35, 'board state  s',
            ha='center', fontsize=9.5, color=C_DARK, style='italic')
    ax.text(board_x0 + bw/2, board_y0 - 0.72, '(white to move)',
            ha='center', fontsize=8, color=C_GREY, style='italic')

    # ── NETWORK BOX ───────────────────────────────────────────────────────────
    net_x, net_y = 3.1, 2.3
    net_w, net_h = 2.6, 3.8

    # Obs label + arrow into network
    obs_x = board_x0 + bw + 0.15
    ax.annotate('', xy=(net_x, net_y + net_h/2), xytext=(obs_x, board_y0 + bw/2),
                arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.8))
    ax.text((obs_x + net_x)/2, board_y0 + bw/2 + 0.28,
            '5×5×115\nobs tensor', ha='center', fontsize=8, color=C_DARK, style='italic')

    # Network outer box
    ax.add_patch(FancyBboxPatch((net_x, net_y), net_w, net_h,
                                boxstyle='round,pad=0.12', linewidth=2,
                                edgecolor=C_DARK, facecolor='#F8F8F8'))
    ax.text(net_x + net_w/2, net_y + net_h - 0.22,
            'AlphaZero  (~1000 Elo)', ha='center', fontsize=9.5,
            fontweight='bold', color=C_DARK)
    ax.text(net_x + net_w/2, net_y + net_h - 0.50,
            'pgx baseline model', ha='center', fontsize=7.5,
            color=C_GREY, style='italic')

    # ResNet body block
    body_y = net_y + 1.5
    ax.add_patch(FancyBboxPatch((net_x + 0.25, body_y), net_w - 0.5, 1.1,
                                boxstyle='round,pad=0.08', linewidth=1.2,
                                edgecolor=C_GREY, facecolor='#EEEEEE'))
    ax.text(net_x + net_w/2, body_y + 0.55, 'ResNet body\n(conv + residual)',
            ha='center', va='center', fontsize=8.5, color=C_DARK)

    # Policy head — dimmed/crossed out
    ph_y = net_y + 0.15
    ax.add_patch(FancyBboxPatch((net_x + 0.2, ph_y), net_w/2 - 0.25, 0.95,
                                boxstyle='round,pad=0.06', linewidth=1.0,
                                edgecolor=C_GREY, facecolor='#EEEEEE', alpha=0.5))
    ax.text(net_x + 0.2 + (net_w/2 - 0.25)/2, ph_y + 0.48,
            'Policy\nhead', ha='center', va='center', fontsize=8,
            color=C_GREY, alpha=0.6)
    ax.text(net_x + 0.2 + (net_w/2 - 0.25)/2, ph_y + 0.0 - 0.22,
            '1225 action logits\n(NOT used here)', ha='center', fontsize=7,
            color=C_GREY, style='italic', alpha=0.7)
    # Strikethrough line over policy head
    px = net_x + 0.2
    ax.plot([px + 0.05, px + net_w/2 - 0.32], [ph_y + 0.15, ph_y + 0.80],
            color=C_RED, lw=1.8, alpha=0.55, zorder=6)
    ax.plot([px + 0.05, px + net_w/2 - 0.32], [ph_y + 0.80, ph_y + 0.15],
            color=C_RED, lw=1.8, alpha=0.55, zorder=6)

    # Value head — highlighted
    vh_x = net_x + net_w/2 + 0.05
    vh_w = net_w/2 - 0.28
    ax.add_patch(FancyBboxPatch((vh_x, ph_y), vh_w, 0.95,
                                boxstyle='round,pad=0.06', linewidth=2.0,
                                edgecolor=C_GREEN, facecolor=C_GREEN + '28'))
    ax.text(vh_x + vh_w/2, ph_y + 0.55, 'Value\nhead', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=C_GREEN)
    ax.text(vh_x + vh_w/2, ph_y - 0.22,
            'scalar  v ∈ [−1, 1]\n← WE USE THIS', ha='center', fontsize=7.5,
            color=C_GREEN, fontweight='bold')

    # Connecting line from body to heads
    mid_body_x = net_x + net_w/2
    body_bottom = body_y
    ax.plot([mid_body_x, mid_body_x], [body_bottom, ph_y + 0.95],
            color=C_GREY, lw=1.2, linestyle=':')
    ax.plot([mid_body_x, net_x + 0.2 + (net_w/2 - 0.25)/2],
            [ph_y + 0.95, ph_y + 0.95], color=C_GREY, lw=1.2, linestyle=':')
    ax.plot([mid_body_x, vh_x + vh_w/2],
            [ph_y + 0.95, ph_y + 0.95], color=C_GREEN, lw=1.5)

    # ── BRANCHES: step to s' → negate → compare ───────────────────────────────
    # Arrow out of value head → oracle computation area
    out_x = net_x + net_w + 0.1
    ax.annotate('', xy=(out_x + 0.4, net_y + net_h/2),
                xytext=(net_x + net_w, net_y + net_h/2 - 0.5),
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=1.8))

    # For each action: step → negate → value
    branch_cx = out_x + 1.6
    branch_data = [
        (7.0, 'a_chosen', C_BLUE,   '−v_φ(s′) = +0.72', True),
        (4.7, 'a_alt 1',  C_ORANGE, '−v_φ(s′) = +0.48', False),
        (2.4, 'a_alt 2',  C_RED,    '−v_φ(s′) = +0.39', False),
    ]

    stem_x = out_x + 0.5
    stem_y = net_y + net_h/2 - 0.5
    for by, label, color, val_str, is_chosen in branch_data:
        # Branch line to box
        ax.plot([stem_x + 0.35, branch_cx - 0.95], [stem_y, by + 0.22],
                color=color, lw=1.5, alpha=0.7)

        # "step env" box
        step_x = branch_cx - 0.9
        ax.add_patch(FancyBboxPatch((step_x, by - 0.15), 1.8, 0.55,
                                    boxstyle='round,pad=0.06', linewidth=1.2,
                                    edgecolor=color, facecolor=color + '18'))
        ax.text(step_x + 0.9, by + 0.12, f'step(s, {label})', ha='center',
                va='center', fontsize=7.5, color=C_DARK)

        # Arrow to value box
        ax.annotate('', xy=(branch_cx + 1.05, by + 0.12),
                    xytext=(step_x + 1.8, by + 0.12),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.3))

        # Value output box
        ax.add_patch(FancyBboxPatch((branch_cx + 1.05, by - 0.15), 2.1, 0.55,
                                    boxstyle='round,pad=0.06', linewidth=1.5,
                                    edgecolor=color,
                                    facecolor=(C_BLUE + '25') if is_chosen else (color + '18')))
        bold = 'bold' if is_chosen else 'normal'
        ax.text(branch_cx + 2.1, by + 0.12, val_str, ha='center',
                va='center', fontsize=8.5, color=color, fontweight=bold)
        if is_chosen:
            ax.text(branch_cx + 2.1, by + 0.44, '← chosen', ha='center',
                    fontsize=7, color=C_BLUE, style='italic')

    # ── ORACLE FORMULA ────────────────────────────────────────────────────────
    formula_x = branch_cx + 1.1
    formula_y = 0.55
    ax.add_patch(FancyBboxPatch((formula_x - 0.1, formula_y - 0.05), 3.5, 1.45,
                                boxstyle='round,pad=0.12', linewidth=2.0,
                                edgecolor=C_PURPLE, facecolor=C_PURPLE + '12'))
    ax.text(formula_x + 1.65, formula_y + 1.25,
            'Oracle Score', ha='center', fontsize=10, fontweight='bold',
            color=C_PURPLE)
    ax.text(formula_x + 1.65, formula_y + 0.75,
            r"Oracle(s) = mean$_{a \neq a_{chosen}}$", ha='center',
            fontsize=9.5, color=C_DARK)
    ax.text(formula_x + 1.65, formula_y + 0.33,
            r"$|v_{chosen} - v_a|$", ha='center',
            fontsize=11, color=C_PURPLE, fontweight='bold')
    ax.text(formula_x + 1.65, formula_y + 0.03,
            r"= mean(|0.72−0.48|, |0.72−0.39|, …)", ha='center',
            fontsize=7.5, color=C_GREY, style='italic')

    # Arrow from branches area to formula
    ax.annotate('', xy=(formula_x - 0.1, formula_y + 0.7),
                xytext=(branch_cx + 3.15, 2.4),
                arrowprops=dict(arrowstyle='->', color=C_PURPLE, lw=1.8,
                                connectionstyle='arc3,rad=-0.25'))

    # Negation note
    ax.text(7.0, 1.55,
            '* pgx returns current-player value → negate after white moves\n'
            '  (black\'s perspective flips sign)',
            ha='center', fontsize=8, color=C_GREY, style='italic')

    plt.tight_layout()
    plt.savefig(OUT / 'alphazero_value_head.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('alphazero_value_head.png saved')


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 14: Chess Claim 1 — Precision@K bar chart
# ─────────────────────────────────────────────────────────────────────────────

def draw_chess_precision_at_k():
    import json

    results_path = (OUT.parent / 'real' / 'claim1' / 'chess'
                    / 'chess_claim1_results.json')
    with open(results_path) as f:
        data = json.load(f)

    p5  = data['precision_5']
    p10 = data['precision_10']
    p20 = data['precision_20']
    n   = data['n_positions']

    labels     = ['Top 5%', 'Top 10%', 'Top 20%']
    cce_vals   = [p5, p10, p20]
    rand_vals  = [0.05, 0.10, 0.20]
    multiples  = [v / r for v, r in zip(cce_vals, rand_vals)]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')

    bars_cce  = ax.bar(x - w/2, cce_vals,  w, color=C_GREEN,  label='CCE (trained)',    zorder=3)
    bars_rand = ax.bar(x + w/2, rand_vals, w, color=C_GREY,   label='Random baseline', zorder=3, alpha=0.7)

    # Multiple labels above CCE bars
    for bar, mult in zip(bars_cce, multiples):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.012,
                f'{mult:.1f}×', ha='center', fontsize=10,
                color=C_GREEN, fontweight='bold')

    ax.set_ylim(0, 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Precision@K', fontsize=11)
    ax.set_title(
        f'Claim 1 — Precision@K: CCE vs Random Baseline\n'
        f'(Gardner Chess, n={n} positions, AlphaZero oracle, seed 2)',
        fontsize=11, fontweight='bold', color=C_DARK, pad=10
    )
    ax.legend(fontsize=10, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_GREY, alpha=0.4)

    out_path = OUT.parent / 'real' / 'claim1' / 'chess' / 'fig_chess_precision_at_k.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'fig_chess_precision_at_k.png saved → {out_path}')


if __name__ == '__main__':
    draw_replay_buffer()
    draw_cce_rollout()
    draw_td_error()
    draw_cce_score()
    draw_priority_mixing()
    draw_pipeline()
    draw_claim1_concept()
    draw_frozen_lake_oracle()
    draw_jax_parallelism()
    draw_dynamic_programming()
    draw_frozen_lake_grid()
    draw_chess_seed_rho()
    draw_alphazero_value_head()
    draw_chess_precision_at_k()
    print('All diagrams saved to', OUT)
