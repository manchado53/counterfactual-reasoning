"""
Fig C6 — Chess game timeline with CCE + oracle scores and board at peak CCE.

Left panel: CCE Total Variation (orange dashed) and oracle (blue solid) vs white move number.
Background shaded by game phase. Vertical dotted line at peak CCE move.

Right panel: 5×5 board at peak CCE state with from/to squares highlighted.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from typing import List

from .score_positions import OPENING_END, MIDDLEGAME_END
from counterfactual_rl.utils.chess_data_structures import ChessConsequenceRecord

_N = 5

# Direction table for pgx gardner chess (planes 9-48, indices 0-39).
# Source: pgx/_src/gardner_chess_utils.py
# square encoding: sq = col * 5 + row  (column-major)
_DR = [-4,-3,-2,-1, 1, 2, 3, 4,  0, 0, 0, 0, 0, 0, 0, 0,
       -4,-3,-2,-1, 1, 2, 3, 4,  4, 3, 2, 1,-1,-2,-3,-4,
       -1,+1,-2,+2,-1,+1,-2,+2]
_DC = [ 0, 0, 0, 0, 0, 0, 0, 0, -4,-3,-2,-1,+1,+2,+3,+4,
       -4,-3,-2,-1,+1,+2,+3,+4, -4,-3,-2,-1,+1,+2,+3,+4,
       -2,-2,-1,-1,+2,+2,+1,+1]


def _decode_action(action: int):
    """Return (from_display_x, from_display_y, to_display_x, to_display_y).

    pgx square encoding is column-major: sq = col*5 + row.
    So: row = sq % 5, col = sq // 5.
    Display coordinates map directly: display_x = col, display_y = row
    (row 0 = current player's back rank, displayed at the bottom of the board).
    """
    from_sq = action // 49
    plane   = action % 49
    ar0 = from_sq % _N   # action-space row (0 = back rank)
    ac0 = from_sq // _N  # action-space col

    to_sq = None
    if plane < 9:
        # Underpromotion: fixed offset in square space
        to_raw = from_sq + [+1, +6, -4][plane % 3]
        if 0 <= to_raw < 25:
            to_sq = to_raw
    else:
        idx = plane - 9
        ar1 = ar0 + _DR[idx]
        ac1 = ac0 + _DC[idx]
        if 0 <= ar1 < _N and 0 <= ac1 < _N:
            to_sq = ac1 * _N + ar1

    fx, fy = ac0, ar0
    if to_sq is not None:
        tx, ty = to_sq // _N, to_sq % _N
    else:
        tx, ty = None, None
    return fx, fy, tx, ty


def _draw_board_matplotlib(ax, state, action: int):
    """Draw a 5×5 chess board on ax, highlighting from/to squares for action."""
    fx, fy, tx, ty = _decode_action(action)

    # Checkerboard
    for r in range(_N):
        for c in range(_N):
            light = (r + c) % 2 == 0
            color = '#F0D9B5' if light else '#B58863'
            ax.add_patch(patches.Rectangle(
                (c, r), 1, 1, linewidth=0, facecolor=color,
            ))

    # To-square (yellow), drawn before from-square so green wins on overlap
    if tx is not None:
        ax.add_patch(patches.Rectangle(
            (tx, ty), 1, 1, linewidth=0, facecolor='#F6F669', alpha=0.85,
        ))

    # From-square (green)
    ax.add_patch(patches.Rectangle(
        (fx, fy), 1, 1,
        linewidth=3, edgecolor='#2E7D32', facecolor='#AAD56A', alpha=0.85,
    ))

    # Arrow from → to
    if tx is not None:
        ax.annotate(
            '', xy=(tx + 0.5, ty + 0.5), xytext=(fx + 0.5, fy + 0.5),
            arrowprops=dict(arrowstyle='->', color='#2E7D32',
                            lw=2.0, mutation_scale=15),
        )

    # Pieces from observation tensor (5,5,115).
    # obs[obs_row, obs_col, ch]: channels 0-5 = white pieces, 6-11 = black pieces.
    # obs_row 4 = white's back rank → display y = (N-1) - obs_row = 0 (bottom).
    WHITE_SYMS = ['♙', '♘', '♗', '♖', '♕', '♔']
    BLACK_SYMS = ['♟', '♞', '♝', '♜', '♛', '♚']

    try:
        obs = np.array(state.observation)  # (5,5,115)
        for obs_r in range(_N):
            dy = (_N - 1) - obs_r  # display y: white back rank at bottom
            for obs_c in range(_N):
                dx = obs_c
                sym = None
                is_white = False
                for ch, s in enumerate(WHITE_SYMS):
                    if obs[obs_r, obs_c, ch] > 0.5:
                        sym = s
                        is_white = True
                        break
                if sym is None:
                    for ch, s in enumerate(BLACK_SYMS):
                        if obs[obs_r, obs_c, ch + 6] > 0.5:
                            sym = s
                            break
                if sym:
                    if is_white:
                        ax.text(dx + 0.5, dy + 0.5, sym,
                                ha='center', va='center', fontsize=15,
                                color='white', fontweight='bold',
                                path_effects=[pe.withStroke(linewidth=2,
                                                            foreground='#222222')])
                    else:
                        ax.text(dx + 0.5, dy + 0.5, sym,
                                ha='center', va='center', fontsize=15,
                                color='#111111')
    except Exception:
        pass

    ax.set_xlim(0, _N)
    ax.set_ylim(0, _N)
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_board_svg(ax, state):
    """Render board via pgx SVG → cairosvg → matplotlib image."""
    import pgx
    import cairosvg
    import imageio.v2 as imageio
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        svg_path = f'{tmpdir}/board.svg'
        png_path = f'{tmpdir}/board.png'
        pgx.save_svg(state, svg_path)
        cairosvg.svg2png(url=svg_path, write_to=png_path, scale=2.0)
        img = imageio.imread(png_path)

    ax.imshow(img)
    ax.axis('off')


def plot_c6_timeline(
    records: List[ChessConsequenceRecord],
    oracle_scores: List[float],
    out_path,
    figsize=(10, 4.5),
):
    """
    Parameters
    ----------
    records       : records for a single game (sequential white moves)
    oracle_scores : oracle score per record (same order)
    out_path      : save path
    """
    moves    = [r.timestep for r in records]
    cce_vals = [r.tv_score or 0.0 for r in records]  # TV distance, matches run_analysis.py
    ora_vals = list(oracle_scores)

    peak_idx  = int(np.argmax(cce_vals))
    peak_move = moves[peak_idx]

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(1, 3, wspace=0.35)
    ax_line  = fig.add_subplot(gs[0, :2])
    ax_board = fig.add_subplot(gs[0, 2])

    # Phase background shading
    max_move = max(moves) if moves else MIDDLEGAME_END
    phase_regions = [
        (0,             OPENING_END,    '#E3F2FD', 'Opening'),
        (OPENING_END,   MIDDLEGAME_END, '#FFF8E1', 'Middlegame'),
        (MIDDLEGAME_END, max_move + 1,  '#FCE4EC', 'Endgame'),
    ]
    for x0, x1, color, label in phase_regions:
        if x0 >= max_move + 1:
            continue
        ax_line.axvspan(x0, min(x1, max_move + 1), alpha=0.35,
                        color=color, label=label)

    ax_line.plot(moves, ora_vals, color='#1565C0', linewidth=1.8,
                 label='Oracle (value head)')
    ax_line.plot(moves, cce_vals, color='#E65100', linewidth=1.8,
                 linestyle='--', label='CCE Total Variation')
    ax_line.axvline(peak_move, color='#37474F', linewidth=1.2,
                    linestyle=':', alpha=0.8)

    ax_line.set_xlabel('White move number', fontsize=10)
    ax_line.set_ylabel('Score', fontsize=10)
    ax_line.set_title('CCE vs Oracle — game timeline', fontsize=11)
    ax_line.legend(fontsize=8, loc='upper left')
    ax_line.set_xlim(0, max_move + 1)

    # Board at peak CCE
    peak_record = records[peak_idx]
    if peak_record.pgx_state is not None:
        try:
            _draw_board_svg(ax_board, peak_record.pgx_state)
        except Exception:
            _draw_board_matplotlib(ax_board, peak_record.pgx_state,
                                   peak_record.action)
    else:
        ax_board.text(0.5, 0.5, 'state\nnot stored', ha='center', va='center',
                      transform=ax_board.transAxes, fontsize=9, color='gray')
        ax_board.axis('off')

    ax_board.set_title(f'Move {peak_move} — peak CCE', fontsize=10)

    from matplotlib.patches import Patch
    ax_board.legend(
        handles=[
            Patch(facecolor='#AAD56A', edgecolor='#2E7D32', label='Moved from'),
            Patch(facecolor='#F6F669', edgecolor='gray',    label='Moved to'),
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.10),
        fontsize=8,
        framealpha=0.9,
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Fig C6 saved → {out_path}')
