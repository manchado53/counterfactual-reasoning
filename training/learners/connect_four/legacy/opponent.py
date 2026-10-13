"""Rule-based opponent for Connect Four: Win → Block → Fork → Center → Random.

Fully JAX-traceable — no Python control flow inside the scoring logic.
pgx convention: row 0 = top, row 5 = bottom. Pieces fill from row 5 upward.
Drop row = 5 - count_of_pieces_in_column.
"""

import numpy as np
import jax
import jax.numpy as jnp


def _make_windows():
    """Enumerate all 69 four-in-a-row windows as (row, col) index pairs."""
    w = []
    for r in range(6):
        for c in range(4):
            w.append([(r, c + i) for i in range(4)])         # horizontal
    for r in range(3):
        for c in range(7):
            w.append([(r + i, c) for i in range(4)])         # vertical
    for r in range(3):
        for c in range(4):
            w.append([(r + i, c + i) for i in range(4)])     # diagonal ↗ (row-0=bottom)
    for r in range(3, 6):
        for c in range(4):
            w.append([(r - i, c + i) for i in range(4)])     # diagonal ↘ (row-0=bottom)
    return np.array(w)   # (69, 4, 2)


_C4_WINDOWS = _make_windows()
_WIN_ROWS = _C4_WINDOWS[:, :, 0]   # (69, 4) numpy int — keep as numpy to avoid tracer leaks
_WIN_COLS = _C4_WINDOWS[:, :, 1]   # (69, 4) numpy int — JAX accepts numpy for gather indices

_CENTER_BIAS = jnp.array([0, 1, 2, 3, 2, 1, 0], dtype=jnp.float32)


def _win_count(board_float, row, col):
    """Number of 4-in-a-row windows completed by placing a piece at (row, col)."""
    new_board = board_float.at[row, col].set(1.0)
    vals = new_board[_WIN_ROWS, _WIN_COLS]   # (69, 4)
    return (vals.sum(axis=1) == 4).sum()


def _threat_count(board_float, row, col):
    """Number of windows with exactly 3 of the player's pieces after placing at (row, col)."""
    new_board = board_float.at[row, col].set(1.0)
    vals = new_board[_WIN_ROWS, _WIN_COLS]   # (69, 4)
    return (vals.sum(axis=1) == 3).sum()


def rule_based_action(state, rng_key):
    """Select a Connect Four action using Win → Block → Fork → Center → Random priority.

    Scores (before tiebreak noise):
        Win  : 1000   — complete a 4-in-a-row this move
        Block:  100   — prevent opponent from completing a 4-in-a-row
        Fork :   10   — create two simultaneous threats (fork >= 2 threats)
        Center:  0-3  — prefer center columns as tiebreak

    Args:
        state:   pgx Connect Four state (current_player is the opponent)
        rng_key: JAX random key for tiebreak noise and random fallback

    Returns:
        int32 scalar — chosen column (0–6)
    """
    obs   = state.observation         # (6, 7, 2) bool — channel 0 = current player
    legal = state.legal_action_mask   # (7,) bool

    opp = obs[:, :, 0].astype(jnp.float32)   # current player (opponent) pieces
    agt = obs[:, :, 1].astype(jnp.float32)   # other player (agent) pieces

    # pgx uses row 0 = top, row 5 = bottom; pieces fill from row 5 upward.
    # drop_row = 5 - count = row where the next piece lands.
    drop_rows = 5 - (opp + agt).astype(jnp.int32).sum(axis=0)   # (7,)

    def score_col(col):
        row    = drop_rows[col]
        win    = _win_count(opp, row, col)
        block  = _win_count(agt, row, col)
        fork   = _threat_count(opp, row, col)
        center = _CENTER_BIAS[col]
        return (
            (win   > 0).astype(jnp.float32) * 1000.0 +
            (block > 0).astype(jnp.float32) * 100.0  +
            (fork  >= 2).astype(jnp.float32) * 10.0  +
            center
        )

    scores = jax.vmap(score_col)(jnp.arange(7))          # (7,)
    noise  = jax.random.uniform(rng_key, (7,))            # tiebreak + random fallback
    final  = jnp.where(legal, scores + noise, -jnp.inf)
    return jnp.argmax(final).astype(jnp.int32)
