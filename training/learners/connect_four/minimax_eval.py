"""Fixed CPU minimax test foe. Search depth counts individual board moves.

Integer, symmetric window heuristic; exact root scores, alpha-beta below root.
Only tied best root moves are randomized. No shared or training RNG is used.
"""
import numpy as np
from numba import njit

WINDOWS = np.array([
    [(r+k*dr)*7+c+k*dc for k in range(4)]
    for dr, dc in ((0,1),(1,0),(1,1),(1,-1))
    for r in range(6) for c in range(7)
    if 0 <= r+3*dr < 6 and 0 <= c+3*dc < 7], dtype=np.int32)
ORDER = np.array([3,2,4,1,5,0,6])
SPEC = dict(version='minimax-window-v1', depths=[1,2,3,4,5],
            window_weights=[0,1,4,16], center_weight=3, win_value=100000,
            ties='uniform-best-root-seedsequence-case-seed-absolute-ply',
            search='negamax-alpha-beta-full-window-each-root', depth_unit='board-ply')


@njit(cache=True)
def won(board, player):
    for window in WINDOWS:
        count = 0
        for k in window:
            count += board[k] == player
        if count == 4:
            return True
    return False


@njit(cache=True)
def heuristic(board, player):
    score = 0
    weights = (0,1,4,16)
    for window in WINDOWS:
        own, other = 0, 0
        for k in window:
            own += board[k] == player
            other += board[k] == -player
        if other == 0:
            score += weights[min(own,3)]
        if own == 0:
            score -= weights[min(other,3)]
    for r in range(6):
        score += 3*board[r*7+3]*player
    return score


@njit(cache=True)
def _search(board, player, depth, alpha, beta):
    if not np.any(board[:7] == 0):
        return 0
    if depth == 0:
        return heuristic(board, player)
    best = -1000000
    for col in ORDER:
        if board[col] != 0:
            continue
        row = 5
        while board[row*7+col] != 0:
            row -= 1
        k = row*7+col
        board[k] = player
        if won(board, player):
            value = 100000+depth
        else:
            value = -_search(board, -player, depth-1, -beta, -alpha)
        board[k] = 0
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


@njit(cache=True)
def root_scores(board, depth):
    scores = np.full(7, -1000000, dtype=np.int64)
    for col in ORDER:
        if board[col] != 0:
            continue
        row = 5
        while board[row*7+col] != 0:
            row -= 1
        k = row*7+col
        board[k] = 1
        scores[col] = (100000+depth if won(board, 1) else
                       -_search(board, -1, depth-1, -1000000, 1000000))
        board[k] = 0
    return scores


@njit(cache=True)
def batch_scores(boards, depths, active):
    result = np.full((len(boards),7), -1000000, dtype=np.int64)
    for i in range(len(boards)):
        if active[i]:
            result[i] = root_scores(boards[i], depths[i])
    return result


def tied_action(scores, seed, absolute_ply):
    best = np.flatnonzero(scores == scores.max())
    if scores.max() == -1000000:
        raise ValueError('No legal move')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(absolute_ply)]))
    return int(rng.choice(best))
