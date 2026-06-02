"""
Collect Gardner chess positions with CCE scores via ChessCounterfactualAnalyzer.

Runs baseline-vs-baseline self-play games and scores each white move using
the Wasserstein CCE metric. Returns records with pgx_state set for oracle labeling.
"""

import jax
from typing import List

from counterfactual_rl.envs.chess import GardnerChessEnv
from counterfactual_rl.analysis.chess_counterfactual import ChessCounterfactualAnalyzer
from counterfactual_rl.utils.chess_data_structures import ChessConsequenceRecord

# Game phase boundaries (white move number, 0-indexed)
OPENING_END    = 10
MIDDLEGAME_END = 25


def phase_from_timestep(timestep: int) -> str:
    if timestep < OPENING_END:
        return 'opening'
    if timestep < MIDDLEGAME_END:
        return 'middlegame'
    return 'endgame'


PHASE_COLORS = {
    'opening':    '#2196F3',  # blue
    'middlegame': '#FF9800',  # orange
    'endgame':    '#F44336',  # red
}


def collect_positions(
    n_episodes: int = 20,
    n_rollouts: int = 32,
    horizon: int = 10,
    top_k: int = 10,
    chunk_size: int = 32,
    seed: int = 0,
    verbose: bool = True,
) -> List[ChessConsequenceRecord]:
    """
    Play n_episodes baseline-vs-baseline games, computing CCE at every white move.

    Uses collect_and_score_batched: games are played first (no rollouts), then all
    positions are batch-scored with a triple-vmapped rollout function (~50× faster
    than the sequential evaluate_multiple_episodes approach).

    Parameters
    ----------
    n_episodes  : number of self-play games to run
    n_rollouts  : counterfactual rollouts per candidate move
    horizon     : rollout depth in full move-pairs
    top_k       : candidate moves to evaluate at each position
    chunk_size  : positions per JIT call in Phase B (lower if OOM)
    seed        : JAX PRNGKey seed
    verbose     : print per-episode summary

    Returns
    -------
    List[ChessConsequenceRecord] with pgx_state populated on every record.
    """
    env = GardnerChessEnv(seed=seed, opponent='baseline')
    analyzer = ChessCounterfactualAnalyzer(
        env,
        rollout_policy='baseline',
        horizon=horizon,
        n_rollouts=n_rollouts,
        top_k=top_k,
        aggregation='mean',
        store_states=True,
    )
    key = jax.random.PRNGKey(seed)
    records = analyzer.collect_and_score_batched(
        key, n_episodes=n_episodes, chunk_size=chunk_size, verbose=verbose
    )
    if verbose:
        print(f'Collected {len(records)} positions from {n_episodes} games')
    return records
