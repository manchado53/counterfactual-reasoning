"""
Gardner chess counterfactual consequence analysis — baseline vs baseline.

Plays one full game with the pgx AlphaZero baseline (~1000 Elo) as white,
computes counterfactual consequence scores at every move, saves an animated
game replay and consequence analysis plots.

Outputs (written to runs/chess_cf_<timestamp>/):
    game.svg                    — animated board replay (open in browser)
    analysis_comprehensive.png  — consequence scores + return distributions

Usage:
    python -m counterfactual_rl.simulations.chess_baseline_analysis
"""

import sys
sys.path.insert(0, __file__.split('src/')[0] + 'src')

import jax
from counterfactual_rl.envs.chess import GardnerChessEnv
from counterfactual_rl.analysis.chess_counterfactual import ChessCounterfactualAnalyzer

env = GardnerChessEnv(seed=0, opponent='baseline')

analyzer = ChessCounterfactualAnalyzer(
    env,
    rollout_policy='baseline',
    horizon=10,
    n_rollouts=100,
    top_k=20,
    store_states=True,
)

key = jax.random.PRNGKey(42)
records = analyzer.evaluate_episode(key)

analyzer.save_game(records)
analyzer.save_plots(records)
