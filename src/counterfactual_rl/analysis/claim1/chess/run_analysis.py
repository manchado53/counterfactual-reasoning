"""
Claim 1 chess analysis — oracle correlation + figures C5 and C6.

Usage:
    # Smoke test (fast):
    python -m counterfactual_rl.analysis.claim1.chess.run_analysis \
        --n-episodes 5 --n-rollouts 8 --horizon 6

    # Full run:
    python -m counterfactual_rl.analysis.claim1.chess.run_analysis
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pgx
import scipy.special
from scipy.stats import spearmanr, pearsonr, t as t_dist

from .oracle import compute_oracle_for_records
from .score_positions import collect_positions, phase_from_timestep
from .scatter import plot_c5_scatter
from .timeline import plot_c6_timeline

FIGURE_DIR = Path(__file__).parents[5] / 'docs' / 'figures' / 'real' / 'claim1' / 'chess'


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def precision_at_k(oracle, cce, k=0.10):
    n = len(oracle)
    top_n = max(1, int(n * k))
    top_oracle = set(np.argsort(oracle)[-top_n:])
    top_cce    = set(np.argsort(cce)[-top_n:])
    return len(top_oracle & top_cce) / top_n


def sampling_kl(oracle, cce, beta=0.4, eps=1e-8):
    p_oracle = (np.array(oracle) + eps) ** beta
    p_oracle /= p_oracle.sum()
    p_cce    = (np.array(cce)    + eps) ** beta
    p_cce   /= p_cce.sum()
    return float(scipy.special.rel_entr(p_oracle, p_cce).sum())


def sampling_pearson(oracle, cce, beta=0.4, eps=1e-8):
    p_oracle = (np.array(oracle) + eps) ** beta
    p_oracle /= p_oracle.sum()
    p_cce    = (np.array(cce)    + eps) ** beta
    p_cce   /= p_cce.sum()
    r, _ = pearsonr(p_oracle, p_cce)
    return float(r)


# ---------------------------------------------------------------------------
# Game splitting (independent unit = game, not position)
# ---------------------------------------------------------------------------

def _split_into_games(records, oracle_scores):
    """Split flat record/score lists into per-game lists.

    Positions within a game share the same board trajectory and are correlated.
    Games are the true independent unit for statistical tests.
    A new game starts when timestep resets (timestep <= previous timestep).
    """
    games = []
    cur_recs, cur_ors = [], []
    prev_t = -1
    for r, o in zip(records, oracle_scores):
        if r.timestep <= prev_t and cur_recs:
            games.append((cur_recs, cur_ors))
            cur_recs, cur_ors = [], []
        cur_recs.append(r)
        cur_ors.append(o)
        prev_t = r.timestep
    if cur_recs:
        games.append((cur_recs, cur_ors))
    return games


def game_level_spearman(records, oracle_scores):
    """Compute per-game Spearman ρ, then aggregate across games.

    Returns (mean_rho, std_rho, n_games, t_stat, p_value, per_game_rhos).
    p_value is one-sided (H1: mean ρ > 0) using a t-test over per-game ρ values.
    Games with fewer than 3 positions are skipped (can't compute ρ).
    """
    games = _split_into_games(records, oracle_scores)
    per_game_rhos = []
    for recs, ors in games:
        cce = [r.tv_score or 0.0 for r in recs]
        if len(set(cce)) < 2 or len(set(ors)) < 2 or len(cce) < 3:
            continue
        rho, _ = spearmanr(ors, cce)
        if not np.isnan(rho):
            per_game_rhos.append(rho)

    n = len(per_game_rhos)
    mean_rho = float(np.mean(per_game_rhos))
    std_rho  = float(np.std(per_game_rhos, ddof=1))
    se       = std_rho / np.sqrt(n)
    t_stat   = mean_rho / se if se > 0 else 0.0
    p_value  = float(t_dist.sf(t_stat, df=n - 1))  # one-sided
    return mean_rho, std_rho, n, t_stat, p_value, per_game_rhos


# ---------------------------------------------------------------------------
# Select best single game for C6 (most variance in CCE across its moves)
# ---------------------------------------------------------------------------

def _select_game_for_c6(records, oracle_scores):
    """Return (game_records, game_oracle_scores) for the game with highest CCE variance."""
    # Group by episode: timestep resets to 0 at each new game
    games = []
    current_game = []
    current_oracle = []
    prev_t = -1
    for r, o in zip(records, oracle_scores):
        if r.timestep <= prev_t:
            if current_game:
                games.append((current_game, current_oracle))
            current_game = []
            current_oracle = []
        current_game.append(r)
        current_oracle.append(o)
        prev_t = r.timestep
    if current_game:
        games.append((current_game, current_oracle))

    if not games:
        return records, oracle_scores

    # Pick game with highest CCE variance (most interesting timeline)
    best_idx = int(np.argmax([
        np.var([r.wasserstein_score or 0.0 for r in g])
        for g, _ in games
    ]))
    return games[best_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Chess Claim 1: CCE vs AlphaZero oracle correlation'
    )
    parser.add_argument('--n-episodes', type=int, default=20)
    parser.add_argument('--n-rollouts', type=int, default=32)
    parser.add_argument('--horizon',    type=int, default=10)
    parser.add_argument('--top-k',      type=int, default=10)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--beta',              type=float, default=0.4,
                        help='Exponent for sampling KL metric')
    parser.add_argument('--oracle-chunk-size', type=int,   default=1024,
                        help='Model batch size for oracle value-head calls')
    parser.add_argument('--chunk-size',        type=int,   default=32,
                        help='Positions per JIT call in batched CCE scoring')
    parser.add_argument('--records-cache', type=str, default=None,
                        help='Path to cached records .pkl (skip collection if provided)')
    parser.add_argument('--cache-path', type=str, default=None,
                        help='Where to save newly collected records (default: records_cache.pkl)')
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH = Path(args.cache_path) if args.cache_path else Path(__file__).parent / 'records_cache.pkl'

    # 1. Collect positions and CCE scores (or load from cache)
    if args.records_cache and Path(args.records_cache).exists():
        print(f'\n=== Loading cached records from {args.records_cache} ===')
        with open(args.records_cache, 'rb') as f:
            records = pickle.load(f)
        print(f'Loaded {len(records)} records')
    elif CACHE_PATH.exists() and not args.records_cache:
        print(f'\n=== Loading cached records from {CACHE_PATH} ===')
        with open(CACHE_PATH, 'rb') as f:
            records = pickle.load(f)
        print(f'Loaded {len(records)} records')
    else:
        print(f'\n=== Collecting positions ({args.n_episodes} games) ===')
        records = collect_positions(
            n_episodes=args.n_episodes,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            seed=args.seed,
            verbose=True,
        )
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(records, f)
        print(f'Records cached → {CACHE_PATH}')
    print(f'Total positions: {len(records)}')

    # 2. Oracle labels
    print('\n=== Computing oracle labels (value head) ===')
    baseline_model = pgx.make_baseline_model("gardner_chess_v0")
    pgx_env = pgx.make("gardner_chess")
    oracle_scores = compute_oracle_for_records(
        records, pgx_env, baseline_model, chunk_size=args.oracle_chunk_size
    )
    print(f'Oracle done. mean={np.mean(oracle_scores):.4f}  '
          f'std={np.std(oracle_scores):.4f}')

    # 3. Extract CCE scores and game phases
    cce_scores  = [r.tv_score or 0.0 for r in records]
    game_phases = [phase_from_timestep(r.timestep) for r in records]

    # 4. Metrics
    # Game-level Spearman (primary) — games are the true independent unit
    mean_rho, std_rho, n_games, t_stat, pval_game, per_game_rhos = \
        game_level_spearman(records, list(oracle_scores))

    # Position-level Spearman (kept for scatter plot annotation only)
    rho_pos, _ = spearmanr(oracle_scores, cce_scores)

    p5  = precision_at_k(oracle_scores, cce_scores, k=0.05)
    p10 = precision_at_k(oracle_scores, cce_scores, k=0.10)
    p20 = precision_at_k(oracle_scores, cce_scores, k=0.20)
    kl  = sampling_kl(oracle_scores, cce_scores, beta=args.beta)
    pr  = sampling_pearson(oracle_scores, cce_scores, beta=args.beta)

    ci95 = 1.96 * std_rho / np.sqrt(n_games)
    pval_str = f'{pval_game:.4f}' if pval_game >= 0.0001 else '<0.0001'
    print(f'\n=== Results ===')
    print(f'n_positions         : {len(records)}')
    print(f'n_games             : {n_games}')
    print(f'Spearman ρ (game)   : {mean_rho:.3f} ± {std_rho:.3f}  '
          f'95% CI [{mean_rho-ci95:.3f}, {mean_rho+ci95:.3f}]  '
          f't={t_stat:.2f}  p={pval_str}  (one-sided, H1: ρ>0)')
    print(f'Spearman ρ (pos)    : {rho_pos:.3f}  (for reference only — positions not independent)')
    print(f'Precision@5%        : {p5:.3f}  (random baseline 0.05)')
    print(f'Precision@10%       : {p10:.3f}  (random baseline 0.10)')
    print(f'Precision@20%       : {p20:.3f}  (random baseline 0.20)')
    print(f'Sampling KL         : {kl:.4f}  (β={args.beta})')
    print(f'Sampling Pearr      : {pr:.3f}')

    # 5. Save results JSON (for make_table.py to use)
    results = {
        'n_positions':       len(records),
        'n_games':           n_games,
        'spearman_rho':      mean_rho,        # game-level (primary)
        'spearman_rho_std':  std_rho,
        'spearman_rho_ci95': ci95,
        'spearman_t':        t_stat,
        'spearman_pval':     pval_game,
        'spearman_rho_pos':  float(rho_pos),  # position-level (reference only)
        'precision_5':       float(p5),
        'precision_10':      float(p10),
        'precision_20':      float(p20),
        'sampling_kl':       float(kl),
        'sampling_pearr':    float(pr),
        'per_game_rhos':     [float(r) for r in per_game_rhos],
    }
    results_path = FIGURE_DIR / 'chess_claim1_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved → {results_path}')

    # 6. Fig C5 — scatter
    c5_path = FIGURE_DIR / 'fig_c5_chess_scatter.png'
    plot_c5_scatter(oracle_scores, cce_scores, game_phases, c5_path)

    # 7. Fig C6 — timeline for most-interesting game
    game_records, game_oracle = _select_game_for_c6(records, oracle_scores)
    c6_path = FIGURE_DIR / 'fig_c6_chess_timeline.png'
    plot_c6_timeline(game_records, game_oracle, c6_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
