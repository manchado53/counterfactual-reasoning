"""
Default config for the routing (CVRP / TSP) agents.

Mirrors the FrozenLake config so every CCE knob keeps the SAME name across envs.
Routing-specific keys replace FrozenLake's map keys:

    capacity        int  -> real CVRP (the load limit); None -> TSP mode
    instance        name of a built-in node layout ('default')
    n_customers     used only by the built-in 'random' layouts

Like FrozenLake, routing trains by EPISODES (one episode = one full delivery plan),
not by chunks — so `alpha / n_episodes / eval_interval` semantics carry over unchanged.
"""

DEFAULT_CONFIG = {
    # ── environment ──────────────────────────────────────────────────────────
    'instance': 'default',
    'capacity': 10,            # None -> TSP mode (no load limit)
    'demand_scale': 1.0,       # multiplies the built-in demands (feasibility is re-checked)
    # Traffic. 0.0 = deterministic -> CLAIM 2 (CCE's replay advantage lives at determinism).
    # >0 = stochastic leg costs -> required for CLAIM 1, since CCE's total-variation score
    # is degenerate under determinism (point-mass returns -> TV collapses to 0/1).
    # Zero-mean, so the exact oracle and the optimal plan are unchanged.
    'travel_noise': 0.0,

    # ── BUDGET MODE (orienteering variant) — the fix for the Claim-2 null ────
    # None -> classic CVRP (reward = -distance). Set budget_mult to switch the goal to
    # "serve as many customers as you can on a closed tour within a travel budget B".
    # Reward becomes an INTEGER COUNT, so action outcomes can TIE and CCE's
    # total-variation score stops saturating; B near the all-customers optimum also
    # keeps the task from being solved in ~750 episodes (the headroom problem).
    # B = round(budget_mult * exact optimal all-customers tour), in integer units.
    # THIS IS THE DIAL. Measured stakes-concentration RISES with budget_mult
    # (gini 0.22 at 0.55 -> 0.37 at 1.30), so the registered prediction is that CCE's
    # advantage is LARGEST at the loose end.
    'budget_mult': None,
    'budget_units': None,      # set B directly in integer units (overrides budget_mult)
    'dist_scale': 10,          # integer distance units per unit euclidean distance

    # ── algorithm ────────────────────────────────────────────────────────────
    'algorithm': 'consequence-dqn',   # dqn-uniform | dqn (=PER) | consequence-dqn
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_episodes': 4000,
    'alpha': 0.001,
    'batch_size': 32,
    'buffer_capacity': 100000,
    'target_update_freq': 200,
    'n_steps_per_update': 4,
    'hidden_dim': 64,
    'n_layers': 2,
    'n_episodes': 10000,
    'eval_interval': 100,
    'eval_episodes': 20,       # the greedy policy is deterministic here, so a few suffice
    'save_every': 500,
    'n_checkpoints': 100,
    'seed': 0,
    'PER_parameters': {'eps': 0.01, 'beta': 0.25, 'maximum_priority': 1.0},

    # ── consequence-weighted PER (identical knob names to every other env) ───
    'mu': 0.25,
    'priority_mixing': 'additive',
    'mu_c': 1.0,
    'mu_delta': 1.0,
    'score_interval': 100,
    'n_score_sample': 128,
    'consequence_metric': 'total_variation',
    'consequence_aggregation': 'weighted_mean',
    'cf_horizon': 30,          # must cover a full plan: n_customers + reloads + 1
    'cf_n_rollouts': 20,
    'cf_gamma': 0.99,

    # ── replay-sampling logging (Option B) ───────────────────────────────────
    'log_sampling': False,
    'sampling_snapshot_interval': 2000,

    # ── early stop: fraction of optimal (1.0 == provably optimal plan) ───────
    'early_stop_opt_ratio': None,

    # ── vectorized training ──────────────────────────────────────────────────
    'vectorized': False,
    'n_envs': 256,
    'collect_steps': 128,
}
