"""Checkpoint → rollout tensor + per-state quantities.

Reuses the agent's compiled counterfactual-rollout fn (the SAME path training uses to score
consequence), mirroring the call convention in
analysis/diagnostics/compute_diagnostics_fl.py:cce_scores — but keeps the FULL returns(S,A,N)
tensor so SNR, concentration and CCE-priority all read from one rollout pass.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import spearmanr

from counterfactual_rl.analysis.metrics import compute_consequence_metric
from counterfactual_rl.analysis.suitability.envs import outcome_probs


def _precision_at_k(true_vals, drilled_vals, k) -> float:
    """Overlap of the top-k by true stakes vs top-k by realized draws.
    Copied from analysis/claim1/frozen_lake/run_analysis.py:precision_at_k (kept inline to
    avoid importing that script module's heavy figure-pipeline dependencies)."""
    n = len(true_vals)
    top_n = max(1, int(n * k))
    top_true = set(np.argsort(true_vals)[-top_n:])
    top_drilled = set(np.argsort(drilled_vals)[-top_n:])
    return len(top_true & top_drilled) / top_n


def q_values(agent, states) -> np.ndarray:
    """Q(s,·) from the loaded network for the given states → (len(states), A)."""
    sa = jnp.asarray(np.asarray(states), dtype=jnp.int32)
    q = jax.vmap(agent.network.apply, in_axes=(None, 0))(agent.params, sa)
    return np.asarray(q)


def greedy_actions(agent, states) -> np.ndarray:
    """The policy's greedy action at each state (the 'chosen' action for CCE and TD)."""
    return q_values(agent, states).argmax(axis=1).astype(np.int32)


def compute_return_tensor(agent, states, batch: int = 256) -> np.ndarray:
    """returns(S, A, N): N greedy-continuation rollouts per (state, action).

    Mirrors compute_diagnostics_fl.cce_scores' rollout loop but returns the full tensor."""
    states = np.asarray(states, dtype=np.int32)
    B = states.shape[0]
    A = agent.n_actions
    N = agent.cf_n_rollouts
    all_actions = jnp.arange(A, dtype=jnp.int32)
    returns = np.zeros((B, A, N), dtype=np.float32)
    for lo in range(0, B, batch):
        hi = min(lo + batch, B)
        sa = jnp.asarray(states[lo:hi], dtype=jnp.int32)
        agent._key, sk = jax.random.split(agent._key)
        keys = jax.random.split(sk, (hi - lo) * A * N).reshape(hi - lo, A, N, 2)
        out = agent._compiled_rollout_fn(agent.params, sa, all_actions, keys)
        returns[lo:hi] = np.asarray(jax.block_until_ready(out))
    return returns


def compute_cce_priority(agent, returns, chosen_actions) -> np.ndarray:
    """Per-state CCE consequence score for the chosen (greedy) action, reusing the exact
    training scoring fn (compute_consequence_metric)."""
    B, A, _ = returns.shape
    scores = np.zeros(B, dtype=np.float64)
    for i in range(B):
        dists = {(a,): returns[i, a] for a in range(A)}
        scores[i] = compute_consequence_metric(
            (int(chosen_actions[i]),), dists,
            metric=agent.consequence_metric, aggregation=agent.consequence_aggregation,
        )
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def compute_abs_td_per_state(agent, states, chosen_actions) -> np.ndarray:
    """|TD| at the chosen (greedy) action for each state, using the EXACT expected next value
    (probability-weighted over outcomes) so there is no rollout sampling noise:
        TD(s) = | E_o[ r + γ (1-done) max_a' Q(s',a') ] − Q(s, a_chosen) |
    """
    env = agent.env
    ns = np.asarray(env.next_states)                 # (S,4,3)
    rw = np.asarray(env.rewards, dtype=np.float64)   # (S,4,3)
    dn = np.asarray(env.dones).astype(np.float64)    # (S,4,3)
    probs = outcome_probs(env)                        # (3,)
    gamma = agent.gamma

    q_all = q_values(agent, np.arange(env.n_states))  # (S_full, A)
    v_next = q_all.max(axis=1)                         # (S_full,)

    states = np.asarray(states, dtype=np.int32)
    td = np.zeros(states.shape[0], dtype=np.float64)
    for i, s in enumerate(states):
        a = int(chosen_actions[i])
        boot = (1.0 - dn[s, a]) * v_next[ns[s, a]]    # (3,)
        target = float(np.sum(probs * (rw[s, a] + gamma * boot)))
        td[i] = abs(target - float(q_all[s, a]))
    return td


def compute_visit_counts(agent, n_episodes: int = 100, gamma=None, max_steps: int = 200) -> np.ndarray:
    """Discounted state-visit frequency d(s) under the greedy policy (tabular).

    Greedy action per state is fixed given params, so precompute it once; then roll episodes."""
    env = agent.env
    gamma = agent.gamma if gamma is None else gamma
    a_star_all = q_values(agent, np.arange(env.n_states)).argmax(axis=1).astype(np.int32)
    d = np.zeros(env.n_states, dtype=np.float64)
    for _ in range(n_episodes):
        agent._key, rk = jax.random.split(agent._key)
        _, state = env.reset(rk)
        state = int(state)
        disc = 1.0
        for _ in range(max_steps):
            d[state] += disc
            a = int(a_star_all[state])
            agent._key, sk = jax.random.split(agent._key)
            _, state, _, done, _ = env.step(sk, jnp.int32(state), jnp.int32(a))
            state = int(state)
            disc *= gamma
            if bool(done):
                break
    total = d.sum()
    return d / total if total > 0 else d


def compute_horizon_sweep(agent, states_sub, horizons) -> dict:
    """mean_s C(s) at several rollout horizons (recompiles the rollout fn per horizon).

    Restores the agent's original cf_horizon + rollout fn on exit."""
    orig = agent.cf_horizon
    means = {}
    try:
        for H in horizons:
            agent.cf_horizon = int(H)
            agent._build_rollout_fn()
            rt = compute_return_tensor(agent, states_sub)
            m = rt.mean(axis=2)
            C = m.max(axis=1) - m.min(axis=1)
            means[int(H)] = float(np.mean(C))
    finally:
        agent.cf_horizon = orig
        agent._build_rollout_fn()
    return means


def _oversampling_stats(d, a, true_stakes, ks):
    """Supply-normalized stats from per-(eval)state draws `d` and supply `a`.

    oversampling(s) = (draws + k0) / (expected + k0),  expected = adds * (Σdraws/Σadds).
    Shrinkage k0 = median expected pulls low-supply states toward 1 (the fair-share null) so tiny
    denominators don't explode. Headline = Spearman(oversampling, stakes)."""
    total_d, total_a = float(d.sum()), float(a.sum())
    if total_d == 0 or total_a == 0:
        return None
    expected = a * (total_d / total_a)
    # Weak, scale-independent prior: 1 expected-draw. Tames only genuinely low-supply states
    # toward the fair-share null (=1); does NOT crush the signal the way median(expected) did.
    k0 = 1.0
    over = (d + k0) / (expected + k0)
    topn = max(1, int(len(true_stakes) * 0.10))
    top = np.argsort(true_stakes)[-topn:]
    return {
        'spearman': float(spearmanr(over, true_stakes).correlation),
        'precision_at_k': {f'{int(k*100)}%': float(_precision_at_k(true_stakes, over, k)) for k in ks},
        'mean_oversampling_top10stakes': float(np.mean(over[top])),
        'shrinkage_k0': k0,
    }


def compute_sampling_timecurve(run_dir, states, true_stakes, ks=(0.05, 0.10, 0.20),
                               n_windows=12) -> dict:
    """Per-WINDOW supply-normalized stats over training (not cumulative).

    Bins the cumulative snapshots into `n_windows` COARSE windows (per-snapshot diffs are mostly
    empty — finer than buffer turnover — which NaN-floods the curve). Each window = what was
    drilled/added between two evenly spaced snapshot boundaries.
    Returns {'updates','progress','spearman','over_top10'} or None if snapshots/adds are absent."""
    path = run_dir if str(run_dir).endswith('.npz') else os.path.join(run_dir, 'sampling.npz')
    if not os.path.exists(path):
        return None
    z = np.load(path)
    if 'adds_snapshots' not in z.files or z['snapshots'].shape[0] == 0:
        return None
    states = np.asarray(states, dtype=np.int64)
    true_stakes = np.asarray(true_stakes, dtype=np.float64)
    dsnap = z['snapshots'].sum(axis=2).astype(np.float64)        # (T, S) cumulative draws
    asnap = z['adds_snapshots'].sum(axis=2).astype(np.float64)   # (T, S) cumulative adds
    ups = z['updates']
    T = dsnap.shape[0]
    bounds = np.unique(np.linspace(0, T - 1, min(n_windows, T) + 1).astype(int))  # coarse boundaries
    upd, spear, over = [], [], []
    for i in range(1, len(bounds)):
        lo, hi = bounds[i - 1], bounds[i]
        dwin = (dsnap[hi] - dsnap[lo])[states]
        awin = (asnap[hi] - asnap[lo])[states]
        st = _oversampling_stats(dwin, awin, true_stakes, ks)
        upd.append(int(ups[hi]))
        spear.append(st['spearman'] if st else np.nan)
        over.append(st['mean_oversampling_top10stakes'] if st else np.nan)
    upd = np.array(upd, dtype=float)
    progress = upd / upd[-1] if upd[-1] > 0 else upd
    return {'updates': upd.tolist(), 'progress': progress.tolist(),
            'spearman': spear, 'over_top10': over}


def compute_realized_sampling(run_dir, states, true_stakes, ks=(0.05, 0.10, 0.20),
                              stakes_argmax_action=None, late_frac=0.5) -> dict:
    """Option B: SUPPLY-NORMALIZED precision@k / Spearman (+ ESS) from a run's `sampling.npz`.

    Raw draws are visitation-confounded, so the headline is the supply-normalized oversampling
    (draws ÷ expected-from-supply). Raw draw stats are kept for the before/after contrast.
    """
    path = run_dir if str(run_dir).endswith('.npz') else os.path.join(run_dir, 'sampling.npz')
    if not os.path.exists(path):
        return {'empty': True, 'reason': 'no sampling.npz', 'path': path}

    z = np.load(path)
    states = np.asarray(states, dtype=np.int64)
    true_stakes = np.asarray(true_stakes, dtype=np.float64)
    draws_state = z['cumulative'].sum(axis=1).astype(np.float64)
    d = draws_state[states]
    out = {'path': path, 'total_draws': int(z['total_draws']), 'n_eval_states': int(states.size),
           'precision_baseline_k': {f'{int(k*100)}%': k for k in ks}}
    if d.sum() == 0:
        out['empty'] = True
        return out

    # RAW (visitation-confounded) — kept for the before/after story.
    out['raw'] = {
        'spearman': float(spearmanr(d, true_stakes).correlation),
        'precision_at_k': {f'{int(k*100)}%': float(_precision_at_k(true_stakes, d, k)) for k in ks},
    }
    p = d / d.sum()
    out['ess'] = float(1.0 / np.sum(p ** 2)); out['ess_frac'] = out['ess'] / states.size

    # SUPPLY-NORMALIZED (the fix) — headline. Needs the `adds` stream (new npz only).
    has_adds = 'adds' in z.files
    out['supply_normalized'] = has_adds
    if has_adds:
        a = z['adds'].sum(axis=1).astype(np.float64)[states]
        out['cumulative'] = _oversampling_stats(d, a, true_stakes, ks)
        snaps, asnaps, ups = z['snapshots'], z['adds_snapshots'], z['updates']
        if snaps.shape[0] > 0:                          # late-window: adds≈occupancy within a window
            cut = int(snaps.shape[0] * late_frac)
            if 0 <= cut < snaps.shape[0]:
                ld = draws_state - snaps[cut].sum(axis=1).astype(np.float64)
                la = z['adds'].sum(axis=1).astype(np.float64) - asnaps[cut].sum(axis=1).astype(np.float64)
                lw = _oversampling_stats(ld[states], la[states], true_stakes, ks)
                if lw is not None:
                    lw['from_update'] = int(ups[cut])
                    out['late_window'] = lw
    return out
