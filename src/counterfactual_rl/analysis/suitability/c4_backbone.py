"""Connect Four backbone for the suitability pipeline.

Reusable C4 pieces, factored out of the validated probe (`probe_c4.py`) and hardened per review:
  - load a frozen checkpoint with a chosen FOE baked into the rollout atom
  - collect agent-to-move boards by seat-normalized self-play, tracking occupancy BEFORE dedup
  - the rollout-atom return tensor with illegal columns set to NaN (so metrics drop them)
  - CCE priority (the total-variation score CCE actually uses) + greedy actions
  - sampled one-step |TD| (no oracle): |r + γ(1-done)·max Q_target(s') − Q(s,a)|, averaged over foes

Foe/greedy logic mirrors `consequence_dqn.py` EXACTLY so the foe that collects boards == the foe
that scores them. Everything here is FL-untouching (new module).
"""

import pickle
import warnings
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp
import pgx


@contextmanager
def _nan_quiet():
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

from counterfactual_rl.agents.connect_four.consequence_dqn import Connect4ConsequenceDQN, C4_ACTIONS
from counterfactual_rl.analysis.metrics import compute_consequence_metric

_ENV = pgx.make("connect_four")
_step = jax.jit(_ENV.step)


# --------------------------------------------------------------------------------------
# load (construct with foe in config, THEN assign frozen params; load() does not overwrite config)
# --------------------------------------------------------------------------------------
def load_agent(ckpt_path, foe, cf_overrides=None):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    cfg = dict(ck["config"])
    cfg["opponent"] = foe
    if cf_overrides:
        cfg.update(cf_overrides)              # e.g. cf_horizon, mcts_n_sims for the mcts budget cut
    agent = Connect4ConsequenceDQN(ck["env_info"], config=cfg)
    agent.params = jax.tree.map(jnp.array, ck["params"])
    agent.target_params = jax.tree.map(jnp.array, ck["target_params"])
    agent._build_batched_rollout_fn()         # bakes the foe into _compiled_batched_fn
    return agent, cfg


# --------------------------------------------------------------------------------------
# foe + greedy (identical to consequence_dqn.py:103-123)
# --------------------------------------------------------------------------------------
def greedy_action(agent, state):
    q = agent.network.apply(agent.params, state.observation.reshape(-1))   # (1,7)
    return jnp.argmax(jnp.where(state.legal_action_mask, q[0], -jnp.inf))


def foe_action(state, key, foe):
    if foe == "rule_based":
        from counterfactual_rl.agents.connect_four.opponent import rule_based_action
        return rule_based_action(state, key)
    if foe == "mcts":
        from counterfactual_rl.agents.connect_four.opponent_mcts import mcts_action
        return mcts_action(state, key)
    logits = jax.random.normal(key, (C4_ACTIONS,))                          # random foe
    return jnp.argmax(jnp.where(state.legal_action_mask, logits, -jnp.inf))


# --------------------------------------------------------------------------------------
# collect boards (seat-normalized) + discounted occupancy (tracked BEFORE dedup, for NEED)
# --------------------------------------------------------------------------------------
def _eps_greedy(agent, state, key, eps):
    """Greedy with prob 1-eps, else a uniform-random LEGAL move. Diversifies the collected board
    set against near-deterministic foes (rule_based/mcts) AND matches the epsilon-greedy replay
    distribution CCE actually scores during training."""
    if eps <= 0:
        return greedy_action(agent, state)
    key, ek = jax.random.split(key)
    if float(jax.random.uniform(ek)) >= eps:
        return greedy_action(agent, state)
    legal = np.array(state.legal_action_mask).astype(bool)
    return jnp.int32(np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30))).choice(
        np.where(legal)[0]))


def collect_c4_states(agent, foe, n_boards, max_games, gamma, seed=0, eps=0.1):
    key = jax.random.PRNGKey(seed)
    states, phases = [], []
    seen = {}                                  # board_key -> index in `states`
    occ = []                                   # discounted visit weight, aligned to `states`
    games = 0
    while len(states) < n_boards and games < max_games:
        key, gk = jax.random.split(key)
        state = _ENV.init(gk)
        if int(state.current_player) != 0:     # seat-normalize: one foe pre-move so it's our turn
            key, fk = jax.random.split(key)
            state = _step(state, foe_action(state, fk, foe))
        ply = 0
        while not bool(state.terminated | state.truncated):
            if int(state.current_player) != 0:
                key, fk = jax.random.split(key)
                state = _step(state, foe_action(state, fk, foe))
                continue
            bkey = np.array(state.observation).tobytes()
            w = gamma ** ply                   # discounted occupancy weight
            if bkey in seen:
                occ[seen[bkey]] += w           # revisit -> accumulate occupancy, do NOT re-add board
            else:
                seen[bkey] = len(states)
                states.append(state)
                phases.append(ply)
                occ.append(w)
            key, ak = jax.random.split(key)
            state = _step(state, _eps_greedy(agent, state, ak, eps))
            if bool(state.terminated | state.truncated):
                break
            key, fk = jax.random.split(key)
            state = _step(state, foe_action(state, fk, foe))
            ply += 1
            if len(states) >= n_boards:
                break
        games += 1
    for s in states:
        assert int(s.current_player) == 0, "seat-normalization violated: current_player != 0"
    occ = np.asarray(occ, dtype=np.float64)
    occ = occ / occ.sum() if occ.sum() > 0 else occ
    return states[:n_boards], occ[:n_boards], np.asarray(phases[:n_boards]), games


# --------------------------------------------------------------------------------------
# return tensor (B,7,N) with ILLEGAL columns set to NaN so metrics drop them
# --------------------------------------------------------------------------------------
def legal_masks(states):
    return np.stack([np.array(s.legal_action_mask).astype(bool) for s in states])   # (B,7)


def compute_return_tensor_c4(agent, states, n_rollouts, chunk=64, seed=1):
    N = n_rollouts
    actions_row = jnp.arange(C4_ACTIONS, dtype=jnp.int32)
    key = jax.random.PRNGKey(seed)
    B = len(states)
    out = np.full((B, C4_ACTIONS, N), np.nan, dtype=np.float64)
    masks = legal_masks(states)
    for lo in range(0, B, chunk):
        sub = states[lo:lo + chunk]
        b = len(sub)
        pad = chunk - b
        padded = sub + [sub[-1]] * pad if pad else sub      # fixed jit shape -> no recompiles
        batched = jax.tree.map(lambda *xs: jnp.stack(xs, 0), *padded)
        actions = jnp.broadcast_to(actions_row, (chunk, C4_ACTIONS))
        key, sk = jax.random.split(key)
        keys = jax.random.split(sk, chunk * C4_ACTIONS * N).reshape(chunk, C4_ACTIONS, N, 2)
        ret = np.array(agent._compiled_batched_fn(agent.params, batched, actions, keys))  # (chunk,7,N)
        out[lo:lo + b] = ret[:b]
    out[~masks] = np.nan                                    # illegal (b,a) -> NaN (drop from stakes)
    return out


# --------------------------------------------------------------------------------------
# CCE priority (total-variation score CCE deploys) + greedy actions
# --------------------------------------------------------------------------------------
def compute_cce_and_greedy(states, returns, metric, aggregation):
    B = len(states)
    cce = np.full(B, np.nan)
    greedy = np.zeros(B, dtype=int)
    masks = legal_masks(states)
    with _nan_quiet():
        m = np.nanmean(returns, axis=2)                     # (B,7) mean return per column
    for i in range(B):
        legal_cols = [c for c in range(C4_ACTIONS) if masks[i, c]]
        if len(legal_cols) < 2:
            greedy[i] = legal_cols[0] if legal_cols else 0
            continue
        greedy[i] = int(legal_cols[int(np.nanargmax([m[i, c] for c in legal_cols]))])
        rd = {(c,): returns[i, c] for c in legal_cols}
        probs = {(c,): 1.0 for c in legal_cols}
        cce[i] = compute_consequence_metric((greedy[i],), rd, metric=metric,
                                            action_probs=probs, aggregation=aggregation)
    return cce, greedy


# --------------------------------------------------------------------------------------
# sampled one-step |TD| (no oracle): |r + γ(1-done)·max_a' Q_target(s') − Q(s,a_greedy)|
# averaged over a few foe replies (random foe is noisy). Same greedy action as CCE.
# --------------------------------------------------------------------------------------
def _q(agent, params, state):
    return np.array(agent.network.apply(params, state.observation.reshape(-1))[0])   # (7,)


def compute_abs_td_c4(agent, states, greedy_actions, foe, gamma, n_foe_replies=4, seed=2):
    key = jax.random.PRNGKey(seed)
    B = len(states)
    abs_td = np.full(B, np.nan)
    for i, s in enumerate(states):
        a = int(greedy_actions[i])
        q_sa = float(_q(agent, agent.params, s)[a])
        s1 = _step(s, jnp.int32(a))
        r1 = float(np.array(s1.rewards)[0])
        if bool(s1.terminated | s1.truncated):
            abs_td[i] = abs(r1 - q_sa)
            continue
        targets = []
        for _ in range(n_foe_replies):
            key, fk = jax.random.split(key)
            s2 = _step(s1, foe_action(s1, fk, foe))
            r = r1 + float(np.array(s2.rewards)[0])
            if bool(s2.terminated | s2.truncated):
                targets.append(r)
            else:
                legal = np.array(s2.legal_action_mask).astype(bool)
                qn = _q(agent, agent.target_params, s2)
                targets.append(r + gamma * float(np.nanmax(np.where(legal, qn, np.nan))))
        abs_td[i] = abs(float(np.mean(targets)) - q_sa)
    return abs_td
