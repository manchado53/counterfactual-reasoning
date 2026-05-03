# Plan: Gardner Chess (pgx) — Implementation & Verification

## Context
Adding Gardner chess (5×5) as a second environment to demonstrate Algorithm 2 (consequence-weighted DQN) generalizes beyond SMAX. Uses `pgx` library — JAX-native, already installed or `pip install pgx`. Agent plays as white against the pre-trained pgx baseline (`gardner_chess_v0`, ~1000 Elo). Pure sparse rewards. Single-episode training loop (not vectorized — consequence scoring is the bottleneck and already vectorized).

---

## Files to Create (in order)

```
src/counterfactual_rl/training/pgx/
├── __init__.py
└── dqn_jax/
    ├── __init__.py
    ├── chess_env.py
    ├── policies.py
    ├── dqn.py
    ├── consequence_dqn.py
    ├── config.py
    └── train.py
```

Also update:
- `src/counterfactual_rl/training/__init__.py` — add `from . import pgx`

---

## Step 1 — Package scaffolding

**`src/counterfactual_rl/training/pgx/__init__.py`**
```python
"""PGX training implementations."""
```

**`src/counterfactual_rl/training/pgx/dqn_jax/__init__.py`**
```python
"""JAX DQN implementation for pgx Gardner chess."""
from .chess_env import GardnerChessEnv
from .dqn import ChessDQN
from .consequence_dqn import ChessConsequenceDQN
__all__ = ['GardnerChessEnv', 'ChessDQN', 'ChessConsequenceDQN']
```

**`src/counterfactual_rl/training/__init__.py`** — append `from . import pgx`

**Verify step 1:**
```bash
python -c "from counterfactual_rl.training import pgx; print('scaffold OK')"
```

---

## Step 2 — `chess_env.py`

Single-agent MDP wrapper. Key invariant: every state returned to the DQN has `current_player == 0` (white to move). The wrapper executes white's move then the opponent's response before returning.

```python
import numpy as np
import jax, jax.numpy as jnp, pgx

CHESS_OBS_FLAT = 5 * 5 * 115   # 2875
CHESS_ACTIONS  = 1225

class GardnerChessEnv:
    def __init__(self, seed=0):
        self._env      = pgx.make("gardner_chess")
        self._opponent = pgx.make_baseline_model("gardner_chess_v0")
        self._rng      = jax.random.PRNGKey(seed)

    def reset(self, key):
        """Returns (obs: (2875,) float32, state: pgx.State)"""
        state = self._env.init(key)
        return self._obs(state), state

    def step(self, state, action: int):
        """
        Execute white's action, then opponent's response.
        Returns (obs, next_state, reward: float, done: bool)
        next_state always has current_player == 0.
        """
        # White moves
        s1 = self._env.step(state, jnp.int32(action))
        r  = float(np.array(s1.rewards[0]))
        d  = bool(np.array(s1.terminated | s1.truncated))
        if d:
            return self._obs(s1), s1, r, d

        # Opponent responds
        self._rng, k = jax.random.split(self._rng)
        logits, _    = self._opponent(s1.observation)
        masked       = jnp.where(s1.legal_action_mask, logits, -jnp.inf)
        black_action = jax.random.categorical(k, masked)
        s2 = self._env.step(s1, black_action)

        r += float(np.array(s2.rewards[0]))
        d  = bool(np.array(s2.terminated | s2.truncated))
        return self._obs(s2), s2, r, d

    def get_legal_mask(self, state) -> np.ndarray:
        """Returns (1, 1225) bool — (n_agents, actions_per_agent) convention."""
        return np.array(state.legal_action_mask)[np.newaxis, :]

    @staticmethod
    def _obs(state) -> np.ndarray:
        return np.array(state.observation, dtype=np.float32).reshape(-1)
```

**Reward note:** verify `state.rewards[0]` is +1 white win / -1 white loss / 0 draw+non-terminal
against https://www.sotets.uk/pgx/gardner_chess/ before trusting win/loss metrics.

**Verify step 2:**
```bash
python -c "
import jax, numpy as np
from counterfactual_rl.training.pgx.dqn_jax.chess_env import GardnerChessEnv
env = GardnerChessEnv(0)
obs, state = env.reset(jax.random.PRNGKey(0))
assert obs.shape == (2875,), obs.shape
assert env.get_legal_mask(state).shape == (1, 1225)
valid = np.where(env.get_legal_mask(state)[0])[0]
obs2, s2, r, done = env.step(state, int(valid[0]))
assert obs2.shape == (2875,)
assert s2.current_player == 0, 'invariant broken: must be white to move'
print('chess_env OK | done=' + str(done) + ' reward=' + str(r))
"
```

---

## Step 3 — `policies.py`

Conv front-end on (5,5,115) spatial board + MLP. Output `(1, 1225)` to match
`(n_agents, actions_per_agent)` convention used by the rest of the code.

```python
import jax.numpy as jnp, flax.linen as nn

class ChessQNetwork(nn.Module):
    hidden_dim: int = 512
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, flat_obs):               # (..., 2875)
        x = flat_obs.reshape((*flat_obs.shape[:-1], 5, 5, 115))
        x = nn.Conv(32, (3,3), padding='SAME')(x); x = nn.relu(x)
        x = nn.Conv(64, (3,3), padding='SAME')(x); x = nn.relu(x)
        x = nn.Conv(64, (1,1))(x);               x = nn.relu(x)
        x = x.reshape((*flat_obs.shape[:-1], -1))   # (..., 1600)
        x = nn.Dense(self.hidden_dim)(x); x = nn.relu(x)
        x = nn.Dense(256)(x);             x = nn.relu(x)
        q = nn.Dense(1225)(x)                       # (..., 1225)
        return jnp.expand_dims(q, axis=-2)          # (..., 1, 1225)
```

**Verify step 3:**
```bash
python -c "
import jax, jax.numpy as jnp
from counterfactual_rl.training.pgx.dqn_jax.policies import ChessQNetwork
net  = ChessQNetwork()
p    = net.init(jax.random.PRNGKey(0), jnp.zeros((2875,)))
out  = net.apply(p, jnp.zeros((2875,)))
assert out.shape == (1, 1225), out.shape
outb = jax.vmap(net.apply, in_axes=(None,0))(p, jnp.zeros((32, 2875)))
assert outb.shape == (32, 1, 1225), outb.shape
print('ChessQNetwork OK')
"
```

---

## Step 4 — `dqn.py`

Standalone `ChessDQN` — does NOT inherit from SMAX DQN (avoids jaxmarl dependency chain).
Mirrors the SMAX DQN structure but adapted for single-agent chess.

**Key differences from SMAX `dqn.py`:**

| | SMAX | Chess |
|---|---|---|
| Network init dummy | `jnp.zeros((obs_dim,))` | `jnp.zeros((2875,))` |
| `greedy_action` output | `(n_agents,)` | `(1,)` |
| Q-update actions shape | `(B, n_agents)` | `(B, 1)` |
| No `get_global_state`, `get_action_masks` utils | — | use env directly |
| `jax_obs` stored in buffer | yes | **no** (pgx state contains obs) |

**`env_info` dict for chess:**
```python
{
    'obs_dim': 2875, 'num_agents': 1, 'actions_per_agent': 1225,
    'agent_names': ['white'], 'obs_type': 'flat', 'env_name': 'gardner_chess',
}
```

**Q-update shapes** (n_agents=1 makes sums trivially correct):
```
q_values:   (B, 1, 1225)
q_taken:    (B,)           <- gather on axis=-1 then squeeze then sum(axis=-1)
max_next_q: (B,)           <- max(axis=-1) then sum(axis=-1)
```

**`make_policy_fn`** (used inside JIT rollouts):
```python
def policy_fn(obs_flat, legal_mask_1d):
    # obs_flat: (2875,), legal_mask_1d: (1225,)
    q = network.apply(params, obs_flat)          # (1, 1225)
    return jnp.argmax(jnp.where(legal_mask_1d, q[0], -jnp.inf))  # scalar int32
```

Reuses `PrioritizedReplayBuffer` from `counterfactual_rl.training.smax.shared.buffers`.

`evaluate()` returns `{'win_rate', 'draw_rate', 'loss_rate', 'avg_return', 'avg_length'}` —
classify outcome from cumulative episode return (>0 win, <0 loss, 0 draw).

**Verify step 4:**
```bash
python -c "
from counterfactual_rl.training.pgx.dqn_jax.train import create_chess_env
from counterfactual_rl.training.pgx.dqn_jax.dqn import ChessDQN
env, key, info = create_chess_env(0)
agent = ChessDQN(env, info, {'n_episodes':10,'eval_interval':None,'M':500,'B':8})
agent.learn(n_episodes=10)
metrics = agent.evaluate(n_episodes=5)
assert 'win_rate' in metrics
print('ChessDQN OK | metrics:', metrics)
"
```

---

## Step 5 — `consequence_dqn.py`

Extends `ChessDQN`. Algorithm 2 logic identical to SMAX; only the rollout engine changes.

**Reuses (direct imports, no changes):**
- `ConsequenceReplayBuffer` from `counterfactual_rl.training.smax.dqn_jax.consequence_buffers`
- `compute_consequence_metric` from `counterfactual_rl.analysis.metrics`
- `beam_search_top_k_joint_actions` from `counterfactual_rl.utils.action_selection`
- `MetricsLogger` from `counterfactual_rl.training.smax.shared.metrics`
- `TrainingTimer` from `counterfactual_rl.training.smax.shared.timing`

**`_build_batched_rollout_fn`** — chess rollout:

Each "horizon step" = white move + opponent response (one full move pair).
State invariant: carry state always has `current_player == 0`.

```python
def single_rollout(params, state, first_action, rng_key):
    # state: pgx.State, current_player==0
    # first_action: int32 scalar

    # White's first move
    s1    = pgx_env.step(state, first_action)
    r1    = s1.rewards[0]
    done1 = s1.terminated | s1.truncated

    # Opponent responds (skip if done)
    rng_key, k = jax.random.split(rng_key)
    s2    = jax.lax.cond(done1, lambda: s1, lambda: _opp_step(s1, k))
    r2    = jnp.where(done1, 0.0, s2.rewards[0])
    done2 = done1 | s2.terminated | s2.truncated

    def scan_step(carry, _):
        s, key, cum, disc, done = carry
        key, ok = jax.random.split(key)
        # White greedy
        aw = jax.lax.cond(done, lambda: jnp.int32(0),
             lambda: policy_fn(params, s.observation.reshape(-1), s.legal_action_mask))
        sw = jax.lax.cond(done, lambda: s, lambda: pgx_env.step(s, aw))
        rw = jnp.where(done, 0.0, sw.rewards[0])
        dw = done | sw.terminated | sw.truncated
        # Opponent
        so = jax.lax.cond(dw, lambda: sw, lambda: _opp_step(sw, ok))
        ro = jnp.where(dw, 0.0, so.rewards[0])
        do = dw | so.terminated | so.truncated
        return (so, key, cum + disc*(rw+ro), jnp.where(do, disc, disc*gamma), do), None

    init = (s2, rng_key, r1+r2, jnp.float32(gamma), done2)
    (_, _, cum, _, _), _ = jax.lax.scan(scan_step, init, None, length=horizon-1)
    return cum

# Triple vmap
f = jax.vmap(single_rollout, in_axes=(None,None,None,0))  # N rollouts
f = jax.vmap(f,              in_axes=(None,None,0,0))      # K actions
f = jax.vmap(f,              in_axes=(None,0,0,0))         # B transitions
self._compiled_batched_fn = jax.jit(f)
# (params, states[B], actions[B,K], keys[B,K,N,2]) -> (B,K,N)
```

**`_score_buffer_transitions`** key chess differences from SMAX:
```python
# Valid actions from stored pgx state (n_agents=1)
legal_mask = np.array(jax_state.legal_action_mask)
valid_actions_wrapped = [[j for j,v in enumerate(legal_mask) if v]]

actual_action = (int(transition['a'][0]),)  # 1-tuple (beam_search uses tuples)

# actions_array: (B, K) scalars — NOT (B, K, 1)
actions_array = jnp.array([[a[0] for a in row] for row in all_actions], dtype=jnp.int32)
# keys_array, compute_consequence_metric call: identical to SMAX
```

**Verify step 5:**
```bash
python -c "
from counterfactual_rl.training.pgx.dqn_jax.train import create_chess_env
from counterfactual_rl.training.pgx.dqn_jax.consequence_dqn import ChessConsequenceDQN
env, key, info = create_chess_env(0)
agent = ChessConsequenceDQN(env, info, {
    'n_episodes':15,'eval_interval':None,'M':500,'B':8,
    'n_score_sample':8,'cf_top_k':3,'cf_n_rollouts':4,'cf_horizon':2,'score_interval':1,
})
agent.learn(n_episodes=15)
# Must see 'Compiling batched rollout function (one-time cost)...' exactly once
print('ChessConsequenceDQN OK')
"

# Verify rollout output shape directly
python -c "
import jax, jax.numpy as jnp
from counterfactual_rl.training.pgx.dqn_jax.train import create_chess_env
from counterfactual_rl.training.pgx.dqn_jax.consequence_dqn import ChessConsequenceDQN
env, key, info = create_chess_env(0)
agent = ChessConsequenceDQN(env, info, {'n_episodes':1,'M':50,'B':4,'n_score_sample':4,'cf_top_k':2,'cf_n_rollouts':3,'cf_horizon':2})
agent._build_batched_rollout_fn()
_, state = env.reset(jax.random.PRNGKey(0))
B, K, N = 2, 2, 3
sb  = jax.tree.map(lambda *xs: jnp.stack(xs), *[state]*B)
act = jnp.zeros((B,K), dtype=jnp.int32)
ks  = jax.random.split(jax.random.PRNGKey(0), B*K*N).reshape(B,K,N,2)
out = agent._compiled_batched_fn(agent.params, sb, act, ks)
assert out.shape == (B,K,N), out.shape
print('rollout shape OK:', out.shape)
"
```

---

## Step 6 — `config.py`

```python
DEFAULT_CHESS_CONFIG = {
    'seed': 0, 'env_name': 'gardner_chess',
    'gamma': 0.99,              # sparse reward -> need long horizon credit
    'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay_episodes': 20000,
    'alpha': 0.0001, 'hidden_dim': 512, 'use_layer_norm': True,
    'M': 200000, 'B': 64,
    'C': 1000, 'n_steps_for_Q_update': 4,
    'PER_parameters': {'eps': 0.01, 'beta': 0.4, 'maximum_priority': 1.0},
    'n_episodes': 100000, 'save_every': 1000,
    'eval_interval': 500, 'eval_episodes': 50,
    'algorithm': 'consequence-dqn',
    'mu': 0.5, 'priority_mixing': 'additive', 'mu_c': 1.0, 'mu_delta': 1.0,
    'score_interval': 200, 'n_score_sample': 128,
    'consequence_metric': 'wasserstein', 'consequence_aggregation': 'weighted_mean',
    'cf_horizon': 10,           # each step = white+black pair -> 10 = 20 half-moves
    'cf_n_rollouts': 16,        # fewer: opponent inference adds GPU cost per rollout
    'cf_top_k': 10, 'cf_gamma': 0.99,
    'diagnostics_enabled': False, 'diagnostics_plot_interval': 100,
}
```

**Verify step 6:**
```bash
python -c "
from counterfactual_rl.training.pgx.dqn_jax.config import DEFAULT_CHESS_CONFIG
assert DEFAULT_CHESS_CONFIG['gamma'] == 0.99
assert DEFAULT_CHESS_CONFIG['cf_horizon'] == 10
print('config OK | keys:', list(DEFAULT_CHESS_CONFIG.keys()))
"
```

---

## Step 7 — `train.py`

Entry point mirroring `smax/shared/train.py` pattern. Reads `CONFIG_OVERRIDES_B64` /
`CONFIG_OVERRIDES` env vars.

```python
def create_chess_env(seed=0):
    env = GardnerChessEnv(seed=seed)
    env_info = {'obs_dim': 2875, 'num_agents': 1, 'actions_per_agent': 1225,
                'agent_names': ['white'], 'obs_type': 'flat', 'env_name': 'gardner_chess'}
    return env, jax.random.PRNGKey(seed), env_info

def main():
    config = DEFAULT_CHESS_CONFIG.copy()
    # load CONFIG_OVERRIDES env var (same pattern as SMAX train.py)
    env, key, env_info = create_chess_env(seed=config['seed'])
    AgentClass = ChessConsequenceDQN if config['algorithm'] == 'consequence-dqn' else ChessDQN
    agent = AgentClass(env, env_info, config=config)
    agent.learn()
```

**Verify step 7:**
```bash
python -c "
import os
os.environ['CONFIG_OVERRIDES'] = '{\"n_episodes\": 5, \"eval_interval\": null}'
from counterfactual_rl.training.pgx.dqn_jax.train import main
main()
print('train.py entry point OK')
"
```

---

## Final end-to-end check

```bash
python -c "
import os
os.environ['CONFIG_OVERRIDES'] = '{\"n_episodes\": 20, \"eval_interval\": 10, \"eval_episodes\": 5}'
from counterfactual_rl.training.pgx.dqn_jax.train import main
main()
# Expected: win_rate/draw_rate/loss_rate printed, no crashes
"
```

---

## Critical Implementation Notes

1. **`jnp.expand_dims(q, axis=-2)`** — use this, NOT `q[..., jnp.newaxis, :]` (wrong shape)
2. **`lax.cond` branches must return identical pytree structure** — both branches return `pgx.State`
3. **`current_player == 0` invariant** — every state in the buffer is white's turn; guaranteed by wrapper
4. **`jax_obs` not stored** — `buffer.add(transition, jax_state=saved_state)` only, no `jax_obs`
5. **`actions_array` is `(B, K)` not `(B, K, 1)`** — chess actions are scalars, not per-agent arrays
6. **opponent model is JAX-traceable** — safe to call inside `jax.jit` / `jax.vmap`
