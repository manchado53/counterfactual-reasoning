"""
JAX FrozenLake environment — exact reimplementation of Gymnasium's FrozenLake-v1.

The only behavioural difference from Gymnasium is the `desc` parameter which
accepts any N×M map string, making the grid size configurable.

Transition dynamics are identical to Gymnasium:
  - Slippery: three equally-probable outcomes [(a-1)%4, a, (a+1)%4]
  - Non-slippery: single deterministic outcome
  - Terminal states (G, H) loop to themselves

API:
    env = FrozenLakeEnv()                   # 4×4, slippery
    env = FrozenLakeEnv(map_name="8x8")
    env = FrozenLakeEnv(desc=["SFF", "FHF", "FFG"])  # custom 3×3

    obs, state = env.reset(key)
    obs, state, reward, done, info = env.step(key, state, action)

Both reset() and step() are JAX-pure: no Python side-effects, safe for
jax.jit and jax.vmap.

env.P mirrors Gymnasium's env.P structure:
    env.P[state][action] = [(prob, next_state, reward, terminated), ...]
"""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

MAPS: Dict[str, List[str]] = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


def _move(row: int, col: int, action: int, nrows: int, ncols: int) -> Tuple[int, int]:
    """Apply action with boundary clamping — matches Gymnasium exactly."""
    if action == LEFT:
        col = max(0, col - 1)
    elif action == DOWN:
        row = min(nrows - 1, row + 1)
    elif action == RIGHT:
        col = min(ncols - 1, col + 1)
    elif action == UP:
        row = max(0, row - 1)
    return row, col


class FrozenLakeEnv:
    """
    JAX FrozenLake. Transition table is precomputed at init and stored as
    JAX arrays for O(1) JIT-compatible step.
    """

    def __init__(
        self,
        map_name: str = "4x4",
        desc: Optional[List[str]] = None,
        is_slippery: bool = True,
    ):
        if desc is not None:
            self.desc = desc
        else:
            if map_name not in MAPS:
                raise ValueError(f"Unknown map_name '{map_name}'. Use '4x4', '8x8', or pass desc=.")
            self.desc = MAPS[map_name]

        self.nrows = len(self.desc)
        self.ncols = len(self.desc[0])
        self.n_states = self.nrows * self.ncols
        self.n_actions = 4
        self.is_slippery = is_slippery

        self.start_states: List[int] = [
            r * self.ncols + c
            for r, row in enumerate(self.desc)
            for c, tile in enumerate(row)
            if tile == "S"
        ]

        self._build_transition_table()

    # ------------------------------------------------------------------
    # Transition table construction
    # ------------------------------------------------------------------

    def _build_transition_table(self) -> None:
        """
        Builds self.P (Python dict, mirrors Gymnasium) and JAX arrays
        (next_states, rewards, dones) for use inside jit/vmap.

        Table shape: (n_states, n_actions, 3)
          - dim 2 always has 3 outcome slots
          - slippery: 3 distinct outcomes, each p=1/3
          - non-slippery: all 3 slots hold the same deterministic outcome
        """
        n = self.n_states
        a = self.n_actions
        k = 3

        next_s_np = np.zeros((n, a, k), dtype=np.int32)
        rew_np = np.zeros((n, a, k), dtype=np.float32)
        done_np = np.zeros((n, a, k), dtype=bool)

        P: Dict = {}

        for s in range(n):
            row, col = divmod(s, self.ncols)
            tile = self.desc[row][col]
            P[s] = {}

            for act in range(a):
                if tile in ("G", "H"):
                    # Gymnasium gives reward=0 for terminal state loops,
                    # even for the goal tile. The 1.0 reward is only on the
                    # transition that first lands on G.
                    for ki in range(k):
                        next_s_np[s, act, ki] = s
                        rew_np[s, act, ki] = 0.0
                        done_np[s, act, ki] = True
                    P[s][act] = [(1.0, s, 0.0, True)]

                elif self.is_slippery:
                    outcomes = [(act - 1) % 4, act, (act + 1) % 4]
                    P[s][act] = []
                    for ki, b in enumerate(outcomes):
                        nr, nc = _move(row, col, b, self.nrows, self.ncols)
                        ns = nr * self.ncols + nc
                        nt = self.desc[nr][nc]
                        r = 1.0 if nt == "G" else 0.0
                        d = nt in ("G", "H")
                        next_s_np[s, act, ki] = ns
                        rew_np[s, act, ki] = r
                        done_np[s, act, ki] = d
                        P[s][act].append((1.0 / 3.0, ns, r, d))

                else:
                    nr, nc = _move(row, col, act, self.nrows, self.ncols)
                    ns = nr * self.ncols + nc
                    nt = self.desc[nr][nc]
                    r = 1.0 if nt == "G" else 0.0
                    d = nt in ("G", "H")
                    for ki in range(k):
                        next_s_np[s, act, ki] = ns
                        rew_np[s, act, ki] = r
                        done_np[s, act, ki] = d
                    P[s][act] = [(1.0, ns, r, d)]

        self.P = P
        self.next_states = jnp.array(next_s_np)
        self.rewards = jnp.array(rew_np)
        self.dones = jnp.array(done_np)

    # ------------------------------------------------------------------
    # JAX API
    # ------------------------------------------------------------------

    def reset(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Reset to a start state.

        Returns (obs, state) where obs == state (integer position).
        """
        if len(self.start_states) == 1:
            state = jnp.int32(self.start_states[0])
        else:
            idx = jax.random.randint(key, shape=(), minval=0, maxval=len(self.start_states))
            state = jnp.array(self.start_states, dtype=jnp.int32)[idx]
        return state, state

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
        """
        Take one step.

        Args:
            key:    JAX PRNGKey (consumed for slippery sampling)
            state:  scalar int32 — current state
            action: scalar int32 — action in {0,1,2,3}

        Returns:
            obs, next_state, reward, done, info
        """
        if self.is_slippery:
            outcome = jax.random.randint(key, shape=(), minval=0, maxval=3)
        else:
            outcome = jnp.int32(0)

        next_state = self.next_states[state, action, outcome]
        reward = self.rewards[state, action, outcome]
        done = self.dones[state, action, outcome]

        return next_state, next_state, reward, done, {}
