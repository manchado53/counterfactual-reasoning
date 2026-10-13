"""
JAX single-agent adapter over JaxMARL's JaxNav.

Exposes the *same functional contract* as ``envs/frozen_lake.py`` so the CCE
machinery plugs in unchanged:

    env = JaxNavEnv()
    obs, state = env.reset(key)                                  # obs:(205,)  state: JaxNav State pytree
    obs, next_state, reward, done, info = env.step(key, state, action)   # scalars; action int in [0,15)

Why an adapter (mirrors FrozenLake's design):
  * ``step`` is JAX-pure and takes ``state`` as an argument, so counterfactual
    rollouts branch by feeding any *stored* state back into ``step`` — exactly
    like FrozenLake's ``step(key, state, action)``.
  * It calls JaxNav's ``step_env`` (the RAW transition, NO auto-reset). The
    base-class ``step`` auto-resets on episode end and would silently corrupt a
    counterfactual branch that steps a terminal action.
  * Single agent: JaxMARL's dict interface (``{"agent_0": ...}``) is unwrapped
    to scalars, matching FrozenLake's scalar API.

State vs observation — the one structural difference from FrozenLake:
  FrozenLake conflates them (state == obs == an int index). JaxNav separates
  them: the ``State`` *pytree* is needed to step/branch; the 205-vector ``obs``
  is the Q-network input. Trainers therefore carry BOTH (obs for TD updates,
  the ``State`` pytree for CCE rollouts).

Sparse goal-only reward (default): with the dense-shaping weights zeroed and
``coll_rew=0``, reward is ``+goal_rew`` on the first goal-reach and 0 otherwise.
Then ``episode_return > 0``  <=>  reached goal, so FrozenLake's ``win_rate``
metric transfers with no change, and JaxNav becomes a structural twin of
FrozenLake (collision == hole == 0-reward terminal; goal == +reward terminal).

All methods are jit/vmap-safe (JaxNav.reset/step_env/get_obs are jitted).
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from jaxmarl.environments.jaxnav import JaxNav

_AGENT = "agent_0"
N_ACTIONS = 15          # JaxNav Discrete(15): {v in 0,.5,1} x {w in +-.5,+-.25,0}
OBS_DIM = 205           # 200 LiDAR beams + v + w + goal_dist + goal_bearing + rew_lambda

# Reward-shaping weights zeroed to leave a sparse goal-only signal.
_SPARSE_REWARD_KWARGS = dict(
    weight_g=0.0,   # kill dense distance-to-goal shaping
    dt_rew=0.0,     # kill per-step time penalty
    lidar_rew=0.0,  # kill LiDAR-proximity penalty
    weight_w=0.0,   # kill angular-effort penalty (already the JaxNav default)
)


class JaxNavEnv:
    """Single-agent, discrete-action JaxNav wrapped in the FrozenLake env contract."""

    def __init__(
        self,
        scenario: Optional[str] = None,       # e.g. "SingleNav1" -> fixed hand-crafted map
        map_id: str = "Grid-Rand-Poly",       # random maps (used when scenario is None)
        map_size: Tuple[int, int] = (11, 11),
        fill: float = 0.3,                     # obstacle density for random maps
        goal_radius: float = 0.5,
        goal_rew: float = 1.0,                 # FrozenLake-style +1 on reaching the goal
        coll_rew: float = 0.0,                 # 0.0 -> collision is a 0-reward terminal (hole analog)
        max_steps: int = 200,
        sparse_reward: bool = True,
        **jaxnav_kwargs,
    ):
        rew_kwargs = dict(goal_rew=goal_rew, coll_rew=coll_rew)
        if sparse_reward:
            rew_kwargs.update(_SPARSE_REWARD_KWARGS)

        if scenario is not None:
            # Fixed singleton map (reproducible start/goal). Lazily imported so the
            # module still loads if the singleton factory API differs.
            from jaxmarl.environments.jaxnav import make_jaxnav_singleton
            self._env = make_jaxnav_singleton(
                scenario, act_type="Discrete", **rew_kwargs, **jaxnav_kwargs
            )
        else:
            self._env = JaxNav(
                num_agents=1,
                act_type="Discrete",
                map_id=map_id,
                map_params={"map_size": map_size, "fill": fill},
                goal_radius=goal_radius,
                max_steps=max_steps,
                **rew_kwargs,
                **jaxnav_kwargs,
            )

        self.scenario = scenario
        self.n_actions = N_ACTIONS
        self.obs_dim = OBS_DIM
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # FrozenLake-style functional API
    # ------------------------------------------------------------------

    def reset(self, key: jax.Array) -> Tuple[jax.Array, "State"]:
        """Reset to a fresh episode. Returns (obs:(205,), state: JaxNav State pytree)."""
        obs, state = self._env.reset(key)
        return obs[_AGENT], state

    def step(
        self,
        key: jax.Array,
        state: "State",
        action: jax.Array,
    ) -> Tuple[jax.Array, "State", jax.Array, jax.Array, dict]:
        """
        One raw transition from ``state`` under ``action`` (no auto-reset).

        Returns (obs, next_state, reward, done, info) with scalar reward/done —
        mirrors FrozenLake's ``step`` signature so trainers/rollouts are identical.
        ``key`` is accepted for signature compatibility; JaxNav transitions are
        deterministic and ignore it.
        """
        actions = {_AGENT: jnp.asarray(action, dtype=jnp.int32)}
        obs, next_state, reward, done, info = self._env.step_env(key, state, actions)
        return obs[_AGENT], next_state, reward[_AGENT], done["__all__"], info

    def get_obs(self, state: "State") -> jax.Array:
        """205-vector observation for a stored state (used inside CCE rollouts)."""
        return self._env.get_obs(state)[_AGENT]
