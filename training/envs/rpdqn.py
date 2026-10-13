"""
Neural CVRP (Capacitated Vehicle Routing Problem) -- the RP-DQN benchmark env.

Reimplementation of the instance distribution and reward used by RP-DQN
(arXiv 2104.12226, source: github.com/inboxedshoe/RP-DQN), so that a CCE replay
priority can be measured on a benchmark we did not build. Their greedy arm leaves
4.26% to optimal at n=20 -- five times the headroom our tabular routing env had.

Recovered from their source, because the paper omits both:
  * ``DataGenerator.py``  -- coords ~ U[0,1]^2; demands ~ randint(1,10) / capacity;
    capacity 30 (n<=20), 40 (n<=50), 50 (else); a FRESH instance every episode.
  * ``problem.py::update_cost`` + ``AttentionModel.py`` -- cost is the sum of leg
    lengths, negated for the reward, with ``discount_factor: 1`` in ``main.py``.
    So the undiscounted return is exactly the negative tour length.

This is NOT ``envs/cvrp.py``. That one is a tabular, BFS-enumerated, fixed
hand-placed 11-node map. This one generates a new instance per reset and never
enumerates anything.

Contract mirrors ``envs/jax_nav.py``:
    reset(key)                  -> (obs, state)
    step(key, state, action)    -> (obs, next_state, reward, done, info)
    get_obs(state)              -> obs
with ``n_actions`` / ``obs_shape`` / ``max_steps`` as env attributes.

CRITICAL: the problem instance lives INSIDE the state pytree, not on the env
object. ``dqn_vectorized.py`` resets a batch with ``jax.vmap(lambda k: env.reset(k)[1])``
and selects with ``_tree_where``, so every leaf must have a fixed shape; and a
counterfactual rollout replays a *stored* state, so it must branch on that state's
own coordinates and demands. Storing the instance as an env attribute would make
every rollout silently score the wrong problem.
"""

import jax
import jax.numpy as jnp
from flax import struct

# Node feature columns fed to the encoder, in order.
#   x, y            static coordinates
#   demand          d_i^(t) -- ZEROED once the node is served (paper eq. 8-11)
#   visited         mu_i^(t)
#   infeasible      rho_i^(t) -- demand exceeds what is left on the truck
#   is_current      tau_i^(t) -- where the vehicle is standing
#   is_depot        phi_i
#   load            c^(t), broadcast to every node so the decoder context can read it
N_FEATURES = 8
_F_X, _F_Y, _F_DEMAND, _F_VISITED, _F_INFEASIBLE, _F_CURRENT, _F_DEPOT, _F_LOAD = range(8)

# From DataGenerator.py.
_MIN_DEMAND = 1
_MAX_DEMAND = 10  # exclusive, i.e. demands are integers 1..9


def default_capacity(n_customers: int) -> float:
    """Vehicle capacity for a problem size, verbatim from their DataGenerator."""
    if n_customers <= 10:
        return 20.0
    elif n_customers <= 20:
        return 30.0
    elif n_customers <= 50:
        return 40.0
    return 50.0


@struct.dataclass
class CVRPState:
    """One CVRP episode. Every leaf is fixed-shape so this batches under vmap.

    The instance (``coords``, ``demands``) is part of the state precisely so a
    counterfactual rollout branches on the right problem.
    """
    coords: jnp.ndarray    # (n_nodes, 2) f32 -- index 0 is the depot
    demands: jnp.ndarray   # (n_nodes,)   f32 -- normalised by capacity; depot 0.0
    visited: jnp.ndarray   # (n_nodes,)   bool -- depot entry stays False
    current: jnp.ndarray   # ()           i32 -- node the vehicle is at
    load: jnp.ndarray      # ()           f32 -- remaining capacity, 1.0 == full
    step: jnp.ndarray      # ()           i32
    done: jnp.ndarray      # ()           bool


class CVRPNeuralEnv:
    """Fresh-instance CVRP with a per-leg negative-distance reward."""

    def __init__(
        self,
        n_customers: int = 20,
        capacity: float = None,
        max_steps: int = None,
    ):
        self.n_customers = int(n_customers)
        self.n_nodes = self.n_customers + 1
        self.capacity = float(capacity) if capacity is not None else default_capacity(self.n_customers)

        # Worst case is one customer per trip: n_customers legs out plus n_customers
        # returns to the depot. Demands are at most 9/capacity, so this is never
        # reached in practice -- it exists to keep the episode length static.
        self.max_steps = int(max_steps) if max_steps is not None else 2 * self.n_customers + 2

        self.n_actions = self.n_nodes
        self.obs_shape = (self.n_nodes, N_FEATURES)
        self.obs_dim = self.n_nodes * N_FEATURES  # flattened, for code that wants a scalar

    # ------------------------------------------------------------------
    # Instance generation
    # ------------------------------------------------------------------

    def reset(self, key):
        """Sample a brand-new instance. Returns ``(obs, state)``."""
        coord_key, demand_key = jax.random.split(key)

        coords = jax.random.uniform(coord_key, (self.n_nodes, 2), dtype=jnp.float32)

        raw = jax.random.randint(
            demand_key, (self.n_nodes,), _MIN_DEMAND, _MAX_DEMAND
        ).astype(jnp.float32)
        demands = (raw / self.capacity).at[0].set(0.0)  # depot has no demand

        state = CVRPState(
            coords=coords,
            demands=demands,
            visited=jnp.zeros((self.n_nodes,), dtype=bool),
            current=jnp.int32(0),
            load=jnp.float32(1.0),
            step=jnp.int32(0),
            done=jnp.bool_(False),
        )
        return self.get_obs(state), state

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_obs(self, state: CVRPState):
        """(n_nodes, N_FEATURES) node features for the encoder."""
        idx = jnp.arange(self.n_nodes)

        # Paper eq. 8-11: a served node's demand reads as 0, so the encoder cannot
        # keep attending to a stop that no longer needs anything.
        demand = jnp.where(state.visited, 0.0, state.demands)
        infeasible = (state.demands > state.load) & (~state.visited) & (idx > 0)

        return jnp.stack(
            [
                state.coords[:, 0],
                state.coords[:, 1],
                demand,
                state.visited.astype(jnp.float32),
                infeasible.astype(jnp.float32),
                (idx == state.current).astype(jnp.float32),
                (idx == 0).astype(jnp.float32),
                jnp.full((self.n_nodes,), state.load, dtype=jnp.float32),
            ],
            axis=-1,
        ).astype(jnp.float32)

    def action_mask(self, state: CVRPState):
        """(n_nodes,) bool -- True where the action is legal.

        A customer is legal if unserved and it fits in what is left on the truck.
        The depot is legal whenever the vehicle is not already standing on it, which
        is what lets the agent choose to reload early.

        From the depot the truck is full and every demand is at most 9/capacity, so
        an unserved customer is always reachable -- the episode cannot deadlock.
        Once terminal, the depot is marked legal so a masked softmax or argmax over
        the Q-values never sees an all -inf row.
        """
        idx = jnp.arange(self.n_nodes)
        customer_ok = (~state.visited) & (state.demands <= state.load) & (idx > 0)
        depot_ok = (idx == 0) & (state.current != 0)
        mask = customer_ok | depot_ok
        return jnp.where(state.done, idx == 0, mask)

    @staticmethod
    def action_mask_from_obs(obs):
        """Same mask, recovered from the observation alone.

        The rollout only ever carries ``obs``, so the Q-network has to be able to
        mask without the state pytree.
        """
        n_nodes = obs.shape[-2]
        idx = jnp.arange(n_nodes)
        visited = obs[..., _F_VISITED] > 0.5
        infeasible = obs[..., _F_INFEASIBLE] > 0.5
        is_depot = obs[..., _F_DEPOT] > 0.5
        at_depot = (obs[..., _F_CURRENT] > 0.5) & is_depot

        customer_ok = (~visited) & (~infeasible) & (idx > 0)
        # Reduce over nodes only, keeping any leading batch axes intact.
        depot_ok = is_depot & (~jnp.any(at_depot, axis=-1, keepdims=True))
        mask = customer_ok | depot_ok
        # All customers served and parked at the depot -> terminal; keep depot legal.
        any_legal = jnp.any(mask, axis=-1, keepdims=True)
        return jnp.where(any_legal, mask, is_depot)

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def step(self, key, state: CVRPState, action):
        """Drive to ``action``. Returns ``(obs, next_state, reward, done, info)``.

        ``key`` is unused -- the transition is deterministic -- but kept so the
        signature matches every other env in the repo and the shared rollout.
        """
        del key
        action = jnp.asarray(action, dtype=jnp.int32)

        reward = -jnp.linalg.norm(state.coords[action] - state.coords[state.current])

        going_to_depot = action == 0
        visited = jnp.where(going_to_depot, state.visited, state.visited.at[action].set(True))
        load = jnp.where(going_to_depot, jnp.float32(1.0), state.load - state.demands[action])

        all_served = jnp.all(visited[1:])
        step = state.step + 1
        done = (all_served & going_to_depot) | (step >= self.max_steps)

        moved = CVRPState(
            coords=state.coords,
            demands=state.demands,
            visited=visited,
            current=action,
            load=load,
            step=step,
            done=done,
        )

        # A terminal state self-loops with zero reward. The shared rollout already
        # masks post-terminal rewards, but a rollout that runs past the end must not
        # keep accumulating distance either.
        next_state = jax.tree.map(
            lambda new, old: jnp.where(state.done, old, new), moved, state
        )
        reward = jnp.where(state.done, 0.0, reward).astype(jnp.float32)
        done = jnp.logical_or(state.done, done)

        info = {'tour_len': -reward, 'all_served': all_served}
        return self.get_obs(next_state), next_state, reward, done, info
