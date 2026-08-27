"""
JAX Routing environment (TSP / CVRP) — a logistics testbed for CCE.

Design goal: a THIRD good-oracle environment for CCE, after FrozenLake and DoorKey,
in a genuinely different domain — vehicle routing (logistics) rather than a grid of
holes or a gating puzzle. Like those two, this is OUR reimplementation, not an import
of Jumanji, so the Claim-1 oracle has an enumerable Gymnasium-style transition table
`env.P`, and there are zero new dependencies.

Why routing fits CCE (see the environment brief):
  - Deterministic transitions — CCE's proven sweet spot (the graded-slip knife-edge).
  - Pivotal decisions — in CVRP the capacity cliff ("reload now, or squeeze one more
    stop?") is a genuinely consequential, catastrophic-if-wrong choice.
  - Exact oracle — on a fixed small instance, dynamic programming gives the optimal
    completion from any state, so per-decision regret is ground truth.

TWO MODES
    TSP  (capacity=None) : visit every customer once, unlimited load. The stepping
                           stone used to validate env + oracle + tests.
    CVRP (capacity=int)  : the real problem — the "C" is the load limit. Customers have
                           demands; the vehicle fills up and must return to the depot to
                           reload before continuing. Remove the limit and CVRP collapses
                           back into TSP, so the limit is the defining feature.

State (Markov), enumerated by BFS and indexed to a single int (obs == state == index):
    (current_node, visited_mask, remaining_capacity)
with remaining_capacity held constant in TSP mode. The integer index means the replay
buffer, the CCE rollout, and the oracle reuse the FrozenLake plumbing UNCHANGED.

TWO additions FrozenLake/DoorKey never needed:
  1. `action_masks` (n_states, n_actions) bool — a served stop cannot be revisited, and
     in CVRP a stop whose demand exceeds the remaining load is temporarily unreachable.
     The trainer MUST apply this at the argmax, during exploration, and over the CCE
     counterfactual action set (illegal Q/score -> -inf).
  2. `state_features` (n_states, feature_dim) float — routing has far more states than a
     grid (thousands, not dozens), so a one-hot state encoding would force the network to
     learn every state independently. Features (one-hot current node + visited bits +
     remaining-load fraction) let it generalize. The buffer still stores the integer state;
     the network just looks its features up by index.

Nodes: index 0 = depot, 1..N = customers. Action a = "travel to node a"; n_actions = N+1.
Reward: r = -euclidean(current, chosen)  (dense; shorter legs -> higher return).
Termination: at the depot with every customer served.

API (identical to FrozenLakeEnv / DoorKeyEnv):
    env = CVRPEnv()                      # 10 customers, capacity 10 -> real CVRP
    env = CVRPEnv(capacity=None)         # TSP mode (no load limit)
    obs, state = env.reset(key)          # obs == state (int32 index)
    obs, state, reward, done, info = env.step(key, state, action)
    env.P[state][action] = [(prob, next_state, reward, terminated), ...]

Both reset() and step() are JAX-pure: safe for jax.jit and jax.vmap.
"""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Default fixed instance: depot (index 0) + 10 customers.
# Total demand 24 against capacity 10 -> the tour needs three loads, so the capacity
# cliff is exercised repeatedly rather than being a corner case.
DEFAULT_XY = np.array(
    [
        [0.500, 0.520],  # 0  depot
        [0.207, 0.288],  # 1  C1
        [0.366, 0.183],  # 2  C2
        [0.573, 0.212],  # 3  C3
        [0.744, 0.317],  # 4  C4
        [0.854, 0.548],  # 5  C5
        [0.732, 0.779],  # 6  C6
        [0.549, 0.846],  # 7  C7
        [0.366, 0.808],  # 8  C8
        [0.201, 0.692],  # 9  C9
        [0.183, 0.490],  # 10 C10
    ],
    dtype=np.float32,
)
DEFAULT_DEMAND = np.array([0, 3, 2, 4, 2, 1, 3, 2, 3, 2, 2], dtype=np.int32)
DEFAULT_CAPACITY = 10

# Small instance (depot + 5 customers) for fast sanity checks and brute-forceable tests.
SMALL_XY = np.array(
    [
        [0.50, 0.50],  # depot
        [0.10, 0.20], [0.85, 0.15], [0.90, 0.70], [0.40, 0.95], [0.05, 0.65],
    ],
    dtype=np.float32,
)
SMALL_DEMAND = np.array([0, 3, 3, 3, 3, 3], dtype=np.int32)
SMALL_CAPACITY = 6

INSTANCES = {
    "default": {"xy": DEFAULT_XY, "demand": DEFAULT_DEMAND, "capacity": DEFAULT_CAPACITY},
    "small": {"xy": SMALL_XY, "demand": SMALL_DEMAND, "capacity": SMALL_CAPACITY},
}

DEPOT = 0


class CVRPEnv:
    """
    Tabular JAX routing env. Reachable states are enumerated at init and transitions
    precomputed into JAX arrays for an O(1) jit/vmap-safe step, plus a Gymnasium-style
    `env.P` dict and an `action_masks` array for the exact oracle and the masked DQN.
    """

    def __init__(
        self,
        node_xy: Optional[np.ndarray] = None,
        demand: Optional[np.ndarray] = None,
        capacity: Optional[int] = DEFAULT_CAPACITY,
        travel_noise: float = 0.0,
    ):
        """
        capacity:
            int  -> CVRP mode (the real problem): finite load, depot reloads.
            None -> TSP mode: no load limit. Stepping stone for validation.

        travel_noise: TRAFFIC. Multiplies each leg's cost by max(0, 1 + travel_noise*z),
            z ~ N(0,1) — the drive takes longer or shorter than planned.
              - 0.0 -> deterministic. Used for CLAIM 2 (sample efficiency), the regime
                where CCE's replay advantage lives (the graded-slip knife-edge).
              - >0  -> stochastic. Required for CLAIM 1, because CCE's total-variation
                score is DEGENERATE under determinism: greedy rollouts in a deterministic
                env give point-mass return distributions, so TV collapses to a coarse 0/1
                and C(s) is constant. FrozenLake's Claim 1 used the SLIPPERY map and
                DoorKey uses slip_prob for exactly this reason.

            The noise is zero-mean in the multiplier, so EXPECTED rewards — and therefore
            env.P, the exact DP oracle, and the optimal plan — are UNCHANGED. Transitions
            are untouched too (only the cost of a leg varies), so a policy trained on the
            deterministic env remains valid on the noisy one. That keeps the Claim-1 ground
            truth exact while giving the rollouts something to vary.
        """
        self.node_xy = DEFAULT_XY if node_xy is None else np.asarray(node_xy, np.float32)
        self.n_nodes = int(self.node_xy.shape[0])
        self.n_customers = self.n_nodes - 1
        self.n_actions = self.n_nodes  # action a == "go to node a" (0 = depot)

        if demand is None:
            demand = DEFAULT_DEMAND if self.n_nodes == DEFAULT_XY.shape[0] else np.ones(
                self.n_nodes, dtype=np.int32
            )
        self.demand = np.asarray(demand, np.int32).copy()
        self.demand[DEPOT] = 0
        if self.demand.shape[0] != self.n_nodes:
            raise ValueError("demand must have one entry per node (including the depot).")

        if travel_noise < 0.0:
            raise ValueError(f"travel_noise must be >= 0, got {travel_noise}")
        self.travel_noise = float(travel_noise)
        self.is_stochastic = self.travel_noise > 0.0

        self.capacity = capacity
        self.is_capacitated = capacity is not None
        if self.is_capacitated:
            if capacity <= 0:
                raise ValueError(f"capacity must be positive, got {capacity}")
            too_big = np.where(self.demand > capacity)[0]
            if too_big.size:
                raise ValueError(
                    f"infeasible instance: customers {too_big.tolist()} have demand greater "
                    f"than capacity {capacity}; they could never be served."
                )

        # Pairwise euclidean distances (depot is index 0).
        diff = self.node_xy[:, None, :] - self.node_xy[None, :, :]
        self.dist = np.sqrt((diff**2).sum(-1)).astype(np.float32)
        self.max_dist = float(self.dist.max())
        # Value stored for masked/illegal actions — never taken once masks are applied,
        # but defined and discouraging in case a caller ignores the mask.
        self._illegal_penalty = -2.0 * self.max_dist

        self._build()

    # ------------------------------------------------------------------
    # Pure-python dynamics (used to BFS-enumerate and to build the tables)
    # ------------------------------------------------------------------

    def _full_mask(self) -> int:
        return (1 << self.n_customers) - 1

    def _initial_cap(self) -> int:
        return int(self.capacity) if self.is_capacitated else 0

    def _is_terminal(self, s: Tuple[int, int, int]) -> bool:
        """Terminal iff back at the depot with every customer served."""
        cur, mask, _ = s
        return cur == DEPOT and mask == self._full_mask()

    def _legal_actions(self, s: Tuple[int, int, int]) -> List[int]:
        cur, mask, cap = s
        if self._is_terminal(s):
            return []

        legal: List[int] = []
        for j in range(1, self.n_nodes):
            if (mask >> (j - 1)) & 1:
                continue                                   # already served
            if self.is_capacitated and self.demand[j] > cap:
                continue                                   # will not fit in the current load
            legal.append(j)

        if cur != DEPOT:
            if self.is_capacitated:
                # Returning to the depot is always available from a customer: it reloads,
                # and when everything is served it closes the tour. This is what creates
                # the pivotal "reload now, or squeeze one more stop?" decision.
                legal.append(DEPOT)
            elif mask == self._full_mask():
                # TSP: the depot is only for closing the completed tour.
                legal.append(DEPOT)
        return sorted(legal)

    def _next(self, s: Tuple[int, int, int], a: int):
        """Deterministic transition for a LEGAL action. Returns (next_state, reward, done)."""
        cur, mask, cap = s
        reward = -float(self.dist[cur, a])
        if a == DEPOT:
            done = mask == self._full_mask()
            # Reload on arrival (a no-op when the tour ends here).
            return (DEPOT, mask, self._initial_cap()), reward, done
        new_mask = mask | (1 << (a - 1))
        new_cap = cap - int(self.demand[a]) if self.is_capacitated else cap
        return (a, new_mask, new_cap), reward, False

    # ------------------------------------------------------------------
    # Enumeration + table construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """
        BFS from the start state to enumerate reachable states, index each to a flat int,
        then build:
          - self.P              : {s: {a: [(prob, s', r, term), ...]}}  (1 outcome; deterministic)
          - self.next_states/rewards/dones : (n_states, n_actions, 1)
          - self.action_masks   : (n_states, n_actions) bool
          - self.state_features : (n_states, feature_dim) float32
          - self.non_terminal   : reachable non-terminal state indices (for the oracle)
          - self.state_current / state_mask / state_cap : per-state decoding
        """
        start = (DEPOT, 0, self._initial_cap())
        index: Dict[Tuple[int, int, int], int] = {start: 0}
        order: List[Tuple[int, int, int]] = [start]
        frontier = [start]
        while frontier:
            s = frontier.pop()
            for a in self._legal_actions(s):
                ns, _, _ = self._next(s, a)
                if ns not in index:
                    index[ns] = len(order)
                    order.append(ns)
                    frontier.append(ns)

        n = len(order)
        self.n_states = n
        self._index = index
        self._order = order
        self.start_states = [index[start]]

        K = 1  # deterministic: one outcome slot (mirrors the FrozenLake non-slip layout)
        self.n_outcomes = K
        next_s = np.zeros((n, self.n_actions, K), dtype=np.int32)
        rew = np.zeros((n, self.n_actions, K), dtype=np.float32)
        done_a = np.zeros((n, self.n_actions, K), dtype=bool)
        masks = np.zeros((n, self.n_actions), dtype=bool)
        state_current = np.zeros(n, dtype=np.int32)
        state_mask = np.zeros(n, dtype=np.int64)
        state_cap = np.zeros(n, dtype=np.int32)

        P: Dict = {}
        non_terminal: List[int] = []

        for s, si in index.items():
            cur, mask, cap = s
            state_current[si] = cur
            state_mask[si] = mask
            state_cap[si] = cap
            terminal = self._is_terminal(s)
            if not terminal:
                non_terminal.append(si)
            legal = set(self._legal_actions(s))
            P[si] = {}
            for a in range(self.n_actions):
                if terminal:
                    # Absorbing: self-loop, reward 0, done True (like FrozenLake's goal).
                    next_s[si, a, 0] = si
                    rew[si, a, 0] = 0.0
                    done_a[si, a, 0] = True
                    P[si][a] = [(1.0, si, 0.0, True)]
                elif a in legal:
                    ns, r, d = self._next(s, a)
                    nsi = index[ns]
                    masks[si, a] = True
                    next_s[si, a, 0] = nsi
                    rew[si, a, 0] = r
                    done_a[si, a, 0] = d
                    P[si][a] = [(1.0, nsi, float(r), bool(d))]
                else:
                    # Illegal (served / unaffordable / depot-hold): self-loop + penalty, masked.
                    next_s[si, a, 0] = si
                    rew[si, a, 0] = self._illegal_penalty
                    done_a[si, a, 0] = False
                    P[si][a] = [(1.0, si, float(self._illegal_penalty), False)]

        self.P = P
        self.non_terminal = sorted(non_terminal)
        self.next_states = jnp.asarray(next_s)
        self.rewards = jnp.asarray(rew)
        self.dones = jnp.asarray(done_a)
        self.action_masks = jnp.asarray(masks)
        self.state_current = jnp.asarray(state_current)
        # numpy copies for python-side use (oracle, plotting, tests)
        self.state_current_np = state_current
        self.state_mask = state_mask
        self.state_cap = state_cap

        self._build_features(state_current, state_mask, state_cap)

    def _build_features(self, cur: np.ndarray, mask: np.ndarray, cap: np.ndarray) -> None:
        """
        Per-state observation features, looked up by integer state index.

        [ one-hot current node (n_nodes) | visited bits (n_customers) | load fraction (1) ]

        Routing has thousands of states, so a one-hot state encoding would make the
        network memorize each one. These features share structure across states, and stay
        exactly Markov (they are a bijection of the enumerated state tuple).
        """
        n = self.n_states
        onehot = np.zeros((n, self.n_nodes), dtype=np.float32)
        onehot[np.arange(n), cur] = 1.0

        bits = ((mask[:, None] >> np.arange(self.n_customers)[None, :]) & 1).astype(np.float32)

        if self.is_capacitated:
            load = (cap[:, None] / float(self.capacity)).astype(np.float32)
        else:
            load = np.ones((n, 1), dtype=np.float32)

        feats = np.concatenate([onehot, bits, load], axis=1)
        self.state_features = jnp.asarray(feats)
        self.feature_dim = int(feats.shape[1])

    # ------------------------------------------------------------------
    # JAX API (mirrors FrozenLakeEnv / DoorKeyEnv)
    # ------------------------------------------------------------------

    def reset(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Reset to the start state at the depot. Returns (obs, state); obs == state."""
        state = jnp.int32(self.start_states[0])
        return state, state

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
        """
        One step. Transitions are always deterministic; with travel_noise > 0 the leg's
        COST is scaled by max(0, 1 + travel_noise * z), z ~ N(0,1) — traffic. The scaling
        is zero-mean, so expected rewards (and hence the exact oracle) are unchanged.

        Returns obs, next_state, reward, done, info — obs == next_state (int32 index).

        NOTE: the caller must mask illegal actions (see env.action_masks); an unmasked
        illegal action self-loops with a penalty rather than raising.
        """
        outcome = jnp.int32(0)
        next_state = self.next_states[state, action, outcome]
        reward = self.rewards[state, action, outcome]
        done = self.dones[state, action, outcome]

        if self.is_stochastic:
            factor = jnp.maximum(0.0, 1.0 + self.travel_noise * jax.random.normal(key))
            reward = reward * factor

        return next_state, next_state, reward, done, {}

    def observe(self, state: jax.Array) -> jax.Array:
        """Feature vector for a state index (vmap-safe: `jax.vmap(env.observe)(states)`)."""
        return self.state_features[state]

    # ------------------------------------------------------------------
    # Convenience (python-side; not used inside jit)
    # ------------------------------------------------------------------

    def tour_length(self, tour: List[int]) -> float:
        """Total euclidean length of a node sequence (e.g. [0, 3, 1, ..., 0])."""
        return float(sum(self.dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

    def min_loads(self) -> int:
        """Lower bound on trips: total demand / capacity (1 in TSP mode)."""
        if not self.is_capacitated:
            return 1
        return int(np.ceil(self.demand.sum() / self.capacity))
