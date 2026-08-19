"""
JAX budget-constrained routing environment — the ORIENTEERING variant of our CVRP.

WHY THIS EXISTS
---------------
`envs/cvrp.py` gave us a second exact-oracle environment and Claim 1 landed there
(rho 0.52 -> 0.67). Claim 2 did NOT: 50 runs, five arms, a flat null. Two causes were
diagnosed and both are properties of the REWARD, not of routing:

  1. NO HEADROOM. Reward = -distance is dense and easy; plain DQN reached 1.0000 of
     optimal by episode ~750 of 14,000. Replay order cannot matter once the task is
     solved that fast — uniform replay tying PER is the proof.
  2. SATURATED SCORE. CCE's total-variation score compares the RETURN DISTRIBUTIONS of
     the available actions. Under a continuous deterministic reward every action yields
     a different real number, so TV = 1 at ~100% of states: the score fires everywhere,
     which is the same as firing nowhere.

The sharpened rule from that finding: the TV score needs outcomes that can TIE — a
discrete reward — or stochasticity. FrozenLake's 0/1 reward ties constantly, so the
score is selective (~4% of states) and CCE wins there.

THE FIX, AND ITS PRIOR ART
--------------------------
Give the vehicle a travel BUDGET B and score it on how many customers it serves:

    return = number of customers served on a closed tour of length <= B

This is the Orienteering Problem (Golden/Levy/Vohra; see also the distance-constrained
VRP of Laporte, Desrochers & Nobert 1984) — a standard, heavily-studied OR variant, not
an invention of ours. It fixes both causes at once:

    reward becomes an INTEGER COUNT (0..N)   -> outcomes tie -> TV is graded again
    B set just above the optimal tour        -> the task stays hard -> headroom exists

B is the experimental DIAL. Tight B -> every choice risks stranding the vehicle ->
high stakes-concentration. Loose B -> everything fits -> back to the null. The
prediction (registered BEFORE running) is that CCE's advantage tracks B monotonically.

DIFFERENCE FROM THE OR LITERATURE (deliberate)
----------------------------------------------
In OR the budget is a hard feasibility constraint. Here it is *also* enforced by
masking — we never let the vehicle strand itself — but the CONSEQUENCE of a choice
survives: spending budget on a far customer now means fewer customers later. Masking
removes illegal actions, not the trade-off, and the trade-off is what CCE measures.

INTEGER DISTANCES (not an approximation)
----------------------------------------
Budget-spent must be part of the state or the MDP is not Markov. To keep it a finite,
EXACT state variable we quantize the distance matrix to integer units
(`dist_scale`, default 10) and define the instance on those integers. The oracle then
solves *this* instance exactly — there is no rounding error in the ground truth, it is
simply a slightly different (and fully specified) set of inter-node distances.

STATE / ACTION / TERMINATION
----------------------------
    state  = (current_node, served_mask, remaining_capacity, budget_spent)
    action = "travel to node a"  (a = 0 is the depot: reload, or go home)
    legal  = unserved AND fits the current load AND the vehicle can still get home:
                 spent + D[cur, a] + D[a, depot] <= B
             the depot is legal from any customer (always affordable by construction)
    reward = +1 on arriving at a customer for the first time, else 0
    done   = at the depot with no affordable unserved customer left

Because `budget_spent` strictly increases on every leg, the reachable state graph is a
DAG — the exact oracle is one backward pass over states sorted by spend.

API matches CVRPEnv / FrozenLakeEnv (obs == state == int32 index), so the masked DQN,
the replay buffers and the CCE scorer are reused unchanged.
"""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .cvrp import DEFAULT_XY, DEFAULT_DEMAND, DEFAULT_CAPACITY, SMALL_XY, SMALL_DEMAND, SMALL_CAPACITY, DEPOT

# Instances are shared with envs/cvrp.py so the two environments describe the SAME map.
INSTANCES = {
    "default": {"xy": DEFAULT_XY, "demand": DEFAULT_DEMAND, "capacity": DEFAULT_CAPACITY},
    "small": {"xy": SMALL_XY, "demand": SMALL_DEMAND, "capacity": SMALL_CAPACITY},
}


def quantize(node_xy: np.ndarray, dist_scale: int) -> np.ndarray:
    """Integer distance matrix. Every off-diagonal leg costs at least 1 unit."""
    diff = node_xy[:, None, :] - node_xy[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    D = np.maximum(1, np.rint(d * dist_scale)).astype(np.int64)
    np.fill_diagonal(D, 0)
    return D


def optimal_closed_tour_units(D: np.ndarray, demand: np.ndarray,
                              capacity: Optional[int]) -> int:
    """
    Exact minimum length (in integer units) of a closed tour serving EVERY customer,
    by backward induction over (current, served_mask, remaining_capacity).

    This is the denominator that sets the budget: B = round(budget_mult * this).
    Reload edges (customer -> depot) leave the mask unchanged, so states are ordered by
    popcount with the depot taken FIRST inside each popcount level — the same
    topological subtlety that produced a real bug in the CVRP oracle.
    """
    n = D.shape[0]
    n_cust = n - 1
    full = (1 << n_cust) - 1
    cap0 = int(capacity) if capacity is not None else 0
    INF = float("inf")

    best: Dict[Tuple[int, int, int], float] = {}

    def order_key(s):
        cur, mask, _ = s
        return (-bin(mask).count("1"), cur != DEPOT)

    states: List[Tuple[int, int, int]] = []
    for mask in range(full + 1):
        caps = range(cap0 + 1) if capacity is not None else [0]
        for cap in caps:
            for cur in range(n):
                states.append((cur, mask, cap))
    states.sort(key=order_key)

    for s in states:
        cur, mask, cap = s
        if cur == DEPOT and mask == full:
            best[s] = 0.0
            continue
        v = INF
        for j in range(1, n):
            if (mask >> (j - 1)) & 1:
                continue
            if capacity is not None and demand[j] > cap:
                continue
            nxt = (j, mask | (1 << (j - 1)),
                   cap - int(demand[j]) if capacity is not None else cap)
            v = min(v, D[cur, j] + best.get(nxt, INF))
        if cur != DEPOT:
            nxt = (DEPOT, mask, cap0)
            v = min(v, D[cur, DEPOT] + best.get(nxt, INF))
        best[s] = v

    start = (DEPOT, 0, cap0)
    out = best[start]
    if out == INF:
        raise ValueError("instance is infeasible: no closed tour serves every customer")
    return int(out)


class BudgetRoutingEnv:
    """
    Orienteering-style routing: serve as many customers as possible on a closed tour
    within an integer travel budget. Same public surface as CVRPEnv.
    """

    def __init__(
        self,
        node_xy: Optional[np.ndarray] = None,
        demand: Optional[np.ndarray] = None,
        capacity: Optional[int] = DEFAULT_CAPACITY,
        budget_mult: float = 1.0,
        dist_scale: int = 10,
        budget_units: Optional[int] = None,
        travel_noise: float = 0.0,
        build_P: bool = False,
    ):
        """
        budget_mult: B as a multiple of the exact all-customers optimum. This is THE DIAL.
            < 1.0 -> serving everyone is impossible; the agent must choose a subset.
            = 1.0 -> exactly the optimal tour fits, and nothing else does.
            > 1.0 -> slack; at large enough values the task collapses back to plain CVRP.
        budget_units: set B directly in integer units (overrides budget_mult).
        dist_scale:  integer units per unit of euclidean distance. Larger = finer
            geometry but more states (budget-spent is a state variable).
        build_P: build the Gymnasium-style transition dict. Off by default — it costs
            ~11 python objects per state and this env has hundreds of thousands.
        """
        self.node_xy = DEFAULT_XY if node_xy is None else np.asarray(node_xy, np.float32)
        self.n_nodes = int(self.node_xy.shape[0])
        self.n_customers = self.n_nodes - 1
        self.n_actions = self.n_nodes

        if demand is None:
            demand = DEFAULT_DEMAND if self.n_nodes == DEFAULT_XY.shape[0] else np.ones(
                self.n_nodes, dtype=np.int32)
        self.demand = np.asarray(demand, np.int32).copy()
        self.demand[DEPOT] = 0

        self.capacity = capacity
        self.is_capacitated = capacity is not None
        if self.is_capacitated and int(self.demand.max()) > capacity:
            raise ValueError("some customer's demand exceeds capacity; it could never be served")

        self.dist_scale = int(dist_scale)
        self.D = quantize(self.node_xy, self.dist_scale)
        # Float distances kept for reporting / plotting only.
        diff = self.node_xy[:, None, :] - self.node_xy[None, :, :]
        self.dist = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)

        self.optimal_all_units = optimal_closed_tour_units(self.D, self.demand, self.capacity)
        if budget_units is not None:
            self.budget = int(budget_units)
            self.budget_mult = self.budget / self.optimal_all_units
        else:
            self.budget_mult = float(budget_mult)
            self.budget = int(round(self.budget_mult * self.optimal_all_units))

        if travel_noise < 0.0:
            raise ValueError(f"travel_noise must be >= 0, got {travel_noise}")
        self.travel_noise = float(travel_noise)
        self.is_stochastic = self.travel_noise > 0.0

        self.is_budget_mode = True
        self.metric_name = "served_ratio"
        self._illegal_penalty = -1.0

        self._build(build_P=build_P)

    # ------------------------------------------------------------------
    # dynamics (pure python; used for enumeration and by the oracle)
    # ------------------------------------------------------------------

    def _full_mask(self) -> int:
        return (1 << self.n_customers) - 1

    def _initial_cap(self) -> int:
        return int(self.capacity) if self.is_capacitated else 0

    def _legal_actions(self, s: Tuple[int, int, int, int]) -> List[int]:
        """Legal = affordable (with the trip home reserved), unserved, and it fits."""
        cur, mask, cap, spent = s
        legal: List[int] = []
        for j in range(1, self.n_nodes):
            if (mask >> (j - 1)) & 1:
                continue
            if self.is_capacitated and self.demand[j] > cap:
                continue
            # Reserve the return leg: never let the vehicle strand itself.
            if spent + self.D[cur, j] + self.D[j, DEPOT] > self.budget:
                continue
            legal.append(j)
        if cur != DEPOT:
            legal.append(DEPOT)          # affordable by construction of the rule above
        return sorted(legal)

    def _is_terminal(self, s) -> bool:
        """Home, with nothing affordable left to do."""
        cur, mask, cap, spent = s
        if cur != DEPOT:
            return False
        return not self._legal_actions(s)

    def _next(self, s, a: int):
        """Deterministic transition for a LEGAL action -> (next_state, reward, done)."""
        cur, mask, cap, spent = s
        spent2 = spent + int(self.D[cur, a])
        if a == DEPOT:
            ns = (DEPOT, mask, self._initial_cap(), spent2)
            return ns, 0.0, self._is_terminal(ns)
        new_mask = mask | (1 << (a - 1))
        new_cap = cap - int(self.demand[a]) if self.is_capacitated else cap
        return (a, new_mask, new_cap, spent2), 1.0, False

    # ------------------------------------------------------------------
    # enumeration
    # ------------------------------------------------------------------

    def _build(self, build_P: bool) -> None:
        start = (DEPOT, 0, self._initial_cap(), 0)
        index: Dict[Tuple[int, int, int, int], int] = {start: 0}
        order: List[Tuple[int, int, int, int]] = [start]
        frontier = [start]
        while frontier:
            s = frontier.pop()
            if self._is_terminal(s):
                continue
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
        self.start_states = [0]

        K = 1
        self.n_outcomes = K
        next_s = np.zeros((n, self.n_actions, K), dtype=np.int32)
        rew = np.zeros((n, self.n_actions, K), dtype=np.float32)
        done_a = np.zeros((n, self.n_actions, K), dtype=bool)
        masks = np.zeros((n, self.n_actions), dtype=bool)
        st_cur = np.zeros(n, dtype=np.int32)
        st_mask = np.zeros(n, dtype=np.int64)
        st_cap = np.zeros(n, dtype=np.int32)
        st_spent = np.zeros(n, dtype=np.int32)
        st_served = np.zeros(n, dtype=np.int32)

        P: Dict = {} if build_P else None
        non_terminal: List[int] = []

        for s, si in index.items():
            cur, mask, cap, spent = s
            st_cur[si], st_mask[si], st_cap[si], st_spent[si] = cur, mask, cap, spent
            st_served[si] = bin(mask).count("1")
            terminal = self._is_terminal(s)
            if not terminal:
                non_terminal.append(si)
            legal = set(self._legal_actions(s)) if not terminal else set()
            if build_P:
                P[si] = {}
            for a in range(self.n_actions):
                if terminal:
                    next_s[si, a, 0] = si
                    rew[si, a, 0] = 0.0
                    done_a[si, a, 0] = True
                    if build_P:
                        P[si][a] = [(1.0, si, 0.0, True)]
                elif a in legal:
                    ns, r, d = self._next(s, a)
                    nsi = index[ns]
                    masks[si, a] = True
                    next_s[si, a, 0] = nsi
                    rew[si, a, 0] = r
                    done_a[si, a, 0] = d
                    if build_P:
                        P[si][a] = [(1.0, nsi, float(r), bool(d))]
                else:
                    next_s[si, a, 0] = si
                    rew[si, a, 0] = self._illegal_penalty
                    done_a[si, a, 0] = False
                    if build_P:
                        P[si][a] = [(1.0, si, float(self._illegal_penalty), False)]

        self.P = P
        self.non_terminal = sorted(non_terminal)
        self.next_states = jnp.asarray(next_s)
        self.rewards = jnp.asarray(rew)
        self.dones = jnp.asarray(done_a)
        self.action_masks = jnp.asarray(masks)
        self.state_current = jnp.asarray(st_cur)
        self.state_current_np = st_cur
        self.state_mask = st_mask
        self.state_cap = st_cap
        self.state_spent = st_spent
        self.state_served = st_served
        # numpy mirrors of the transition tables (the exact oracle runs on these)
        self.next_states_np = next_s[:, :, 0]
        self.rewards_np = rew[:, :, 0]
        self.dones_np = done_a[:, :, 0]
        self.action_masks_np = masks

        self._build_features(st_cur, st_mask, st_cap, st_spent)

    def _build_features(self, cur, mask, cap, spent) -> None:
        """
        [ one-hot current node | served bits | load fraction | budget-REMAINING fraction ]

        The budget fraction is not optional: without it the network cannot tell a vehicle
        with fuel to spare from one that is nearly out, and the task stops being Markov
        from the agent's point of view.
        """
        n = self.n_states
        onehot = np.zeros((n, self.n_nodes), dtype=np.float32)
        onehot[np.arange(n), cur] = 1.0
        bits = ((mask[:, None] >> np.arange(self.n_customers)[None, :]) & 1).astype(np.float32)
        load = ((cap[:, None] / float(self.capacity)) if self.is_capacitated
                else np.ones((n, 1))).astype(np.float32)
        fuel = (1.0 - spent[:, None] / float(self.budget)).astype(np.float32)
        feats = np.concatenate([onehot, bits, load, fuel], axis=1)
        self.state_features = jnp.asarray(feats)
        self.feature_dim = int(feats.shape[1])

    # ------------------------------------------------------------------
    # JAX API
    # ------------------------------------------------------------------

    def reset(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        state = jnp.int32(self.start_states[0])
        return state, state

    def step(self, key, state, action):
        outcome = jnp.int32(0)
        next_state = self.next_states[state, action, outcome]
        reward = self.rewards[state, action, outcome]
        done = self.dones[state, action, outcome]
        return next_state, next_state, reward, done, {}

    def observe(self, state: jax.Array) -> jax.Array:
        return self.state_features[state]

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def tour_length(self, tour: List[int]) -> float:
        return float(sum(self.dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

    def tour_units(self, tour: List[int]) -> int:
        return int(sum(self.D[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

    def min_loads(self) -> int:
        if not self.is_capacitated:
            return 1
        return int(np.ceil(self.demand.sum() / self.capacity))

    def describe(self) -> str:
        return (f"BudgetRouting: {self.n_customers} customers, capacity {self.capacity}, "
                f"scale {self.dist_scale}, B={self.budget}u "
                f"({self.budget_mult:.2f}x all-customers optimum {self.optimal_all_units}u), "
                f"{self.n_states:,} states")
