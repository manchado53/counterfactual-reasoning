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

# A LARGER instance — the size axis. 12 customers on a jittered ring around a central
# depot, same flavour as the 10-stop default map. Measured cost with budget-spent in the
# state: ~507k states / 15 s build / 1.0 GB at budget_mult 0.80, and the all-customers
# optimum no longer fits the budget there (9 of 12 servable), which is exactly the
# headroom the 10-customer map loses. 13 customers was measured too and REJECTED:
# 4.4M states and 6.4 GB at 1.00x is too heavy to build inside every run.
RING12_XY = np.array(
    [
        [0.500, 0.500],  # depot
        [0.884, 0.500],  # C1,
        [0.800, 0.673],  # C2,
        [0.662, 0.781],  # C3,
        [0.500, 0.822],  # C4,
        [0.299, 0.848],  # C5,
        [0.144, 0.706],  # C6,
        [0.119, 0.500],  # C7,
        [0.160, 0.304],  # C8,
        [0.313, 0.176],  # C9,
        [0.500, 0.086],  # C10,
        [0.701, 0.152],  # C11,
        [0.777, 0.340],  # C12
    ],
    dtype=np.float32,
)
RING12_DEMAND = np.array([0] + [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4], dtype=np.int32)


# Alternative GEOMETRIES, same 10 customers and demands as the default ring.
# Used to test whether findings on the default map are properties of the DOMAIN
# or just of one layout.
CLUSTERED_XY = np.array(
    [
        [0.500, 0.500],
        [0.180, 0.200],
        [0.240, 0.300],
        [0.140, 0.320],
        [0.280, 0.180],
        [0.200, 0.400],
        [0.800, 0.780],
        [0.860, 0.680],
        [0.740, 0.860],
        [0.900, 0.840],
        [0.680, 0.720],
    ],
    dtype=np.float32,
)

HUB_OUTLIERS_XY = np.array(
    [
        [0.500, 0.500],
        [0.630, 0.500],
        [0.581, 0.602],
        [0.471, 0.627],
        [0.383, 0.556],
        [0.383, 0.444],
        [0.471, 0.373],
        [0.581, 0.398],
        [0.050, 0.050],
        [0.950, 0.100],
        [0.920, 0.950],
    ],
    dtype=np.float32,
)

TWO_LOBES_XY = np.array(
    [
        [0.500, 0.500],
        [0.620, 0.500],
        [0.532, 0.613],
        [0.320, 0.660],
        [0.108, 0.613],
        [0.020, 0.500],
        [0.380, 0.500],
        [0.468, 0.387],
        [0.680, 0.340],
        [0.892, 0.387],
        [0.980, 0.500],
    ],
    dtype=np.float32,
)

# Instances are shared with envs/cvrp.py so the two environments describe the SAME map.
INSTANCES = {
    "default": {"xy": DEFAULT_XY, "demand": DEFAULT_DEMAND, "capacity": DEFAULT_CAPACITY},
    "small": {"xy": SMALL_XY, "demand": SMALL_DEMAND, "capacity": SMALL_CAPACITY},
    "ring12": {"xy": RING12_XY, "demand": RING12_DEMAND, "capacity": 10},
    "clustered": {"xy": CLUSTERED_XY, "demand": np.array([0, 3, 2, 4, 2, 1, 3, 2, 3, 2, 2], dtype=np.int32), "capacity": 10},
    "hub_outliers": {"xy": HUB_OUTLIERS_XY, "demand": np.array([0, 3, 2, 4, 2, 1, 3, 2, 3, 2, 2], dtype=np.int32), "capacity": 10},
    "two_lobes": {"xy": TWO_LOBES_XY, "demand": np.array([0, 3, 2, 4, 2, 1, 3, 2, 3, 2, 2], dtype=np.int32), "capacity": 10},
}


def quantize(node_xy: np.ndarray, dist_scale: int) -> np.ndarray:
    """Integer distance matrix. Every off-diagonal leg costs at least 1 unit."""
    diff = node_xy[:, None, :] - node_xy[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    D = np.maximum(1, np.rint(d * dist_scale)).astype(np.int64)
    np.fill_diagonal(D, 0)
    return D


def optimal_closed_tour_units(D: np.ndarray, demand: np.ndarray,
                              capacity: Optional[int], return_tour: bool = False):
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
    if not return_tour:
        return int(out)

    # Walk the DP forward, always taking a move that realises the optimum. Arrival times
    # are the running cost, which is what time windows get centred on.
    s, tour, arrivals = start, [DEPOT], {}
    t = 0
    while not (s[0] == DEPOT and s[1] == full):
        cur, mask, cap = s
        best_move, best_cost = None, INF
        for j in range(1, n):
            if (mask >> (j - 1)) & 1:
                continue
            if capacity is not None and demand[j] > cap:
                continue
            nxt = (j, mask | (1 << (j - 1)),
                   cap - int(demand[j]) if capacity is not None else cap)
            c = D[cur, j] + best.get(nxt, INF)
            if c < best_cost:
                best_move, best_cost = (j, nxt), c
        if cur != DEPOT:
            nxt = (DEPOT, mask, cap0)
            c = D[cur, DEPOT] + best.get(nxt, INF)
            if c < best_cost:
                best_move, best_cost = (DEPOT, nxt), c
        j, nxt = best_move
        t += int(D[cur, j])
        tour.append(j)
        if j != DEPOT:
            arrivals[j] = t
        s = nxt
    return int(out), tour, arrivals


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
        time_windows: bool = False,
        n_windowed: int = 3,
        window_width: int = 6,
        allow_stranding: bool = False,
        reward_shape: str = 'stepwise',
        strand_penalty: float = -10.0,
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

        # ── OPTION A: the two changes that give routing cliffs ───────────────
        # Both default OFF, so every committed budget-mode result reproduces byte for byte.
        #
        # `spent` is reinterpreted as TIME rather than fuel: driving advances the clock, B is
        # the end of the shift, and a window is a delivery appointment. Same state variable,
        # and it matches the standard VRPTW formulation instead of inventing one.
        if reward_shape not in ('stepwise', 'terminal'):
            raise ValueError(f"reward_shape must be 'stepwise' or 'terminal', got {reward_shape!r}")
        self.allow_stranding = bool(allow_stranding)
        self.reward_shape = reward_shape
        self.strand_penalty = float(strand_penalty)
        self.time_windows = bool(time_windows)
        self.window_width = int(window_width)

        # Wide-open windows by default; a subset gets tightened below.
        BIG = 10 ** 6
        self.window_open = np.zeros(self.n_nodes, dtype=np.int64)
        self.window_close = np.full(self.n_nodes, BIG, dtype=np.int64)
        self.windowed_customers: List[int] = []

        if self.time_windows:
            # Windows are centred on when the exact all-customers optimum reaches each stop,
            # rescaled into this budget, so at least one sensible route stays feasible and the
            # oracle keeps meaning something. Only a SUBSET is windowed: that makes the stakes
            # bimodal by construction (a few pivotal stops, the rest free) rather than
            # constraining everything equally, which would just rebuild the smooth middle band.
            _, tour, arrivals = optimal_closed_tour_units(
                self.D, self.demand, self.capacity, return_tour=True)
            order = [j for j in tour if j != DEPOT]
            k = max(1, int(n_windowed))
            step = max(1, len(order) // k)
            self.windowed_customers = sorted(order[::step][:k])
            scale = self.budget / max(1, self.optimal_all_units)
            half = max(1, self.window_width // 2)
            for c in self.windowed_customers:
                centre = int(round(arrivals[c] * scale))
                self.window_open[c] = max(0, centre - half)
                self.window_close[c] = min(self.budget, centre + half)

        self._build(build_P=build_P)

    # ------------------------------------------------------------------
    # dynamics (pure python; used for enumeration and by the oracle)
    # ------------------------------------------------------------------

    def _full_mask(self) -> int:
        return (1 << self.n_customers) - 1

    def _initial_cap(self) -> int:
        return int(self.capacity) if self.is_capacitated else 0

    def _service_time(self, cur: int, j: int, spent: int) -> int:
        """Clock after driving cur -> j, waiting for the window to open if we arrive early."""
        arrival = spent + int(self.D[cur, j])
        return max(arrival, int(self.window_open[j])) if self.time_windows else arrival

    def _legal_actions(self, s: Tuple[int, int, int, int]) -> List[int]:
        """
        Legal = unserved, fits the load, the window has not closed, and it is affordable.

        "Affordable" depends on `allow_stranding`:
          False (default) - the trip home is RESERVED, so the vehicle can never strand itself
                            and the worst outcome is serving fewer customers.
          True            - only the leg itself must fit before B. The vehicle CAN drive
                            somewhere it cannot return from, and then the run fails outright.
                            That is what makes a decision catastrophic.
        """
        cur, mask, cap, spent = s
        legal: List[int] = []
        for j in range(1, self.n_nodes):
            if (mask >> (j - 1)) & 1:
                continue
            if self.is_capacitated and self.demand[j] > cap:
                continue
            t = self._service_time(cur, j, spent)
            if self.time_windows and t > self.window_close[j]:
                continue                       # window missed - gone for the rest of the run
            if self.allow_stranding:
                if t > self.budget:
                    continue
            elif t + self.D[j, DEPOT] > self.budget:
                continue
            legal.append(j)
        if cur != DEPOT:
            if not self.allow_stranding or spent + self.D[cur, DEPOT] <= self.budget:
                legal.append(DEPOT)
        return sorted(legal)

    def _is_failure(self, s) -> bool:
        """Stranded: out of moves while still away from the depot, or home past the deadline."""
        cur, mask, cap, spent = s
        if not self.allow_stranding:
            return False
        return cur != DEPOT and not self._legal_actions(s)

    def _is_terminal(self, s) -> bool:
        """
        Out of moves. With stranding off that can only happen at the depot, so this keeps the
        original behaviour exactly; with stranding on it also covers the failure case of being
        stuck at a customer.
        """
        return not self._legal_actions(s)

    def _next(self, s, a: int):
        """
        Deterministic transition for a LEGAL action -> (next_state, reward, done).

        Reward shape:
          'stepwise' - +1 on first arrival at a customer (original behaviour), plus
                       `strand_penalty` if the run ends stranded.
          'terminal' - nothing along the way; the whole run pays out the served count on a
                       SUCCESSFUL return, and zero otherwise. Sparse and all-or-nothing, which
                       is the shape that makes per-seed outcomes bimodal.
        """
        cur, mask, cap, spent = s
        spent2 = self._service_time(cur, a, spent)

        if a == DEPOT:
            ns = (DEPOT, mask, self._initial_cap(), spent2)
            done = self._is_terminal(ns)
            if done and self.reward_shape == 'terminal':
                return ns, float(bin(mask).count("1")), True
            return ns, 0.0, done

        new_mask = mask | (1 << (a - 1))
        new_cap = cap - int(self.demand[a]) if self.is_capacitated else cap
        ns = (a, new_mask, new_cap, spent2)
        step_r = 1.0 if self.reward_shape == 'stepwise' else 0.0
        if self._is_failure(ns):
            # Drove somewhere with no way back: the run is over and it failed.
            penalty = self.strand_penalty if self.reward_shape == 'stepwise' else 0.0
            return ns, step_r + penalty, True
        return ns, step_r, False

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
        st_failed = np.zeros(n, dtype=bool)     # terminal AND stranded -> the run scored nothing

        P: Dict = {} if build_P else None
        non_terminal: List[int] = []

        for s, si in index.items():
            cur, mask, cap, spent = s
            st_cur[si], st_mask[si], st_cap[si], st_spent[si] = cur, mask, cap, spent
            st_served[si] = bin(mask).count("1")
            st_failed[si] = self._is_failure(s)
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
        self.state_failed = st_failed
        self.n_failure_states = int(st_failed.sum())
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
        parts = [onehot, bits, load, fuel]

        if self.time_windows:
            # One bit per customer: has this stop's window already closed? Derivable from
            # `spent` and the fixed deadlines, but making it explicit spares the network from
            # rediscovering a per-customer threshold out of a single scalar clock.
            close = self.window_close[1:][None, :]
            missed = (spent[:, None] > close).astype(np.float32)
            parts.append(missed)

        feats = np.concatenate(parts, axis=1)
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
