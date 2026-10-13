"""
JAX DoorKey environment — a tabular reimplementation of MiniGrid's DoorKey.

Design goal: a SECOND good-oracle environment for CCE (the first being FrozenLake).
Like FrozenLake here, this is OUR reimplementation, not an import of Farama `minigrid`
or Navix — because the Claim-1 oracle needs an enumerable Gymnasium-style transition
table `env.P`, which neither library exposes.

DoorKey's full MDP state on a FIXED layout is small:
    state = (agent_cell, agent_dir, has_key, door_state)
so we enumerate the *reachable* state set by BFS at construction time and index it
into a single integer — exactly like FrozenLake indexes (row, col). The Q-network,
the CCE counterfactual rollout, the replay buffer, and the oracle then all reuse the
FrozenLake plumbing UNCHANGED (obs == state == int32).

Stochasticity (`slip_prob`) — mirrors FrozenLake's `is_slippery`:
    With probability `slip_prob` the executed action is replaced by a uniformly random
    action (a noisy actuator); otherwise the chosen action is executed.
      - slip_prob = 0  -> fully deterministic  (used for the Claim-2 sample-efficiency
        experiments, where determinism gives CCE its cleanest win, as on FrozenLake).
      - slip_prob > 0  -> stochastic  (used for the Claim-1 oracle-correlation
        experiments: CCE's total-variation signal is only non-degenerate when rollout
        returns actually vary, exactly as FrozenLake's Claim-1 used the SLIPPERY map).
    The reachable state set is identical for any slip_prob (BFS explores all actions),
    so the same enumeration/indexing is reused.

Why the reward differs from Farama MiniGrid:
    MiniGrid uses reward = 1 - 0.9*(step_count/max_steps), which depends on step_count.
    step_count is NOT part of our (cell, dir, has_key, door) state, so that reward would
    make the tabular MDP non-Markov and break value iteration. We instead use a plain
    +1.0 on first reaching the goal, 0.0 otherwise, with gamma < 1 in the learner /
    oracle. Discounting makes shorter paths — and the gating decisions (get the key,
    open the door) — consequential, which is the structure CCE targets.

Deviations from Farama MiniGrid (documented, minor):
    - The `drop` action (index 4) is a NO-OP. Dropping the key would make the key's
      position a state variable and blow up the enumeration; it is never useful in
      DoorKey. `has_key` is therefore monotonic (0 -> 1).
    - Layout is fixed (not re-randomized per reset) so the state set is enumerable.

Lava (optional, layout-dependent — see DOORKEY_6x6_LAVA):
    Walkable but fatal, matching MiniGrid's own Lava.can_overlap()=True + terminate-on-entry
    semantics: FORWARD onto a lava cell succeeds (like walking onto the goal) but ends the
    episode with reward 0 (like FrozenLake's holes) rather than blocking movement like a wall.
    This gives DoorKey the catastrophe structure it otherwise lacks: without lava every wrong
    action is merely recoverable (costs a few extra discounted steps), so the oracle's
    action-value gaps stay small almost everywhere and CCE's total-variation score has little
    to lock onto. A lava-adjacent state has a genuinely large gap — one action still reaches
    the goal, the other is a dead end — exactly the FrozenLake-hole mechanism CCE is built to
    detect.

Action space (Discrete(7), MiniGrid order):
    0 left    turn CCW (dir -= 1 mod 4)
    1 right   turn CW  (dir += 1 mod 4)
    2 forward move onto the front cell if overlappable
    3 pickup  grab the key if it is the front cell and not already carried
    4 drop    NO-OP (see above)
    5 toggle  act on a front door: locked+key -> open; else flip open/closed
    6 done    NO-OP

Directions (MiniGrid order), as (drow, dcol):
    0 right/east (0,+1) · 1 down/south (+1,0) · 2 left/west (0,-1) · 3 up/north (-1,0)

API (identical to FrozenLakeEnv):
    env = DoorKeyEnv()                      # 6x6 default, deterministic
    env = DoorKeyEnv(slip_prob=0.2)         # stochastic (for Claim 1)
    obs, state = env.reset(key)             # obs == state (int32 index)
    obs, state, reward, done, info = env.step(key, state, action)

    env.P[state][action] = [(prob, next_state, reward, terminated), ...]

Both reset() and step() are JAX-pure: safe for jax.jit and jax.vmap.
"""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Actions
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE = 0, 1, 2, 3, 4, 5, 6
N_ACTIONS = 7

# Directions (MiniGrid order). NOTE these are DISTINCT from the LEFT/RIGHT *actions*
# above — LEFT/RIGHT turn the agent; DIR_* name which way it faces.
DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP = 0, 1, 2, 3
# Direction -> (drow, dcol). MiniGrid order: right/east, down/south, left/west, up/north.
DIR_VEC = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=np.int32)
N_DIR = 4

# Door states
DOOR_LOCKED, DOOR_CLOSED, DOOR_OPEN = 0, 1, 2  # locked-closed, unlocked-closed, unlocked-open

# Tile glyphs
WALL, FLOOR, KEY, DOOR, GOAL, LAVA = "W", ".", "K", "D", "G", "L"

# Fixed 6x6 DoorKey layout (walls on the perimeter, one vertical split wall at col 3
# with a single door cell, key + agent start in the left room, goal in the right room).
#   col:  0 1 2 3 4 5
DOORKEY_6x6: List[str] = [
    "WWWWWW",   # row 0
    "W..W.W",   # row 1  (col3 = split wall)
    "W..D.W",   # row 2  (col3 = door)
    "WK.W.W",   # row 3  (col1 = key)
    "W..WGW",   # row 4  (col4 = goal)
    "WWWWWW",   # row 5
]
DOORKEY_6x6_START = (1, 1)
DOORKEY_6x6_START_DIR = DIR_RIGHT

# Fixed 6x6 DoorKey+Lava layout — the 6x6 above with ONE lava tile at (4,2).
#
# Why 6x6 and only one lava tile: lava must add catastrophe WITHOUT destroying the reward
# signal DQN needs to bootstrap. Measured with a uniform-random policy over 20k episodes
# (80-step cap), the fraction of episodes that reach the goal at all:
#     6x6 no lava .............. 0.215%   (trains fine — this is the reference)
#     6x6 + 1 lava (this) ...... 0.110%   (still bootstraps; lava-adjacent oracle gap 4.2x)
#     8x8 + 1..4 lava .......... 0.000%   (ZERO goals in 20k episodes — unlearnable)
# The blocker on 8x8 was path length (19 steps vs 11), not lava density: even a single lava
# tile placed far from the route left 0 successes, because a 19-step three-stage task
# (key -> door -> goal) is already past what random exploration completes. So the catastrophe
# structure goes on the SHORT map.
#
# Lava sits at (4,2), in the lower-left room one cell diagonally off the key at (3,1) and
# directly below the (3,2) corridor cell the agent crosses on its way to the key/door. Its
# two live approaches — (3,2) facing DOWN and (4,1) facing RIGHT — are both states a real
# policy passes through, which is what makes those neighbouring states consequential rather
# than a hazard parked somewhere the agent never goes.
#   col:  0 1 2 3 4 5
DOORKEY_6x6_LAVA: List[str] = [
    "WWWWWW",   # row 0
    "W..W.W",   # row 1  (col3 = split wall)
    "W..D.W",   # row 2  (col3 = door)
    "WK.W.W",   # row 3  (col1 = key)
    "W.LWGW",   # row 4  (col2 = lava, col4 = goal)
    "WWWWWW",   # row 5
]
DOORKEY_6x6_LAVA_START = (1, 1)
DOORKEY_6x6_LAVA_START_DIR = DIR_RIGHT

def make_doorkey_desc(size: int, split_col: Optional[int] = None,
                      door_row: Optional[int] = None,
                      key_cell: Optional[Tuple[int, int]] = None,
                      lava: Tuple[Tuple[int, int], ...] = ()) -> List[str]:
    """
    Build a MiniGrid-style DoorKey map: perimeter wall, one vertical split wall carrying a
    single door, key + agent in the left room, goal at (size-2, size-2).

    Larger sizes exist because the small fixed maps are trivially solved. Published
    success rates for DQN with full-grid observations (VLM-Guided Experience Replay,
    arXiv 2602.01915) show uniform/PER replay saturating at 1.00 on DoorKey-8x8 but
    sitting at 0.62/0.82 (12x12) and 0.24/0.48 (16x16) — i.e. headroom only appears at
    the larger sizes. Enumeration stays cheap: 12x12 ~1.1k states, 16x16 ~2.2k, and
    value iteration over those takes well under a second.
    """
    n = size
    split_col = split_col if split_col is not None else n // 2
    door_row = door_row if door_row is not None else n // 2
    key_cell = key_cell if key_cell is not None else (n - 2, 1)
    g = [[FLOOR] * n for _ in range(n)]
    for i in range(n):
        g[0][i] = g[n - 1][i] = g[i][0] = g[i][n - 1] = WALL
    for r in range(1, n - 1):
        g[r][split_col] = WALL
    g[door_row][split_col] = DOOR
    g[key_cell[0]][key_cell[1]] = KEY
    g[n - 2][n - 2] = GOAL
    for (lr, lc) in lava:
        g[lr][lc] = LAVA
    return ["".join(row) for row in g]


LAYOUTS: Dict[str, dict] = {
    "6x6": {"desc": DOORKEY_6x6, "start": DOORKEY_6x6_START, "start_dir": DOORKEY_6x6_START_DIR},
    "6x6_lava": {"desc": DOORKEY_6x6_LAVA, "start": DOORKEY_6x6_LAVA_START,
                 "start_dir": DOORKEY_6x6_LAVA_START_DIR},
    # Larger fixed layouts. Optimal path: 12x12 = 31 steps, 16x16 = 43 steps (vs 6x6's 11).
    # Random exploration cannot solve these from a fixed start (measured 0 goals in 4k
    # episodes), so they are intended to be trained with random_start='exploring' and
    # EVALUATED from the canonical fixed start.
    "12x12": {"desc": make_doorkey_desc(12), "start": (1, 1), "start_dir": DIR_RIGHT},
    "16x16": {"desc": make_doorkey_desc(16), "start": (1, 1), "start_dir": DIR_RIGHT},
    "12x12_lava": {"desc": make_doorkey_desc(12, lava=((7, 2), (2, 9))),
                   "start": (1, 1), "start_dir": DIR_RIGHT},
    "16x16_lava": {"desc": make_doorkey_desc(16, lava=((9, 2), (2, 13))),
                   "start": (1, 1), "start_dir": DIR_RIGHT},
}


class DoorKeyEnv:
    """
    Tabular JAX DoorKey. Reachable states are enumerated at init and transitions are
    precomputed into JAX arrays for an O(1) jit/vmap-safe step, plus a Gymnasium-style
    `env.P` dict for the Claim-1 value-iteration oracle.
    """

    def __init__(
        self,
        layout_name: str = "6x6",
        desc: Optional[List[str]] = None,
        start: Optional[Tuple[int, int]] = None,
        start_dir: int = DIR_RIGHT,
        slip_prob: float = 0.0,
        random_start: str = "fixed",
    ):
        if desc is not None:
            self.desc = desc
            self.start_cell = start if start is not None else self._find_tile("S")
            self.start_dir = start_dir
        else:
            if layout_name not in LAYOUTS:
                raise ValueError(
                    f"Unknown layout '{layout_name}'. Use one of {list(LAYOUTS)} or pass desc=."
                )
            cfg = LAYOUTS[layout_name]
            self.desc = cfg["desc"]
            self.start_cell = cfg["start"]
            self.start_dir = cfg["start_dir"]

        if not (0.0 <= slip_prob < 1.0):
            raise ValueError(f"slip_prob must be in [0, 1); got {slip_prob}")
        self.slip_prob = float(slip_prob)

        if random_start not in ("fixed", "agent", "exploring"):
            raise ValueError(
                f"random_start must be 'fixed', 'agent' or 'exploring'; got {random_start!r}")
        self.random_start = random_start

        self.nrows = len(self.desc)
        self.ncols = len(self.desc[0])
        self.n_actions = N_ACTIONS

        self.key_cell = self._find_tile(KEY)
        self.door_cell = self._find_tile(DOOR)
        self.goal_cell = self._find_tile(GOAL)
        self.lava_cells = self._find_all_tiles(LAVA)

        self._build_states_and_table()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _find_tile(self, glyph: str) -> Tuple[int, int]:
        for r, row in enumerate(self.desc):
            for c, tile in enumerate(row):
                if tile == glyph:
                    return (r, c)
        raise ValueError(f"Tile '{glyph}' not found in layout.")

    def _find_all_tiles(self, glyph: str) -> List[Tuple[int, int]]:
        return [
            (r, c)
            for r, row in enumerate(self.desc)
            for c, tile in enumerate(row)
            if tile == glyph
        ]

    def _tile(self, row: int, col: int) -> str:
        return self.desc[row][col]

    def _is_goal(self, row: int, col: int) -> bool:
        return self._tile(row, col) == GOAL

    def _is_lava(self, row: int, col: int) -> bool:
        return self._tile(row, col) == LAVA

    def _is_terminal(self, row: int, col: int) -> bool:
        return self._is_goal(row, col) or self._is_lava(row, col)

    # ------------------------------------------------------------------
    # Pure-python transition (used to build the table + BFS reachability)
    # ------------------------------------------------------------------

    def _transition(self, s: Tuple[int, int, int, int, int], action: int):
        """
        Apply `action` deterministically to state s = (row, col, dir, has_key, door).

        Returns (next_state_tuple, reward, done). Goal and lava states are both terminal and
        loop to themselves with reward 0 (the goal's +1 is only granted on the transition that
        first enters it — see the FORWARD branch below). This is the *executed*-action
        transition; slip is applied on top by mixing over executed actions in the table build.
        """
        row, col, d, has_key, door = s

        if self._is_terminal(row, col):
            return (row, col, d, has_key, door), 0.0, True

        nr, nc, nd, nkey, ndoor = row, col, d, has_key, door
        reward, done = 0.0, False

        if action == LEFT:
            nd = (d - 1) % N_DIR
        elif action == RIGHT:
            nd = (d + 1) % N_DIR
        elif action == FORWARD:
            fr = row + int(DIR_VEC[d, 0])
            fc = col + int(DIR_VEC[d, 1])
            ftile = self._tile(fr, fc)
            can_move = False
            if ftile == WALL:
                can_move = False
            elif ftile == DOOR:
                can_move = (door == DOOR_OPEN)
            elif ftile == KEY:
                can_move = (has_key == 1)
            elif ftile == GOAL:
                can_move = True
                reward, done = 1.0, True
            elif ftile == LAVA:
                can_move = True
                reward, done = 0.0, True
            else:  # FLOOR
                can_move = True
            if can_move:
                nr, nc = fr, fc
        elif action == PICKUP:
            fr = row + int(DIR_VEC[d, 0])
            fc = col + int(DIR_VEC[d, 1])
            if self._tile(fr, fc) == KEY and has_key == 0:
                nkey = 1
        elif action == DROP:
            pass  # NO-OP (documented deviation)
        elif action == TOGGLE:
            fr = row + int(DIR_VEC[d, 0])
            fc = col + int(DIR_VEC[d, 1])
            if self._tile(fr, fc) == DOOR:
                if door == DOOR_LOCKED:
                    if has_key == 1:
                        ndoor = DOOR_OPEN
                elif door == DOOR_CLOSED:
                    ndoor = DOOR_OPEN
                elif door == DOOR_OPEN:
                    ndoor = DOOR_CLOSED
        elif action == DONE:
            pass  # NO-OP

        return (nr, nc, nd, nkey, ndoor), reward, done

    # ------------------------------------------------------------------
    # Enumeration + transition table construction
    # ------------------------------------------------------------------

    def _build_states_and_table(self) -> None:
        """
        BFS from the initial state to enumerate the reachable state set (identical for
        any slip_prob), assign each a flat integer index, then build:
          - self.P                 : {s: {a: [(prob, s', r, term), ...]}}  (K outcomes)
          - self.next_states/rewards/dones/outcome_probs : (n_states, n_actions, K)
          - self.non_terminal      : reachable non-goal state indices (for the oracle)
          - self.state_to_cell     : idx -> (row, col) for the heatmap

        Outcome layout:
          - slip_prob == 0 : K = 1, slot 0 is the intended action's outcome (prob 1).
          - slip_prob  > 0 : K = n_actions, slot e is the outcome of *executing* action
            e, with prob P(execute e | intended a) = (1-slip) + slip/K if e==a else slip/K.
        """
        init = (self.start_cell[0], self.start_cell[1], self.start_dir, 0, DOOR_LOCKED)

        index: Dict[Tuple[int, int, int, int, int], int] = {}
        order: List[Tuple[int, int, int, int, int]] = []
        frontier = [init]
        index[init] = 0
        order.append(init)
        while frontier:
            s = frontier.pop()
            for a in range(self.n_actions):
                ns, _, _ = self._transition(s, a)
                if ns not in index:
                    index[ns] = len(order)
                    order.append(ns)
                    frontier.append(ns)

        n = len(order)
        self.n_states = n
        self._index = index
        self._order = order

        # Which states an episode may begin in. This only changes the INITIAL-STATE
        # DISTRIBUTION — the transition table and therefore the value-iteration oracle are
        # untouched, so Claim 1 stays exact for every mode.
        #   'fixed'     : the single canonical start (original behaviour).
        #   'agent'     : any reachable state with no key and the door still locked, i.e.
        #                 randomise only where the agent stands/faces. Mirrors MiniGrid,
        #                 which randomises the agent's position but always starts it
        #                 key-less with the door locked (so this stays in the left room).
        #   'exploring' : any reachable non-terminal state, including mid-task ones
        #                 (key already held, door already open). This is Sutton & Barto's
        #                 "exploring starts": it guarantees the policy is trained from
        #                 EVERY state, which both (a) supplies a natural curriculum —
        #                 some episodes begin near the goal, so a sparse-reward agent gets
        #                 its first successes — and (b) is what CCE's rollout estimator
        #                 needs, since that estimator rolls the policy out from arbitrary
        #                 states and is uninformative wherever the policy is untrained.
        if self.random_start == "fixed":
            self.start_states = [index[init]]
        elif self.random_start == "agent":
            self.start_states = sorted(
                si for s, si in index.items()
                if s[3] == 0 and s[4] == DOOR_LOCKED and not self._is_terminal(s[0], s[1])
            )
        else:  # 'exploring'
            self.start_states = sorted(
                si for s, si in index.items() if not self._is_terminal(s[0], s[1])
            )
        self._start_states_arr = jnp.array(self.start_states, dtype=jnp.int32)

        slip = self.slip_prob
        deterministic = (slip == 0.0)
        K = 1 if deterministic else self.n_actions
        self.n_outcomes = K

        next_s_np = np.zeros((n, self.n_actions, K), dtype=np.int32)
        rew_np = np.zeros((n, self.n_actions, K), dtype=np.float32)
        done_np = np.zeros((n, self.n_actions, K), dtype=bool)
        prob_np = np.zeros((n, self.n_actions, K), dtype=np.float32)

        P: Dict = {}
        non_terminal: List[int] = []
        state_to_cell: Dict[int, Tuple[int, int]] = {}

        for s, si in index.items():
            row, col = s[0], s[1]
            state_to_cell[si] = (row, col)
            if not self._is_terminal(row, col):
                non_terminal.append(si)
            P[si] = {}
            for a in range(self.n_actions):
                if deterministic:
                    ns, r, done = self._transition(s, a)
                    nsi = index[ns]
                    next_s_np[si, a, 0] = nsi
                    rew_np[si, a, 0] = r
                    done_np[si, a, 0] = done
                    prob_np[si, a, 0] = 1.0
                    P[si][a] = [(1.0, nsi, float(r), bool(done))]
                else:
                    P[si][a] = []
                    for e in range(self.n_actions):  # executed action e in slot e
                        ns, r, done = self._transition(s, e)
                        nsi = index[ns]
                        p = (1.0 - slip) + slip / K if e == a else slip / K
                        next_s_np[si, a, e] = nsi
                        rew_np[si, a, e] = r
                        done_np[si, a, e] = done
                        prob_np[si, a, e] = p
                        P[si][a].append((float(p), nsi, float(r), bool(done)))

        self.P = P
        self.non_terminal = sorted(non_terminal)
        self.state_to_cell = state_to_cell
        self.next_states = jnp.array(next_s_np)
        self.rewards = jnp.array(rew_np)
        self.dones = jnp.array(done_np)
        self.outcome_probs = jnp.array(prob_np)

    # ------------------------------------------------------------------
    # JAX API (mirrors FrozenLakeEnv)
    # ------------------------------------------------------------------

    def reset(self, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Start a new episode. Returns (obs, state); obs == state.

        With random_start='fixed' this is the single canonical start; otherwise a start
        state is sampled uniformly from self.start_states (same pattern FrozenLake uses
        for multi-'S' maps). jit/vmap-safe either way.
        """
        if len(self.start_states) == 1:
            state = jnp.int32(self.start_states[0])
        else:
            idx = jax.random.randint(key, shape=(), minval=0,
                                     maxval=len(self.start_states))
            state = self._start_states_arr[idx]
        return state, state

    def step(
        self,
        key: jax.Array,
        state: jax.Array,
        action: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
        """
        One step. Deterministic when slip_prob == 0; otherwise the executed outcome is
        sampled from the stored per-(state, action) outcome distribution.

        Returns obs, next_state, reward, done, info — obs == next_state (int32 index).
        """
        if self.n_outcomes == 1:
            outcome = jnp.int32(0)
        else:
            probs = self.outcome_probs[state, action]          # (K,)
            outcome = jax.random.choice(key, self.n_outcomes, p=probs)

        next_state = self.next_states[state, action, outcome]
        reward = self.rewards[state, action, outcome]
        done = self.dones[state, action, outcome]
        return next_state, next_state, reward, done, {}
