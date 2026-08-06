"""
Correctness tests for the JAX DoorKeyEnv.

DoorKey is OUR tabular reimplementation of MiniGrid DoorKey (see envs/doorkey.py for
why we reimplement rather than import). There is no external ground-truth P table to
diff against (MiniGrid exposes none), so these tests verify internal consistency,
hand-worked transitions against the documented action semantics, and solvability.

Run with:
    conda run -n counterfactual python -m pytest tests/test_doorkey_env.py -v
"""

import collections
import sys
sys.path.insert(0, __file__.replace("tests/test_doorkey_env.py", "src"))

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from counterfactual_rl.envs.doorkey import (
    DoorKeyEnv, LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE,
    DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP,
    DOOR_LOCKED, DOOR_CLOSED, DOOR_OPEN,
)

KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def env():
    return DoorKeyEnv("6x6")


@pytest.fixture(scope="module")
def env_lava():
    return DoorKeyEnv("6x6_lava")


def _step_tuple(env, state_tuple, action):
    """Step from a (row,col,dir,has_key,door) tuple; return (next_tuple, reward, done)."""
    i = env._index[state_tuple]
    _, ns, r, d, _ = env.step(KEY, jnp.int32(i), jnp.int32(action))
    return env._order[int(ns)], float(r), bool(d)


def _find_state_facing(env, row, col, direction):
    """Find any reachable state at (row,col) facing `direction`, regardless of has_key/door
    (turning is always available, so if the cell is reachable at all, every facing is too)."""
    for s in env._order:
        if s[0] == row and s[1] == col and s[2] == direction:
            return s
    return None


# ---------------------------------------------------------------------------
# 1. Structure / enumeration
# ---------------------------------------------------------------------------

class TestStructure:

    def test_basic_attrs(self, env):
        assert env.n_actions == 7
        assert env.n_states == len(env._order)
        assert env.start_states == [0]
        assert env._order[0] == (1, 1, DIR_RIGHT, 0, DOOR_LOCKED)

    def test_special_cells(self, env):
        assert env.key_cell == (3, 1)
        assert env.door_cell == (2, 3)
        assert env.goal_cell == (4, 4)

    def test_all_door_and_key_states_reachable(self, env):
        assert sorted({s[4] for s in env._order}) == [DOOR_LOCKED, DOOR_CLOSED, DOOR_OPEN]
        assert sorted({s[3] for s in env._order}) == [0, 1]

    def test_non_terminal_excludes_goal(self, env):
        for si in env.non_terminal:
            assert env.state_to_cell[si] != env.goal_cell
        # every enumerated state is either terminal (on goal) or non-terminal
        n_term = sum(1 for si in range(env.n_states)
                     if env.state_to_cell[si] == env.goal_cell)
        assert n_term + len(env.non_terminal) == env.n_states

    def test_transition_closure(self, env):
        """Every next-state index produced by step is a valid enumerated state."""
        for si in range(env.n_states):
            for a in range(env.n_actions):
                nsi = int(env.next_states[si, a, 0])
                assert 0 <= nsi < env.n_states


# ---------------------------------------------------------------------------
# 2. Hand-worked transitions (against documented action semantics)
# ---------------------------------------------------------------------------

class TestHandWorkedTransitions:

    def test_turn_left(self, env):
        ns, _, _ = _step_tuple(env, (1, 1, DIR_RIGHT, 0, DOOR_LOCKED), LEFT)
        assert ns == (1, 1, DIR_UP, 0, DOOR_LOCKED)

    def test_turn_right(self, env):
        ns, _, _ = _step_tuple(env, (1, 1, DIR_RIGHT, 0, DOOR_LOCKED), RIGHT)
        assert ns == (1, 1, DIR_DOWN, 0, DOOR_LOCKED)

    def test_forward_onto_floor(self, env):
        ns, _, _ = _step_tuple(env, (1, 1, DIR_RIGHT, 0, DOOR_LOCKED), FORWARD)
        assert ns == (1, 2, DIR_RIGHT, 0, DOOR_LOCKED)

    def test_forward_into_wall_is_noop(self, env):
        ns, _, _ = _step_tuple(env, (1, 1, DIR_UP, 0, DOOR_LOCKED), FORWARD)
        assert ns == (1, 1, DIR_UP, 0, DOOR_LOCKED)

    def test_forward_into_locked_door_is_noop(self, env):
        ns, _, _ = _step_tuple(env, (2, 2, DIR_RIGHT, 1, DOOR_LOCKED), FORWARD)
        assert ns == (2, 2, DIR_RIGHT, 1, DOOR_LOCKED)

    def test_forward_onto_grounded_key_blocked(self, env):
        ns, _, _ = _step_tuple(env, (2, 1, DIR_DOWN, 0, DOOR_LOCKED), FORWARD)
        assert ns == (2, 1, DIR_DOWN, 0, DOOR_LOCKED)

    def test_pickup_key(self, env):
        ns, _, _ = _step_tuple(env, (2, 1, DIR_DOWN, 0, DOOR_LOCKED), PICKUP)
        assert ns == (2, 1, DIR_DOWN, 1, DOOR_LOCKED)

    def test_forward_onto_picked_key_cell(self, env):
        ns, _, _ = _step_tuple(env, (2, 1, DIR_DOWN, 1, DOOR_LOCKED), FORWARD)
        assert ns == (3, 1, DIR_DOWN, 1, DOOR_LOCKED)

    def test_toggle_locked_door_without_key_noop(self, env):
        ns, _, _ = _step_tuple(env, (2, 2, DIR_RIGHT, 0, DOOR_LOCKED), TOGGLE)
        assert ns == (2, 2, DIR_RIGHT, 0, DOOR_LOCKED)

    def test_toggle_locked_door_with_key_opens(self, env):
        ns, _, _ = _step_tuple(env, (2, 2, DIR_RIGHT, 1, DOOR_LOCKED), TOGGLE)
        assert ns == (2, 2, DIR_RIGHT, 1, DOOR_OPEN)

    def test_toggle_open_door_closes(self, env):
        ns, _, _ = _step_tuple(env, (2, 2, DIR_RIGHT, 1, DOOR_OPEN), TOGGLE)
        assert ns == (2, 2, DIR_RIGHT, 1, DOOR_CLOSED)

    def test_forward_through_open_door(self, env):
        ns, _, _ = _step_tuple(env, (2, 2, DIR_RIGHT, 1, DOOR_OPEN), FORWARD)
        assert ns == (2, 3, DIR_RIGHT, 1, DOOR_OPEN)

    def test_forward_onto_goal_rewards_and_terminates(self, env):
        ns, r, d = _step_tuple(env, (3, 4, DIR_DOWN, 1, DOOR_OPEN), FORWARD)
        assert ns == (4, 4, DIR_DOWN, 1, DOOR_OPEN)
        assert r == 1.0 and d is True

    def test_drop_is_noop(self, env):
        ns, _, _ = _step_tuple(env, (2, 1, DIR_DOWN, 1, DOOR_LOCKED), DROP)
        assert ns == (2, 1, DIR_DOWN, 1, DOOR_LOCKED)

    def test_done_is_noop(self, env):
        ns, _, _ = _step_tuple(env, (2, 1, DIR_DOWN, 1, DOOR_LOCKED), DONE)
        assert ns == (2, 1, DIR_DOWN, 1, DOOR_LOCKED)

    def test_goal_state_absorbs(self, env):
        for a in range(env.n_actions):
            ns, r, d = _step_tuple(env, (4, 4, DIR_DOWN, 1, DOOR_OPEN), a)
            assert ns == (4, 4, DIR_DOWN, 1, DOOR_OPEN)
            assert r == 0.0 and d is True


# ---------------------------------------------------------------------------
# 3. P dict <-> JAX arrays <-> step consistency
# ---------------------------------------------------------------------------

class TestConsistency:

    def test_p_matches_arrays(self, env):
        ns_arr = np.array(env.next_states)
        rew_arr = np.array(env.rewards)
        done_arr = np.array(env.dones)
        for si in range(env.n_states):
            for a in range(env.n_actions):
                (prob, nsi, r, term) = env.P[si][a][0]
                assert prob == 1.0
                assert ns_arr[si, a, 0] == nsi
                assert abs(rew_arr[si, a, 0] - r) < 1e-9
                assert bool(done_arr[si, a, 0]) == term

    def test_p_matches_step(self, env):
        for si in range(env.n_states):
            for a in range(env.n_actions):
                (prob, nsi, r, term) = env.P[si][a][0]
                _, ns2, r2, d2, _ = env.step(KEY, jnp.int32(si), jnp.int32(a))
                assert int(ns2) == nsi
                assert abs(float(r2) - r) < 1e-9
                assert bool(d2) == term


# ---------------------------------------------------------------------------
# 4. Reset / obs contract
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_returns_start(self, env):
        obs, state = env.reset(KEY)
        assert int(obs) == int(state) == env.start_states[0]
        assert obs.dtype == jnp.int32

    def test_reset_deterministic(self, env):
        for seed in range(5):
            _, state = env.reset(jax.random.PRNGKey(seed))
            assert int(state) == env.start_states[0]


# ---------------------------------------------------------------------------
# 5. Solvability
# ---------------------------------------------------------------------------

def _solve_and_verify(env, max_len):
    """BFS shortest path to the goal, replay it, assert it wins. Lava states are terminal
    absorbing dead-ends (no outgoing edge continues past them), so any path this BFS finds
    to the goal is automatically lava-free — no separate lava-avoidance check needed."""
    start = env.start_states[0]
    prev = {start: (None, None)}
    dq = collections.deque([start])
    goal_idx = None
    while dq:
        u = dq.popleft()
        if env.state_to_cell[u] == env.goal_cell:
            goal_idx = u
            break
        for a in range(env.n_actions):
            v = int(env.next_states[u, a, 0])
            if v not in prev:
                prev[v] = (u, a)
                dq.append(v)
    assert goal_idx is not None, "goal not reachable"

    path = []
    cur = goal_idx
    while prev[cur][0] is not None:
        u, a = prev[cur]
        path.append(a)
        cur = u
    path.reverse()

    s = jnp.int32(start)
    total_r, done = 0.0, False
    for a in path:
        _, s, r, done, _ = env.step(KEY, s, jnp.int32(a))
        total_r += float(r)
    assert bool(done) is True
    assert total_r == 1.0
    assert len(path) <= max_len
    return path


class TestSolvability:

    def test_goal_reachable_and_solution_rewards_one(self, env):
        _solve_and_verify(env, max_len=20)  # 6x6, optimal is 11

    def test_lava_goal_reachable_avoiding_lava(self, env_lava):
        _solve_and_verify(env_lava, max_len=20)  # 6x6+lava, safe route is 11 steps


# ---------------------------------------------------------------------------
# 6. Lava (6x6_lava layout only)
# ---------------------------------------------------------------------------

class TestLava:

    def test_lava_cells_present(self, env_lava):
        assert set(env_lava.lava_cells) == {(4, 2)}

    @pytest.mark.parametrize("lava_cell,approach_cell,facing", [
        ((4, 2), (3, 2), DIR_DOWN),    # from the corridor cell the agent crosses en route
        ((4, 2), (4, 1), DIR_RIGHT),   # from below the key, the other live approach
    ])
    def test_forward_into_lava_is_terminal_zero_reward(self, env_lava, lava_cell, approach_cell, facing):
        s = _find_state_facing(env_lava, approach_cell[0], approach_cell[1], facing)
        assert s is not None, f"no reachable state found at {approach_cell} facing {facing}"
        ns, r, d = _step_tuple(env_lava, s, FORWARD)
        assert (ns[0], ns[1]) == lava_cell
        assert r == 0.0
        assert d is True

    def test_lava_state_absorbs(self, env_lava):
        """Any reachable state sitting AT a lava cell must loop to itself under every action."""
        lava_states = [s for s in env_lava._order if (s[0], s[1]) in env_lava.lava_cells]
        assert lava_states, "no lava states were ever entered during BFS enumeration"
        for s in lava_states:
            for a in range(env_lava.n_actions):
                ns, r, d = _step_tuple(env_lava, s, a)
                assert ns == s
                assert r == 0.0
                assert d is True

    def test_lava_states_excluded_from_non_terminal(self, env_lava):
        lava_indices = {env_lava._index[s] for s in env_lava._order if (s[0], s[1]) in env_lava.lava_cells}
        assert lava_indices, "expected at least one enumerated lava state"
        assert lava_indices.isdisjoint(set(env_lava.non_terminal))


# ---------------------------------------------------------------------------
# 7. JAX compatibility
# ---------------------------------------------------------------------------

class TestJaxCompatibility:

    def test_step_jittable(self, env):
        jstep = jax.jit(lambda st, a: env.step(KEY, st, a))
        _, ns, r, d, _ = jstep(jnp.int32(0), jnp.int32(FORWARD))
        assert int(ns) == env.P[0][FORWARD][0][1]

    def test_step_vmappable_over_states_lava(self, env_lava):
        f = jax.vmap(lambda st: env_lava.step(KEY, st, jnp.int32(FORWARD))[1])
        states = jnp.arange(env_lava.n_states, dtype=jnp.int32)
        out = f(states)
        assert out.shape == (env_lava.n_states,)
        for s in range(env_lava.n_states):
            assert int(out[s]) == env_lava.P[s][FORWARD][0][1]

    def test_step_vmappable_over_states(self, env):
        f = jax.vmap(lambda st: env.step(KEY, st, jnp.int32(FORWARD))[1])
        states = jnp.arange(env.n_states, dtype=jnp.int32)
        out = f(states)
        assert out.shape == (env.n_states,)
        for s in range(env.n_states):
            assert int(out[s]) == env.P[s][FORWARD][0][1]

    def test_step_vmappable_over_actions(self, env):
        f = jax.vmap(lambda a: env.step(KEY, jnp.int32(0), a)[1])
        actions = jnp.arange(env.n_actions, dtype=jnp.int32)
        out = f(actions)
        assert out.shape == (env.n_actions,)
        for a in range(env.n_actions):
            assert int(out[a]) == env.P[0][a][0][1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
