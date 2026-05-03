"""
Correctness tests for JAX FrozenLakeEnv against Gymnasium's FrozenLake-v1.

Every test that checks transition dynamics compares directly against
Gymnasium's env.P — the ground truth for this reimplementation.

Run with:
    conda run -n counterfactual python -m pytest tests/test_frozen_lake_env.py -v
"""

import sys
sys.path.insert(0, __file__.replace("tests/test_frozen_lake_env.py", "src"))

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym
import pytest

from counterfactual_rl.envs.frozen_lake import FrozenLakeEnv, MAPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gym_env(map_name="4x4", is_slippery=True):
    return gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=None,
    ).unwrapped


def our_env(map_name="4x4", is_slippery=True):
    return FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery)


# ---------------------------------------------------------------------------
# 1. Transition table parity with Gymnasium
# ---------------------------------------------------------------------------

class TestTransitionTableParity:
    """Compare env.P against Gymnasium's env.P entry by entry."""

    @pytest.mark.parametrize("map_name,is_slippery", [
        ("4x4", True),
        ("4x4", False),
        ("8x8", True),
        ("8x8", False),
    ])
    def test_p_table_matches_gymnasium(self, map_name, is_slippery):
        ref = gym_env(map_name, is_slippery)
        ours = our_env(map_name, is_slippery)

        n_states = ours.n_states
        n_actions = ours.n_actions

        for s in range(n_states):
            for a in range(n_actions):
                ref_outcomes = ref.P[s][a]
                our_outcomes = ours.P[s][a]

                assert len(ref_outcomes) == len(our_outcomes), (
                    f"map={map_name} slippery={is_slippery} s={s} a={a}: "
                    f"outcome count {len(our_outcomes)} != gymnasium {len(ref_outcomes)}"
                )

                for k, (ref_o, our_o) in enumerate(zip(ref_outcomes, our_outcomes)):
                    ref_prob, ref_ns, ref_r, ref_d = ref_o
                    our_prob, our_ns, our_r, our_d = our_o

                    assert abs(our_prob - ref_prob) < 1e-9, (
                        f"s={s} a={a} k={k}: prob {our_prob} != {ref_prob}")
                    assert our_ns == ref_ns, (
                        f"s={s} a={a} k={k}: next_state {our_ns} != {ref_ns}")
                    assert abs(our_r - ref_r) < 1e-9, (
                        f"s={s} a={a} k={k}: reward {our_r} != {ref_r}")
                    assert our_d == ref_d, (
                        f"s={s} a={a} k={k}: done {our_d} != {ref_d}")

    def test_n_states_matches_gymnasium(self):
        for map_name in ("4x4", "8x8"):
            ref = gym_env(map_name)
            ours = our_env(map_name)
            assert ours.n_states == ref.observation_space.n, (
                f"{map_name}: n_states {ours.n_states} != gymnasium {ref.observation_space.n}")

    def test_n_actions_matches_gymnasium(self):
        ref = gym_env()
        ours = our_env()
        assert ours.n_actions == ref.action_space.n


# ---------------------------------------------------------------------------
# 2. JAX arrays match the P dict
# ---------------------------------------------------------------------------

class TestJaxTableConsistency:
    """Verify that the JAX arrays stored on the env are consistent with env.P."""

    @pytest.mark.parametrize("map_name,is_slippery", [
        ("4x4", True),
        ("4x4", False),
        ("8x8", True),
    ])
    def test_jax_arrays_match_p_dict(self, map_name, is_slippery):
        env = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery)
        n, a = env.n_states, env.n_actions
        ns_arr = np.array(env.next_states)
        rew_arr = np.array(env.rewards)
        done_arr = np.array(env.dones)

        for s in range(n):
            for act in range(a):
                outcomes = env.P[s][act]
                for k, (prob, ns, r, d) in enumerate(outcomes):
                    assert ns_arr[s, act, k] == ns
                    assert abs(rew_arr[s, act, k] - r) < 1e-9
                    assert bool(done_arr[s, act, k]) == d


# ---------------------------------------------------------------------------
# 3. Reset behaviour
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_returns_start_state_4x4(self):
        env = FrozenLakeEnv("4x4")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        # Default 4x4 map has S at position 0
        assert int(obs) == 0
        assert int(state) == 0
        assert obs.dtype == jnp.int32

    def test_reset_returns_start_state_8x8(self):
        env = FrozenLakeEnv("8x8")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        assert int(obs) == 0
        assert int(state) == 0

    def test_obs_equals_state(self):
        """Gymnasium contract: observation is the state integer."""
        env = FrozenLakeEnv("4x4")
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        assert int(obs) == int(state)

    def test_reset_multiple_keys_deterministic(self):
        """4x4 has one S tile — reset should always return 0."""
        env = FrozenLakeEnv("4x4")
        for seed in range(10):
            obs, state = env.reset(jax.random.PRNGKey(seed))
            assert int(state) == 0


# ---------------------------------------------------------------------------
# 4. Step — non-slippery (deterministic, easy to verify)
# ---------------------------------------------------------------------------

class TestStepNonSlippery:

    def setup_method(self):
        self.env = FrozenLakeEnv("4x4", is_slippery=False)
        self.ref = gym_env("4x4", is_slippery=False)
        self.key = jax.random.PRNGKey(0)

    def _ref_step(self, state, action):
        """Get the single deterministic outcome from Gymnasium's P table."""
        outcomes = self.ref.P[state][action]
        assert len(outcomes) == 1
        return outcomes[0]  # (prob, next_state, reward, done)

    @pytest.mark.parametrize("state,action", [
        (0, 0),   # S, LEFT → stays at 0 (boundary)
        (0, 1),   # S, DOWN → 4
        (0, 2),   # S, RIGHT → 1
        (0, 3),   # S, UP → stays at 0 (boundary)
        (1, 1),   # F, DOWN → 5 (hole)
        (14, 2),  # F, RIGHT → 15 (goal)
        (5, 0),   # H, LEFT → stays (terminal)
        (15, 1),  # G, DOWN → stays (terminal)
    ])
    def test_step_matches_gymnasium(self, state, action):
        _, ref_ns, ref_r, ref_d = self._ref_step(state, action)
        obs, ns, r, d, _ = self.env.step(
            self.key, jnp.int32(state), jnp.int32(action)
        )
        assert int(ns) == ref_ns, f"s={state} a={action}: next_state {int(ns)} != {ref_ns}"
        assert abs(float(r) - ref_r) < 1e-9
        assert bool(d) == ref_d
        assert int(obs) == int(ns)

    def test_reaching_goal_gives_reward_1(self):
        """State 14, action RIGHT (2) leads to goal (state 15)."""
        _, ns, r, d, _ = self.env.step(self.key, jnp.int32(14), jnp.int32(2))
        assert int(ns) == 15
        assert float(r) == 1.0
        assert bool(d) is True

    def test_reaching_hole_gives_reward_0(self):
        """State 1, action DOWN (1) leads to hole (state 5)."""
        _, ns, r, d, _ = self.env.step(self.key, jnp.int32(1), jnp.int32(1))
        assert int(ns) == 5
        assert float(r) == 0.0
        assert bool(d) is True

    def test_frozen_tile_gives_reward_0_not_done(self):
        """State 0, action RIGHT (2) leads to frozen tile (state 1)."""
        _, ns, r, d, _ = self.env.step(self.key, jnp.int32(0), jnp.int32(2))
        assert int(ns) == 1
        assert float(r) == 0.0
        assert bool(d) is False

    def test_boundary_clamping_left_at_col_0(self):
        """State 0, action LEFT — should stay at 0."""
        _, ns, _, _, _ = self.env.step(self.key, jnp.int32(0), jnp.int32(0))
        assert int(ns) == 0

    def test_boundary_clamping_up_at_row_0(self):
        """State 0, action UP — should stay at 0."""
        _, ns, _, _, _ = self.env.step(self.key, jnp.int32(0), jnp.int32(3))
        assert int(ns) == 0

    def test_terminal_state_loops_to_itself(self):
        """Hole state 5: any action keeps state at 5."""
        for action in range(4):
            _, ns, r, d, _ = self.env.step(
                self.key, jnp.int32(5), jnp.int32(action)
            )
            assert int(ns) == 5
            assert float(r) == 0.0
            assert bool(d) is True

    def test_goal_state_loops_to_itself(self):
        """Goal state 15: any action keeps state at 15, reward=0 (Gymnasium behaviour)."""
        for action in range(4):
            _, ns, r, d, _ = self.env.step(
                self.key, jnp.int32(15), jnp.int32(action)
            )
            assert int(ns) == 15
            assert float(r) == 0.0
            assert bool(d) is True


# ---------------------------------------------------------------------------
# 5. Step — slippery (stochastic, verify distribution)
# ---------------------------------------------------------------------------

class TestStepSlippery:

    def setup_method(self):
        self.env = FrozenLakeEnv("4x4", is_slippery=True)
        self.ref = gym_env("4x4", is_slippery=True)

    def test_outcomes_are_subset_of_gymnasium_outcomes(self):
        """
        For every (state, action), the set of possible next states from our
        step() should match Gymnasium's P table.
        """
        n_samples = 300
        for s in range(self.env.n_states):
            for a in range(self.env.n_actions):
                ref_next = {ns for _, ns, _, _ in self.ref.P[s][a]}

                seen = set()
                key = jax.random.PRNGKey(0)
                for i in range(n_samples):
                    key, sk = jax.random.split(key)
                    _, ns, _, _, _ = self.env.step(sk, jnp.int32(s), jnp.int32(a))
                    seen.add(int(ns))

                # All observed next states must be in Gymnasium's set
                assert seen <= ref_next, (
                    f"s={s} a={a}: got next states {seen}, gymnasium allows {ref_next}"
                )

    def test_empirical_distribution_matches_gymnasium(self):
        """
        Each outcome should occur with probability ~1/3.
        Sample 3000 times and check empirical counts are within 5σ.
        """
        n_samples = 3000
        # Use state=0, action=DOWN (1) — guaranteed 3 distinct outcomes on 4x4
        s, a = 0, 1
        ref_outcomes = self.ref.P[s][a]
        expected_ns = [ns for _, ns, _, _ in ref_outcomes]

        counts = {ns: 0 for ns in expected_ns}
        key = jax.random.PRNGKey(99)
        for _ in range(n_samples):
            key, sk = jax.random.split(key)
            _, ns, _, _, _ = self.env.step(sk, jnp.int32(s), jnp.int32(a))
            ns_int = int(ns)
            if ns_int in counts:
                counts[ns_int] += 1

        for ns, count in counts.items():
            empirical = count / n_samples
            # Expected ~1/3, 5σ tolerance: σ = sqrt(p(1-p)/n) ≈ 0.0086 → 5σ ≈ 0.043
            assert abs(empirical - 1/3) < 0.05, (
                f"s={s} a={a} ns={ns}: empirical prob {empirical:.3f} expected ~0.333"
            )

    def test_slippery_terminal_states_always_loop(self):
        """Terminal states must loop regardless of slippery flag."""
        key = jax.random.PRNGKey(0)
        for s in [5, 7, 11, 12, 15]:  # holes + goal on 4x4
            for a in range(4):
                for _ in range(10):
                    key, sk = jax.random.split(key)
                    _, ns, _, d, _ = self.env.step(sk, jnp.int32(s), jnp.int32(a))
                    assert int(ns) == s
                    assert bool(d) is True


# ---------------------------------------------------------------------------
# 6. Custom map (arbitrary size)
# ---------------------------------------------------------------------------

class TestCustomMap:

    def test_3x3_custom_map(self):
        """Minimal 3x3 map: verify n_states and start state."""
        desc = ["SFF", "FHF", "FFG"]
        env = FrozenLakeEnv(desc=desc)
        assert env.n_states == 9
        assert env.n_actions == 4
        assert env.start_states == [0]
        assert env.nrows == 3
        assert env.ncols == 3

    def test_custom_map_goal_reachable(self):
        """On a 3x3 non-slippery map, step right twice then down twice reaches goal."""
        desc = ["SFF", "FFF", "FFG"]
        env = FrozenLakeEnv(desc=desc, is_slippery=False)
        key = jax.random.PRNGKey(0)

        _, state = env.reset(key)
        assert int(state) == 0

        # RIGHT → state 1
        _, state, r, d, _ = env.step(key, state, jnp.int32(2))
        assert int(state) == 1 and not d

        # RIGHT → state 2
        _, state, r, d, _ = env.step(key, state, jnp.int32(2))
        assert int(state) == 2 and not d

        # DOWN → state 5
        _, state, r, d, _ = env.step(key, state, jnp.int32(1))
        assert int(state) == 5 and not d

        # DOWN → state 8 (goal)
        _, state, r, d, _ = env.step(key, state, jnp.int32(1))
        assert int(state) == 8
        assert float(r) == 1.0
        assert bool(d) is True

    def test_custom_map_boundary_matches_size(self):
        """Boundary clamping should respect the custom map dimensions."""
        desc = ["SF", "FG"]
        env = FrozenLakeEnv(desc=desc, is_slippery=False)
        key = jax.random.PRNGKey(0)

        # State 0, UP → stays at 0
        _, ns, _, _, _ = env.step(key, jnp.int32(0), jnp.int32(3))
        assert int(ns) == 0

        # State 0, LEFT → stays at 0
        _, ns, _, _, _ = env.step(key, jnp.int32(0), jnp.int32(0))
        assert int(ns) == 0

        # State 1, RIGHT → stays at 1 (right boundary)
        _, ns, _, _, _ = env.step(key, jnp.int32(1), jnp.int32(2))
        assert int(ns) == 1

    def test_custom_map_p_table_has_correct_shape(self):
        desc = ["SFF", "FHF", "FFG"]
        env = FrozenLakeEnv(desc=desc, is_slippery=True)
        terminal = set()
        for r, row in enumerate(desc):
            for c, tile in enumerate(row):
                if tile in ("G", "H"):
                    terminal.add(r * len(row) + c)

        for s in range(env.n_states):
            for a in range(env.n_actions):
                expected = 1 if s in terminal else 3
                assert len(env.P[s][a]) == expected, (
                    f"s={s} a={a}: expected {expected} outcomes, got {len(env.P[s][a])}"
                )


# ---------------------------------------------------------------------------
# 7. JAX compatibility — jit and vmap
# ---------------------------------------------------------------------------

class TestJaxCompatibility:

    def test_step_is_jittable(self):
        env = FrozenLakeEnv("4x4", is_slippery=False)

        @jax.jit
        def jit_step(key, state, action):
            return env.step(key, state, action)

        key = jax.random.PRNGKey(0)
        obs, ns, r, d, _ = jit_step(key, jnp.int32(0), jnp.int32(2))
        assert int(ns) == 1  # RIGHT from state 0

    def test_step_slippery_is_jittable(self):
        env = FrozenLakeEnv("4x4", is_slippery=True)

        @jax.jit
        def jit_step(key, state, action):
            return env.step(key, state, action)

        key = jax.random.PRNGKey(0)
        obs, ns, r, d, _ = jit_step(key, jnp.int32(0), jnp.int32(1))
        assert int(ns) in {0, 1, 4}  # valid outcomes for DOWN from state 0

    def test_step_is_vmappable_over_states(self):
        """Run step for all 16 states simultaneously."""
        env = FrozenLakeEnv("4x4", is_slippery=False)

        @jax.vmap
        def batched_step(state):
            key = jax.random.PRNGKey(0)
            return env.step(key, state, jnp.int32(2))  # RIGHT for all states

        states = jnp.arange(16, dtype=jnp.int32)
        obs, ns, r, d, _ = batched_step(states)
        assert obs.shape == (16,)
        assert ns.shape == (16,)

    def test_step_is_vmappable_over_actions(self):
        """Run all 4 actions from state 0 simultaneously."""
        env = FrozenLakeEnv("4x4", is_slippery=False)

        @jax.vmap
        def batched_step(action):
            key = jax.random.PRNGKey(0)
            return env.step(key, jnp.int32(0), action)

        actions = jnp.arange(4, dtype=jnp.int32)
        obs, ns, r, d, _ = batched_step(actions)
        assert ns.shape == (4,)

        # Verify against P table
        for a in range(4):
            expected_ns = env.P[0][a][0][1]
            assert int(ns[a]) == expected_ns

    def test_reset_is_jittable(self):
        env = FrozenLakeEnv("4x4")

        @jax.jit
        def jit_reset(key):
            return env.reset(key)

        key = jax.random.PRNGKey(0)
        obs, state = jit_reset(key)
        assert int(state) == 0

    def test_vmap_over_batch_of_keys(self):
        """Simulate a batch of independent environments."""
        env = FrozenLakeEnv("4x4", is_slippery=True)
        n_envs = 32

        @jax.jit
        def run_batch(keys):
            def single(key):
                obs, state = env.reset(key)
                obs2, state2, r, d, _ = env.step(key, state, jnp.int32(1))
                return state2

            return jax.vmap(single)(keys)

        keys = jax.random.split(jax.random.PRNGKey(7), n_envs)
        states = run_batch(keys)
        assert states.shape == (n_envs,)
        # All outcomes must be valid next states for DOWN from state 0
        valid = {ns for _, ns, _, _ in env.P[0][1]}
        for s in states:
            assert int(s) in valid


# ---------------------------------------------------------------------------
# 8. 8×8 map — spot checks
# ---------------------------------------------------------------------------

class TestEightByEight:

    def setup_method(self):
        self.env = FrozenLakeEnv("8x8", is_slippery=False)
        self.ref = gym_env("8x8", is_slippery=False)

    def test_start_state_is_0(self):
        obs, state = self.env.reset(jax.random.PRNGKey(0))
        assert int(state) == 0

    def test_goal_state_is_63(self):
        key = jax.random.PRNGKey(0)
        # State 62, action RIGHT → 63 (goal)
        _, ns, r, d, _ = self.env.step(key, jnp.int32(62), jnp.int32(2))
        assert int(ns) == 63
        assert float(r) == 1.0
        assert bool(d) is True

    def test_full_p_table_matches_gymnasium(self):
        ref = gym_env("8x8", is_slippery=False)
        for s in range(64):
            for a in range(4):
                ref_o = ref.P[s][a]
                our_o = self.env.P[s][a]
                assert len(ref_o) == len(our_o)
                for (rp, rns, rr, rd), (op, ons, or_, od) in zip(ref_o, our_o):
                    assert rns == ons
                    assert abs(rr - or_) < 1e-9
                    assert rd == od


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
