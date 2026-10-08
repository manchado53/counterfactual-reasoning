"""
Correctness tests for the neural CVRP env (the RP-DQN benchmark).

Run:  conda run -n counterfactual python -m pytest tests/test_cvrp_neural_env.py -v

There is no reference implementation to diff against the way
``test_frozen_lake_env.py`` diffs against Gymnasium, so these instead pin the
problem's own invariants -- capacity, coverage, the reward identity -- plus the
jit/vmap safety that every vectorized trainer in this repo depends on.

The reward identity is the load-bearing one. RP-DQN trains with
``discount_factor: 1`` and a per-leg negative-distance reward, so the undiscounted
return must equal the negative tour length exactly. If that drifts, our numbers
stop being comparable to their Table 1 and the whole benchmark argument collapses.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import jax
import jax.numpy as jnp
import numpy as np

from counterfactual_rl.envs.cvrp_neural import (
    CVRPNeuralEnv, CVRPState, N_FEATURES, default_capacity,
    _F_DEMAND, _F_VISITED, _F_CURRENT, _F_DEPOT, _F_LOAD,
)

SIZES = [5, 20, 50]


def rollout_random_legal(env, key, max_iters=None):
    """Play one episode choosing uniformly among legal actions. Returns a trace."""
    obs, state = env.reset(key)
    tour = [0]
    rewards = []
    loads = []
    k = key
    for _ in range(max_iters or env.max_steps + 5):
        mask = np.array(env.action_mask(state))
        k, ak, sk = jax.random.split(k, 3)
        action = int(jax.random.choice(ak, jnp.asarray(np.flatnonzero(mask))))
        obs, state, reward, done, _ = env.step(sk, state, action)
        tour.append(action)
        rewards.append(float(reward))
        loads.append(float(state.load))
        if bool(done):
            break
    return state, tour, rewards, loads, bool(done)


class TestCapacityRule:
    """Capacity comes from their DataGenerator, not from us."""

    @pytest.mark.parametrize("n,expected", [(10, 20.0), (20, 30.0), (50, 40.0), (100, 50.0)])
    def test_default_capacity(self, n, expected):
        assert default_capacity(n) == expected

    def test_env_uses_the_rule(self):
        assert CVRPNeuralEnv(n_customers=20).capacity == 30.0

    def test_capacity_can_be_overridden(self):
        assert CVRPNeuralEnv(n_customers=20, capacity=99.0).capacity == 99.0


class TestReset:
    @pytest.mark.parametrize("n", SIZES)
    def test_shapes_and_dtypes(self, n):
        env = CVRPNeuralEnv(n_customers=n)
        obs, st = env.reset(jax.random.PRNGKey(0))
        assert obs.shape == (n + 1, N_FEATURES) == env.obs_shape
        assert obs.dtype == jnp.float32
        assert st.coords.shape == (n + 1, 2)
        assert st.demands.shape == (n + 1,)
        assert st.visited.shape == (n + 1,)
        assert st.visited.dtype == bool

    def test_starts_at_depot_fully_loaded(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(3))
        assert int(st.current) == 0
        assert float(st.load) == 1.0
        assert int(st.step) == 0
        assert not bool(st.done)
        assert not np.array(st.visited).any()

    def test_depot_has_no_demand(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(4))
        assert float(st.demands[0]) == 0.0

    @pytest.mark.parametrize("n", SIZES)
    def test_demands_are_integers_1_to_9_over_capacity(self, n):
        env = CVRPNeuralEnv(n_customers=n)
        _, st = env.reset(jax.random.PRNGKey(5))
        raw = np.array(st.demands)[1:] * env.capacity
        assert np.allclose(raw, np.round(raw), atol=1e-4)
        assert raw.min() >= 1 - 1e-4 and raw.max() <= 9 + 1e-4

    def test_coordinates_in_unit_square(self):
        env = CVRPNeuralEnv(n_customers=50)
        _, st = env.reset(jax.random.PRNGKey(6))
        c = np.array(st.coords)
        assert c.min() >= 0.0 and c.max() <= 1.0

    def test_same_key_same_instance(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, a = env.reset(jax.random.PRNGKey(7))
        _, b = env.reset(jax.random.PRNGKey(7))
        assert np.array_equal(np.array(a.coords), np.array(b.coords))
        assert np.array_equal(np.array(a.demands), np.array(b.demands))

    def test_fresh_instance_every_reset(self):
        """Their protocol: every problem instance is used exactly once."""
        env = CVRPNeuralEnv(n_customers=20)
        _, a = env.reset(jax.random.PRNGKey(8))
        _, b = env.reset(jax.random.PRNGKey(9))
        assert not np.array_equal(np.array(a.coords), np.array(b.coords))


class TestActionMask:
    def test_depot_illegal_while_parked_on_it(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(10))
        assert not bool(env.action_mask(st)[0])

    def test_all_customers_reachable_from_a_full_truck(self):
        """Max demand is 9/30 = 0.3, so nothing is ever infeasible at the depot.

        This is what makes the episode deadlock-free.
        """
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(11))
        assert np.array(env.action_mask(st))[1:].all()

    def test_served_customer_becomes_illegal(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(12))
        _, st, _, _, _ = env.step(jax.random.PRNGKey(0), st, 3)
        assert not bool(env.action_mask(st)[3])

    def test_overweight_customer_becomes_illegal(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(13))

        def blocked_customers(state):
            """Unserved customers that no longer fit on the truck."""
            over = np.array(state.demands) > float(state.load)
            unserved = ~np.array(state.visited)
            idx = np.flatnonzero(over & unserved)
            return idx[idx > 0]

        # Drain the truck one legal customer at a time until something no longer
        # fits. Note the stopping condition is specifically "infeasible", not
        # "illegal" -- a served customer is illegal too, and stopping on that would
        # make this test pass without ever exercising the capacity rule.
        for _ in range(env.max_steps):
            if blocked_customers(st).size:
                break
            mask = np.array(env.action_mask(st))
            legal = [a for a in np.flatnonzero(mask) if a != 0]
            assert legal, "ran out of customers before capacity ever bound"
            _, st, _, _, _ = env.step(jax.random.PRNGKey(0), st, int(legal[0]))

        blocked = blocked_customers(st)
        assert blocked.size > 0, "never managed to make a customer infeasible"
        assert not np.array(env.action_mask(st))[blocked].any()
        # ...and the depot must stay available so the truck can always reload.
        assert bool(env.action_mask(st)[0])

    @pytest.mark.parametrize("n", SIZES)
    def test_obs_derived_mask_matches_state_mask(self, n):
        """The rollout only carries obs, so the two must never disagree."""
        env = CVRPNeuralEnv(n_customers=n)
        obs, st = env.reset(jax.random.PRNGKey(14))
        k = jax.random.PRNGKey(15)
        for _ in range(env.max_steps):
            m_state = np.array(env.action_mask(st))
            m_obs = np.array(env.action_mask_from_obs(obs))
            assert np.array_equal(m_state, m_obs)
            k, ak, sk = jax.random.split(k, 3)
            a = int(jax.random.choice(ak, jnp.asarray(np.flatnonzero(m_state))))
            obs, st, _, done, _ = env.step(sk, st, a)
            if bool(done):
                break

    def test_mask_is_batch_safe(self):
        env = CVRPNeuralEnv(n_customers=20)
        keys = jax.random.split(jax.random.PRNGKey(16), 6)
        obs_b = jnp.stack([env.reset(k)[0] for k in keys])
        batched = np.array(env.action_mask_from_obs(obs_b))
        assert batched.shape == (6, 21)
        for i, k in enumerate(keys):
            assert np.array_equal(batched[i], np.array(env.action_mask_from_obs(env.reset(k)[0])))


class TestStep:
    def test_reward_is_negative_leg_distance(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(20))
        before = np.array(st.coords[st.current])
        _, st2, r, _, _ = env.step(jax.random.PRNGKey(0), st, 7)
        expected = -np.linalg.norm(np.array(st.coords[7]) - before)
        assert float(r) == pytest.approx(expected, abs=1e-5)

    def test_serving_marks_visited_and_consumes_load(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(21))
        d = float(st.demands[5])
        _, st2, _, _, _ = env.step(jax.random.PRNGKey(0), st, 5)
        assert bool(st2.visited[5])
        assert int(st2.current) == 5
        assert float(st2.load) == pytest.approx(1.0 - d, abs=1e-6)

    def test_depot_refills_and_does_not_mark_visited(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(22))
        _, st, _, _, _ = env.step(jax.random.PRNGKey(0), st, 5)
        assert float(st.load) < 1.0
        _, st, _, _, _ = env.step(jax.random.PRNGKey(0), st, 0)
        assert float(st.load) == 1.0
        assert not bool(st.visited[0])
        assert int(st.current) == 0

    def test_terminal_only_when_all_served_and_back_at_depot(self):
        env = CVRPNeuralEnv(n_customers=5)
        st, tour, _, _, done = rollout_random_legal(env, jax.random.PRNGKey(23))
        assert done
        assert np.array(st.visited)[1:].all()
        assert tour[-1] == 0

    def test_terminal_self_loops_with_zero_reward(self):
        env = CVRPNeuralEnv(n_customers=5)
        st, _, _, _, _ = rollout_random_legal(env, jax.random.PRNGKey(24))
        assert bool(st.done)
        _, st2, r, done, _ = env.step(jax.random.PRNGKey(0), st, 3)
        assert float(r) == 0.0
        assert bool(done)
        assert int(st2.current) == int(st.current)
        assert np.array_equal(np.array(st2.visited), np.array(st.visited))


class TestEpisodeInvariants:
    """The properties the plan named, checked over many random-legal episodes."""

    @pytest.mark.parametrize("n", [5, 20])
    def test_invariants_hold_over_many_episodes(self, n):
        env = CVRPNeuralEnv(n_customers=n)
        for seed in range(25):
            st, tour, rewards, loads, done = rollout_random_legal(
                env, jax.random.PRNGKey(1000 + seed)
            )
            assert done, f"episode {seed} did not terminate within max_steps"

            # capacity never exceeded
            assert min(loads) >= -1e-6, f"capacity exceeded (load {min(loads)})"

            # every customer served exactly once
            customers = [a for a in tour if a != 0]
            assert sorted(customers) == list(range(1, n + 1)), "coverage violated"

            # gamma = 1 return equals the negative tour length
            coords = np.array(st.coords)
            truth = -sum(
                np.linalg.norm(coords[tour[i + 1]] - coords[tour[i]])
                for i in range(len(tour) - 1)
            )
            assert sum(rewards) == pytest.approx(truth, abs=1e-4)

            assert len(tour) - 1 <= env.max_steps

    def test_load_never_goes_negative_under_adversarial_play(self):
        """Always take the heaviest legal customer -- the fastest way to break capacity."""
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(77))
        for _ in range(env.max_steps):
            mask = np.array(env.action_mask(st))
            d = np.where(mask, np.array(st.demands), -1.0)
            d[0] = -1.0 if mask[1:].any() else 1e9  # depot only as a last resort
            a = int(np.argmax(d))
            _, st, _, done, _ = env.step(jax.random.PRNGKey(0), st, a)
            assert float(st.load) >= -1e-6
            if bool(done):
                break


class TestObservation:
    def test_served_node_demand_reads_zero(self):
        """Paper eq. 8-11: d_i = 0 once the node is served."""
        env = CVRPNeuralEnv(n_customers=20)
        obs, st = env.reset(jax.random.PRNGKey(30))
        assert float(obs[4, _F_DEMAND]) > 0
        obs, st, _, _, _ = env.step(jax.random.PRNGKey(0), st, 4)
        assert float(obs[4, _F_DEMAND]) == 0.0
        assert float(obs[4, _F_VISITED]) == 1.0

    def test_current_and_depot_flags(self):
        env = CVRPNeuralEnv(n_customers=20)
        obs, st = env.reset(jax.random.PRNGKey(31))
        assert float(obs[0, _F_DEPOT]) == 1.0
        assert float(obs[0, _F_CURRENT]) == 1.0
        obs, st, _, _, _ = env.step(jax.random.PRNGKey(0), st, 9)
        assert float(obs[9, _F_CURRENT]) == 1.0
        assert float(obs[0, _F_CURRENT]) == 0.0
        assert float(obs[0, _F_DEPOT]) == 1.0

    def test_load_column_is_broadcast(self):
        env = CVRPNeuralEnv(n_customers=20)
        obs, st = env.reset(jax.random.PRNGKey(32))
        obs, st, _, _, _ = env.step(jax.random.PRNGKey(0), st, 2)
        col = np.array(obs[:, _F_LOAD])
        assert np.allclose(col, float(st.load))


class TestJaxCompatibility:
    """Every vectorized trainer here composes jit + vmap + scan over the env."""

    def test_jit_reset_and_step(self):
        env = CVRPNeuralEnv(n_customers=20)
        r = jax.jit(env.reset)
        s = jax.jit(env.step)
        obs, st = r(jax.random.PRNGKey(40))
        obs, st, rew, done, _ = s(jax.random.PRNGKey(0), st, jnp.int32(6))
        assert obs.shape == env.obs_shape
        assert np.isfinite(float(rew))

    def test_vmap_reset_and_step(self):
        env = CVRPNeuralEnv(n_customers=20)
        keys = jax.random.split(jax.random.PRNGKey(41), 16)
        states = jax.jit(jax.vmap(lambda k: env.reset(k)[1]))(keys)
        assert states.coords.shape == (16, 21, 2)
        out = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))(
            keys, states, jnp.arange(1, 17, dtype=jnp.int32)
        )
        assert out[0].shape == (16, 21, N_FEATURES)
        assert out[2].shape == (16,)

    def test_tree_where_selects_whole_instances(self):
        """dqn_vectorized auto-resets with _tree_where; a mixed instance would be a
        silent disaster -- coordinates from one problem, demands from another."""
        env = CVRPNeuralEnv(n_customers=20)
        keys = jax.random.split(jax.random.PRNGKey(42), 4)
        a = jax.vmap(lambda k: env.reset(k)[1])(keys)
        b = jax.vmap(lambda k: env.reset(k)[1])(jax.random.split(jax.random.PRNGKey(43), 4))
        cond = jnp.array([True, False, True, False])[:, None, None]
        picked_coords = jnp.where(cond, a.coords, b.coords)
        assert np.array_equal(np.array(picked_coords[0]), np.array(a.coords[0]))
        assert np.array_equal(np.array(picked_coords[1]), np.array(b.coords[1]))

    def test_scan_over_steps(self):
        env = CVRPNeuralEnv(n_customers=20)
        _, st = env.reset(jax.random.PRNGKey(44))

        def body(carry, key):
            state, obs = carry
            mask = env.action_mask_from_obs(obs)
            action = jax.random.categorical(key, jnp.where(mask, 0.0, -1e9))
            obs2, s2, r, d, _ = env.step(key, state, action)
            return (s2, obs2), r

        (final, _), rewards = jax.jit(
            lambda s, o, ks: jax.lax.scan(body, (s, o), ks)
        )(st, env.get_obs(st), jax.random.split(jax.random.PRNGKey(45), env.max_steps))
        assert np.isfinite(np.array(rewards)).all()
        assert bool(final.done)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
