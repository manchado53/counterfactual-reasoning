"""
Unit tests for the JAX routing env (envs/cvrp.py) — TSP and CVRP modes.

Runnable two ways:
    pytest tests/test_cvrp_env.py
    python tests/test_cvrp_env.py       # self-contained runner, no pytest needed

Checks correctness before anything downstream (the pre-flight discipline): distances,
the legal-action mask, reward == -tour length, env.P == arrays, illegal self-loops,
terminal absorbing, jit/vmap safety, and — for CVRP — that the load limit is actually
enforced and reloading works.
"""

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.envs.cvrp import CVRPEnv, DEPOT

# Small capacitated instance: 5 customers, demand 3 each (15 total) against capacity 6
# -> at most 2 customers per load, so the tour needs 3 loads and the cliff is exercised.
CAP_XY = np.array(
    [[0.50, 0.50], [0.10, 0.20], [0.85, 0.15], [0.90, 0.70], [0.40, 0.95], [0.05, 0.65]],
    dtype=np.float32,
)
CAP_DEMAND = np.array([0, 3, 3, 3, 3, 3], dtype=np.int32)
CAP_CAPACITY = 6


def _tsp_env():
    return CVRPEnv(capacity=None)          # TSP mode: no load limit


def _cvrp_env():
    return CVRPEnv()                        # default 10-customer CVRP, capacity 10


def _small_cvrp():
    return CVRPEnv(node_xy=CAP_XY, demand=CAP_DEMAND, capacity=CAP_CAPACITY)


def _rollout(env, policy):
    """Run `policy(env, state) -> action` to termination; return (path, total_reward)."""
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    state = int(state)
    masks = np.asarray(env.action_masks)
    path = [int(env.state_current_np[state])]
    total, done = 0.0, False
    for _ in range(4 * env.n_nodes + 10):
        legal = np.where(masks[state])[0]
        assert legal.size > 0, "non-terminal state has no legal action"
        a = int(policy(env, state, legal))
        _, nxt, reward, d, _ = env.step(key, jnp.int32(state), jnp.int32(a))
        state, total = int(nxt), total + float(reward)
        path.append(int(env.state_current_np[state]))
        if bool(d):
            done = True
            break
    assert done, "rollout did not terminate"
    return path, total


# ----------------------------------------------------------------------
# TSP mode
# ----------------------------------------------------------------------

def test_tsp_construction():
    env = _tsp_env()
    assert env.n_nodes == 11 and env.n_customers == 10 and env.n_actions == 11
    assert np.allclose(env.dist, env.dist.T)
    assert np.allclose(np.diag(env.dist), 0.0)
    # reachable TSP states = N * 2^(N-1) + 2 (start + terminal)
    assert env.n_states == env.n_customers * (2 ** (env.n_customers - 1)) + 2
    assert env.start_states == [0]
    assert not env.is_capacitated


def test_tsp_mask_at_start():
    env = _tsp_env()
    m = np.asarray(env.action_masks[0])
    assert m[DEPOT] == False          # cannot sit at the depot
    assert m[1:].all()                # every customer available


def test_tsp_reward_equals_negative_length():
    env = _tsp_env()
    path, total = _rollout(env, lambda e, s, legal: legal[0])
    assert path[0] == DEPOT and path[-1] == DEPOT
    assert sorted(path[1:-1]) == list(range(1, env.n_nodes))
    assert np.isclose(total, -env.tour_length(path), atol=1e-4)


# ----------------------------------------------------------------------
# CVRP mode — the load limit is the defining feature
# ----------------------------------------------------------------------

def test_cvrp_construction():
    env = _cvrp_env()
    assert env.is_capacitated and env.capacity == 10
    assert env.demand[DEPOT] == 0
    assert env.min_loads() == 3, "default instance should need three loads"
    assert env.n_states > 0


def test_cvrp_capacity_is_enforced_by_the_mask():
    """A customer whose demand exceeds the remaining load must be masked out."""
    env = _small_cvrp()
    masks = np.asarray(env.action_masks)
    for s in range(env.n_states):
        cap = int(env.state_cap[s])
        for a in np.where(masks[s])[0]:
            if a != DEPOT:
                assert env.demand[a] <= cap, (
                    f"state {s}: action {a} legal with demand {env.demand[a]} > load {cap}"
                )


def test_cvrp_reload_at_depot():
    """Arriving at the depot restores a full load."""
    env = _small_cvrp()
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    # serve one customer, load drops
    _, s1, _, _, _ = env.step(key, jnp.int32(int(state)), jnp.int32(1))
    assert int(env.state_cap[int(s1)]) == CAP_CAPACITY - int(env.demand[1])
    # return to the depot, load is restored
    _, s2, _, done, _ = env.step(key, jnp.int32(int(s1)), jnp.int32(DEPOT))
    assert int(env.state_cap[int(s2)]) == CAP_CAPACITY
    assert bool(done) is False, "not done — customers remain"
    assert int(env.state_current_np[int(s2)]) == DEPOT


def test_cvrp_visits_every_customer_and_reloads():
    """A nearest-legal rollout serves everyone, revisiting the depot to reload."""
    env = _small_cvrp()

    def nearest(e, s, legal):
        cur = int(e.state_current_np[s])
        cust = [a for a in legal if a != DEPOT]
        pool = cust if cust else list(legal)
        return min(pool, key=lambda a: e.dist[cur, a])

    path, total = _rollout(env, nearest)
    assert path[0] == DEPOT and path[-1] == DEPOT
    served = [p for p in path if p != DEPOT]
    assert sorted(served) == list(range(1, env.n_nodes)), "every customer served once"
    # 15 demand / capacity 6 -> at least 3 loads -> at least 2 mid-tour depot reloads
    assert path.count(DEPOT) >= env.min_loads() + 1
    assert np.isclose(total, -env.tour_length(path), atol=1e-4)


def test_cvrp_terminal_only_at_depot_with_all_served():
    env = _small_cvrp()
    masks = np.asarray(env.action_masks)
    full = env._full_mask()
    terminal = np.where(~masks.any(axis=1))[0]
    assert terminal.size == 1, "expected exactly one terminal state"
    t = int(terminal[0])
    assert int(env.state_current_np[t]) == DEPOT
    assert int(env.state_mask[t]) == full


def test_infeasible_instance_rejected():
    """A customer demanding more than the whole truck can never be served — fail loudly."""
    try:
        CVRPEnv(node_xy=CAP_XY, demand=np.array([0, 99, 1, 1, 1, 1], np.int32), capacity=6)
    except ValueError as e:
        assert "infeasible" in str(e)
    else:
        raise AssertionError("expected ValueError for an infeasible instance")


# ----------------------------------------------------------------------
# Shared invariants (both modes)
# ----------------------------------------------------------------------

def test_P_matches_arrays():
    for env in (_tsp_env(), _small_cvrp()):
        ns = np.asarray(env.next_states)
        rw = np.asarray(env.rewards)
        dn = np.asarray(env.dones)
        for si in range(env.n_states):
            for a in range(env.n_actions):
                outs = env.P[si][a]
                assert len(outs) == 1
                prob, nsi, r, term = outs[0]
                assert prob == 1.0
                assert nsi == ns[si, a, 0]
                assert np.isclose(r, rw[si, a, 0], atol=1e-6)
                assert term == bool(dn[si, a, 0])


def test_illegal_action_self_loops():
    env = _cvrp_env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    assert bool(env.action_masks[int(state), DEPOT]) == False
    _, nxt, reward, done, _ = env.step(key, state, jnp.int32(DEPOT))
    assert int(nxt) == int(state)
    assert float(reward) < 0.0
    assert bool(done) == False


def test_terminal_is_absorbing():
    env = _small_cvrp()
    masks = np.asarray(env.action_masks)
    t = int(np.where(~masks.any(axis=1))[0][0])
    key = jax.random.PRNGKey(0)
    for a in range(env.n_actions):
        _, nxt, reward, done, _ = env.step(key, jnp.int32(t), jnp.int32(a))
        assert int(nxt) == t and float(reward) == 0.0 and bool(done) is True


def test_features():
    """Features must be Markov-faithful: one-hot current + visited bits + load fraction."""
    env = _small_cvrp()
    feats = np.asarray(env.state_features)
    assert feats.shape == (env.n_states, env.feature_dim)
    assert env.feature_dim == env.n_nodes + env.n_customers + 1
    for s in range(0, env.n_states, max(1, env.n_states // 50)):
        f = feats[s]
        assert f[: env.n_nodes].sum() == 1.0
        assert int(np.argmax(f[: env.n_nodes])) == int(env.state_current_np[s])
        bits = f[env.n_nodes : env.n_nodes + env.n_customers]
        assert int("".join(str(int(b)) for b in bits[::-1]), 2) == int(env.state_mask[s])
        assert np.isclose(f[-1], env.state_cap[s] / env.capacity)
    # distinct states must have distinct features (no aliasing)
    assert len(np.unique(feats, axis=0)) == env.n_states


def test_jit_and_vmap():
    env = _cvrp_env()
    key = jax.random.PRNGKey(0)
    step = jax.jit(env.step)
    _, nxt, r, d, _ = step(key, jnp.int32(0), jnp.int32(1))
    assert nxt.dtype == jnp.int32
    states = jnp.array([0, 0, 0], dtype=jnp.int32)
    actions = jnp.array([1, 2, 3], dtype=jnp.int32)
    keys = jax.random.split(key, 3)
    _, nxts, _, _, _ = jax.vmap(env.step, in_axes=(0, 0, 0))(keys, states, actions)
    assert len(set(int(x) for x in nxts)) == 3
    # observe() is vmap-safe too (the trainer needs this)
    obs = jax.vmap(env.observe)(nxts)
    assert obs.shape == (3, env.feature_dim)


TESTS = [
    test_tsp_construction,
    test_tsp_mask_at_start,
    test_tsp_reward_equals_negative_length,
    test_cvrp_construction,
    test_cvrp_capacity_is_enforced_by_the_mask,
    test_cvrp_reload_at_depot,
    test_cvrp_visits_every_customer_and_reloads,
    test_cvrp_terminal_only_at_depot_with_all_served,
    test_infeasible_instance_rejected,
    test_P_matches_arrays,
    test_illegal_action_self_loops,
    test_terminal_is_absorbing,
    test_features,
    test_jit_and_vmap,
]


if __name__ == "__main__":
    import traceback

    tsp, cvrp = _tsp_env(), _cvrp_env()
    print(f"TSP  mode: n_states={tsp.n_states}")
    print(f"CVRP mode: n_states={cvrp.n_states}, capacity={cvrp.capacity}, "
          f"total demand={int(cvrp.demand.sum())}, min loads={cvrp.min_loads()}, "
          f"feature_dim={cvrp.feature_dim}\n")
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
