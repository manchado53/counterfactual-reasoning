import numpy as np
import pytest
import jax
import jax.numpy as jnp


def test_raw_terminal_and_timeout_are_distinct():
    from experiments.envs.frozenlake import FrozenLakeEnv
    env = FrozenLakeEnv(desc=['SFG'], is_slippery=False)
    key = jax.random.PRNGKey(7)
    obs, state = env.reset(key)
    assert int(obs) == int(state) == 0
    assert env.n_actions == 4 and env.n_states == 3
    np.testing.assert_array_equal(env.legal_actions(state), [True]*4)
    obs, state, reward, done, info = env.step(key, state, 2)
    assert (int(obs), float(reward), bool(done)) == (1, 0., False)
    assert not info['terminated'] and not info['truncated']
    cut = env.step_with_limit(key, 0, 2, elapsed_steps=0, max_steps=1)
    assert cut[4]['truncated'] and not cut[4]['terminated']
    assert cut[4]['bootstrap'] and int(cut[0]) == 1
    goal = env.step_with_limit(key, state, 2, elapsed_steps=0, max_steps=1)
    assert goal[4]['terminated'] and not goal[4]['truncated']
    assert not goal[4]['bootstrap'] and float(goal[2]) == 1.
    again = env.step(key, goal[1], 2)
    assert int(again[0]) == 2 and float(again[2]) == 0. and again[3]


@pytest.mark.parametrize('slip', [0., .2, 2/3])
def test_branch_keys_replay_without_touching_live_state(slip):
    from experiments.envs.frozenlake import FrozenLakeEnv
    env = FrozenLakeEnv(map_name='8x8', slip_prob=slip)
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)
    before = np.asarray(key).copy()
    expected = env.step(key, state, 2)
    branches = jax.random.split(jax.random.PRNGKey(3), 12000)
    values = np.asarray(jax.jit(jax.vmap(lambda k: env.step(k, state, 2)[0]))(branches))
    for tile, probability in [(0, slip/2), (1, 1-slip), (8, slip/2)]:
        assert abs(np.mean(values == tile) - probability) < .02
    actual = env.step(key, state, 2)
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(a, b)
    assert int(state) == int(obs) == 0
    np.testing.assert_array_equal(key, before)


def test_fixed_build_and_eval_have_separate_rng():
    from experiments.learners.frozenlake.build import build_agent, smoke_config, validate_config
    config = smoke_config('dqn-uniform')
    validate_config(config)
    agent = build_agent(config)
    assert agent.config == config
    before = np.asarray(agent._key).copy()
    assert agent.evaluate(4) == agent.evaluate(4)
    np.testing.assert_array_equal(agent._key, before)
    with pytest.raises(ValueError):
        validate_config({**config, 'invented_algorithm': True})
    with pytest.raises(ValueError):
        validate_config({**config, 'resume': 'last.pkl'})


@pytest.mark.parametrize('algorithm,mixing', [
    ('dqn-uniform', {}), ('dqn', {}), ('consequence-dqn', {}),
    pytest.param('consequence-dqn', {'priority_mixing': 'multiplicative'}, id='cce_mul'),
    pytest.param('consequence-dqn', {'mu': 1.0}, id='cce_only'),
])
def test_tiny_step_budget_wraps_replay_and_repeats(tmp_path, monkeypatch, algorithm, mixing):
    import json
    from experiments.learners.frozenlake.build import build_agent, smoke_config
    cfg = {**smoke_config(algorithm), **mixing}
    runs = []
    for i in range(2):
        folder = tmp_path / str(i)
        monkeypatch.setenv('CCE_RUN_DIR', str(folder))
        agent = build_agent(cfg)
        before = jax.tree.map(np.asarray, agent.params)
        assert agent.learn(verbose=False) is agent
        assert agent.total_steps == 32 and agent.q_update_count == 8
        assert len(agent.buffer) == 16
        assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before),
                                                           jax.tree.leaves(agent.params)))
        records = [json.loads(line) for line in (folder/'eval.jsonl').read_text().splitlines()]
        assert [r['env_steps'] for r in records] == [16, 32]
        assert records[0]['eval_bank_hash'] == agent.evaluation_identity()['eval_bank_hash']
        assert (folder/'last.pkl').is_file()
        if algorithm == 'consequence-dqn':
            assert agent._scored_arrivals == 32
            assert records[-1]['scored_frac'] == 1.
        runs.append(agent)
    for a, b in zip(jax.tree.leaves((runs[0].params, runs[0].target_params, runs[0].opt_state,
                                    runs[0].collector_state, runs[0]._key)),
                    jax.tree.leaves((runs[1].params, runs[1].target_params, runs[1].opt_state,
                                    runs[1].collector_state, runs[1]._key))):
        np.testing.assert_array_equal(a, b)


def test_old_collector_keeps_its_eval_rng_behavior():
    from experiments.learners.frozenlake.build import build_agent
    from experiments.learners.frozenlake.config import DEFAULT_CONFIG
    cfg = {**DEFAULT_CONFIG, 'vectorized': True, 'collector_variant': 'old',
           'training_budget_kind': 'episodes', 'algorithm': 'dqn-uniform',
           'map_name': '4x4', 'is_slippery': False}
    agent = build_agent(cfg)
    before = np.asarray(agent._key).copy()
    agent.evaluate(4)
    assert not np.array_equal(before, agent._key)


@pytest.mark.parametrize('variant', ['old', 'fixed'])
def test_smoke_overrides_keep_task_and_collector(variant):
    from experiments.learners.frozenlake.build import smoke_overrides
    cfg = {'collector_variant': variant, 'map_name': '8x8', 'is_slippery': False, 'slip_prob': .2}
    overrides = smoke_overrides(cfg)
    for key in cfg:
        assert key not in overrides
    assert overrides['cf_horizon'] == overrides['cf_n_rollouts'] == 4
    if variant == 'old':
        assert overrides['n_episodes'] == 6 and 'blocks' not in overrides
    else:
        assert overrides['blocks'] == 4


def test_hole_ends_without_reward_or_auto_reset():
    from experiments.envs.frozenlake import FrozenLakeEnv
    env = FrozenLakeEnv(desc=['SHG'], is_slippery=False)
    result = env.step(jax.random.PRNGKey(1), 0, 2)
    assert int(result[0]) == int(result[1]) == 1
    assert float(result[2]) == 0. and result[3]
    assert result[4]['terminated'] and not result[4]['bootstrap']


def test_weighted_return_distance_excludes_taken_action():
    from experiments.learners.frozenlake.transfer import score_table
    returns = np.array([[[0., 0.], [1., 1.], [2., 2.], [3., 3.]]])
    np.testing.assert_allclose(score_table(returns, np.full((1, 4), .25)),
                               [[2., 4/3, 4/3, 2.]], rtol=1e-12, atol=1e-12)


def test_all_saved_recipes_are_resolved():
    import json
    from pathlib import Path
    from experiments.learners.frozenlake.build import validate_config
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'configs/experiment').glob('frozenlake*.json'))
    assert len(paths) == 3
    for path in paths:
        for row in json.loads(path.read_text())['rows']:
            assert validate_config(row['config']) == row['config']


def test_extra_arm_probes_change_only_mix_fields():
    import json
    from pathlib import Path
    from experiments.learners.frozenlake.build import validate_config
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root/'configs/experiment/frozenlake-smoke.json').read_text())['rows'][2]['config']
    extension = json.loads((root/'studies/frozenlake-migration/extra-arms-smoke.json').read_text())
    assert [row['arm'] for row in extension['rows']] == ['cce_mul', 'cce_only']
    for row, expected in zip(extension['rows'],
                             [{'priority_mixing': 'multiplicative'}, {'mu': 1.0}]):
        config = row['config']
        assert {k: v for k, v in config.items() if v != original[k]} == expected
        assert validate_config(config) == config
