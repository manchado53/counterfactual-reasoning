"""Reduced old/new checks, not evidence for the full historical studies."""
import hashlib
import ast
import base64
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'studies/legacy-environments-migration'


@pytest.mark.parametrize('env', ['smax', 'chess', 'cvrp', 'rpdqn'])
def test_explicit_recipes(env):
    build = importlib.import_module(f'experiments.learners.{env}.build')
    rows = json.loads((ROOT / f'configs/experiment/{env}-historical.json').read_text())['rows']
    for row in rows:
        cfg = row['config']
        assert cfg['seed'] == row['seed']
        assert cfg['collector_variant'] == row['collector']
        assert build.validate_config(cfg) == cfg
        assert build.validate_config({**cfg, **build.smoke_overrides(cfg)})
        bad = dict(cfg, normalise_priorities='rank')
        with pytest.raises(ValueError):
            build.validate_config(bad)
        bad = dict(cfg)
        del bad['cf_gamma']
        with pytest.raises(ValueError):
            build.validate_config(bad)


def test_original_sources_recoverable():
    manifest = json.loads((STUDY / 'source-manifest.json').read_text())
    for path, row in manifest['sources'].items():
        assert hashlib.sha256((ROOT / row['archive']).read_bytes()).hexdigest() == row['sha256'], path


def test_study_sources_recoverable():
    history = json.loads((STUDY / 'history-manifest.json').read_text())
    for row in history['files']:
        if 'exact_bytes' in row:
            data = base64.b64decode(json.loads((ROOT / row['exact_bytes']).read_text())['base64'])
        else:
            data = (ROOT / row['archive']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['sha256'], row['source']
    catalog = json.loads((STUDY / 'study-catalog.json').read_text())
    for row in catalog['manifests']:
        data = base64.b64decode(json.loads((ROOT / row['archive']).read_text())['base64'])
        assert hashlib.sha256(data).hexdigest() == row['sha256']
    for row in json.loads((STUDY / 'run-headers.json').read_text())['rows']:
        assert hashlib.sha256(row['header'].encode()).hexdigest() == row['header_sha256']


def test_scientific_code_unchanged():
    """Import moves and run-directory routing are the only source edits allowed."""
    manifest = json.loads((STUDY / 'source-manifest.json').read_text())

    class Normalize(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

        def visit_Assign(self, node):
            if any(ast.unparse(t) in ('self.dir', 'run_dir') for t in node.targets):
                return None
            return self.generic_visit(node)

    for dest, source in manifest['ports'].items():
        left = ast.parse((STUDY / 'sources' / source).read_text())
        right = ast.parse((ROOT / dest).read_text())
        assert ast.dump(Normalize().visit(left)) == ast.dump(Normalize().visit(right)), dest
    manifest = json.loads((STUDY / 'rpdqn-source-manifest.json').read_text())
    for dest, source in manifest['ports'].items():
        left = ast.parse((STUDY / 'rpdqn-sources' / source).read_text())
        right = ast.parse((ROOT / dest).read_text())
        assert ast.dump(Normalize().visit(left)) == ast.dump(Normalize().visit(right)), dest


@pytest.mark.parametrize('env,count', [('cvrp', 61), ('rpdqn', 32)])
def test_logged_configs_complete_and_source_linked(env, count):
    build = importlib.import_module(f'experiments.learners.{env}.build')
    recipe = json.loads((ROOT / f'configs/experiment/{env}-logged-settings.json').read_text())
    headers = {row['source']: row for row in json.loads((STUDY / 'run-headers.json').read_text())['rows']}
    assert len(recipe['rows']) == count
    for row in recipe['rows']:
        assert row['source_header_sha256'] == headers[row['source_log']]['header_sha256']
        build.validate_config(row['config'])
        assert row['config']['seed'] == row['seed']


@pytest.mark.parametrize('noise', [0.0, 0.2])
@pytest.mark.parametrize('capacity', [None, 6])
def test_cvrp_all_transitions_and_oracle(capacity, noise):
    import jax
    import jax.numpy as jnp
    from counterfactual_rl.envs import cvrp as old
    from experiments.envs import cvrp as new
    from counterfactual_rl.analysis.claim1.cvrp.oracle import compute_oracle as old_oracle
    from experiments.learners.cvrp._historical.oracle import compute_oracle as new_oracle
    kwargs = dict(node_xy=old.SMALL_XY, demand=old.SMALL_DEMAND, capacity=capacity, travel_noise=noise)
    a, b = old.CVRPEnv(**kwargs), new.CVRPEnv(**kwargs)
    assert a.P == b.P
    for field in ('action_masks', 'state_features', 'next_states', 'rewards', 'dones'):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    states = jnp.repeat(jnp.arange(a.n_states), a.n_actions)
    actions = jnp.tile(jnp.arange(a.n_actions), a.n_states)
    keys = jax.random.split(jax.random.PRNGKey(7), len(states))
    for x, y in zip(jax.tree.leaves(jax.vmap(a.step)(keys, states, actions)),
                    jax.tree.leaves(jax.vmap(b.step)(keys, states, actions))):
        np.testing.assert_array_equal(x, y)
    qa, oa, sa = old_oracle(a)
    qb, ob, sb = new_oracle(b)
    np.testing.assert_array_equal(qa, qb)
    assert oa == ob and sa == sb


@pytest.mark.parametrize('algorithm', ['dqn-uniform', 'dqn', 'consequence-dqn'])
def test_cvrp_reduced_learning_exact(algorithm, tmp_path, monkeypatch):
    import jax
    from counterfactual_rl.agents.cvrp import dqn as old
    from counterfactual_rl.agents.cvrp.consequence_dqn import CVRPConsequenceDQN
    from experiments.learners.cvrp.build import build_agent, smoke_overrides
    cfg = json.loads((ROOT / 'configs/experiment/cvrp-historical.json').read_text())['rows'][0]['config']
    cfg.update(smoke_overrides(cfg))
    cfg.update(algorithm=algorithm, instance='small', capacity=6)
    # Original logger's __file__-relative output goes into a fresh test directory.
    monkeypatch.setattr(old, '__file__', str(tmp_path / 'old' / 'dqn.py'))
    a = (CVRPConsequenceDQN if algorithm == 'consequence-dqn' else old.CVRPDQN)(cfg)
    b = build_agent(cfg)
    assert a.config == b.config == cfg
    a.learn(verbose=False)
    monkeypatch.setenv('CCE_RUN_DIR', str(tmp_path / 'new'))
    b.learn(verbose=False)
    for x, y in zip(jax.tree.leaves(a.params), jax.tree.leaves(b.params)):
        np.testing.assert_array_equal(x, y)
    assert a.evaluate(2) == b.evaluate(2)
    assert (tmp_path / 'new' / 'metrics.log').is_file()


@pytest.mark.parametrize('env', ['smax', 'chess', 'cvrp'])
def test_historical_metric_fallback_preserved(env):
    from counterfactual_rl.analysis.metrics import compute_consequence_metric as old
    new = importlib.import_module(f'experiments.learners.{env}._historical.score_metrics').compute_consequence_metric
    returns = np.array([[0., 0., 0.], [1., 1., 1.], [0., 1., 0.]])
    # Signature and every value match the pinned helper, including the old fallback.
    for aggregation in ('mean', 'max', 'weighted_mean'):
        distributions = dict(enumerate(returns))
        assert new(0, distributions, aggregation=aggregation) == old(0, distributions, aggregation=aggregation)


@pytest.mark.parametrize('env', ['smax', 'chess'])
@pytest.mark.parametrize('algorithm', ['dqn-uniform', 'dqn', 'consequence-dqn'])
def test_simulator_reduced_learning_exact(env, algorithm, tmp_path, monkeypatch):
    pytest.importorskip('jaxmarl' if env == 'smax' else 'pgx')
    import jax
    build = importlib.import_module(f'experiments.learners.{env}.build')
    old = importlib.import_module(f'counterfactual_rl.agents.{env}.dqn')
    old_cce = importlib.import_module(f'counterfactual_rl.agents.{env}.consequence_dqn')
    cfg = json.loads((ROOT / f'configs/experiment/{env}-historical.json').read_text())['rows'][0]['config']
    cfg.update(build.smoke_overrides(cfg))
    cfg['algorithm'] = algorithm
    if env == 'smax':
        # Exact source env, reduced scenario stated here rather than hidden in smoke.
        cfg.update(scenario='3m', n_episodes=1)
        from counterfactual_rl.envs.smax import create_smax_env
        old_env, _, info = create_smax_env(cfg['scenario'], cfg['seed'], cfg['obs_type'])
        cls = old_cce.ConsequenceDQN if algorithm == 'consequence-dqn' else old.DQN
    else:
        cfg.update(n_chunks=1)
        from counterfactual_rl.agents.chess.train import create_chess_env
        old_env, _, info = create_chess_env(cfg['seed'])
        cls = old_cce.ChessConsequenceDQN if algorithm == 'consequence-dqn' else old.ChessDQN
    monkeypatch.setattr(old, '__file__', str(tmp_path / 'old' / 'dqn.py'))
    monkeypatch.setattr(old_cce, '__file__', str(tmp_path / 'old' / 'consequence_dqn.py'))
    a, b = cls(old_env, info, cfg), build.build_agent(cfg)
    assert a.config == b.config == cfg
    np.random.seed(cfg['seed'])
    a.learn(verbose=False)
    monkeypatch.setenv('CCE_RUN_DIR', str(tmp_path / 'new'))
    np.random.seed(cfg['seed'])
    b.learn(verbose=False)
    for x, y in zip(jax.tree.leaves(a.params), jax.tree.leaves(b.params)):
        np.testing.assert_array_equal(x, y)
    if env == 'smax':
        from counterfactual_rl.agents.smax.utils import evaluate
        left, right = evaluate(a, n_episodes=1), b.evaluate(n_episodes=1)
    else:
        left, right = a.evaluate(1, opponent='random'), b.evaluate(1, opponent='random')
    assert left == right


@pytest.mark.parametrize('algorithm', ['dqn-uniform', 'dqn', 'consequence-dqn'])
def test_rpdqn_reduced_learning_exact(algorithm, tmp_path, monkeypatch):
    """Load old code from the retained byte-checked snapshot, never a private worktree."""
    import sys
    import types
    import jax
    manifest = json.loads((STUDY / 'rpdqn-source-manifest.json').read_text())
    for path, row in manifest['sources'].items():
        assert hashlib.sha256((ROOT / row['archive']).read_bytes()).hexdigest() == row['sha256'], path
    package = types.ModuleType('legacy_rpdqn')
    package.__path__ = [str(STUDY / 'rpdqn-sources/src/counterfactual_rl')]
    monkeypatch.setitem(sys.modules, 'legacy_rpdqn', package)
    env_module = importlib.import_module('legacy_rpdqn.envs.cvrp_neural')
    score_module = importlib.import_module('legacy_rpdqn.analysis.metrics')
    monkeypatch.setitem(sys.modules, 'counterfactual_rl.envs.cvrp_neural', env_module)
    monkeypatch.setitem(sys.modules, 'counterfactual_rl.analysis.metrics', score_module)
    old = importlib.import_module('legacy_rpdqn.agents.cvrp_neural.dqn')
    if algorithm == 'consequence-dqn':
        cls = importlib.import_module('legacy_rpdqn.agents.cvrp_neural.consequence_dqn_vectorized').CVRPNeuralConsequenceDQNVectorized
    else:
        cls = importlib.import_module('legacy_rpdqn.agents.cvrp_neural.dqn_vectorized').CVRPNeuralDQNVectorized
    from experiments.learners.rpdqn.build import build_agent, smoke_overrides
    cfg = json.loads((ROOT / 'configs/experiment/rpdqn-historical.json').read_text())['rows'][0]['config']
    cfg.update(smoke_overrides(cfg))
    # Separate reduced task, not a claim about n=20 held-out generalization.
    cfg.update(n_customers=3, capacity=20.0, max_steps=8, algorithm=algorithm)
    monkeypatch.setattr(old, '__file__', str(tmp_path / 'old' / 'dqn.py'))
    a, b = cls(cfg), build_agent(cfg)
    assert a.config == b.config == cfg
    a.learn(verbose=False)
    monkeypatch.setenv('CCE_RUN_DIR', str(tmp_path / 'new'))
    b.learn(verbose=False)
    for x, y in zip(jax.tree.leaves(a.params), jax.tree.leaves(b.params)):
        np.testing.assert_array_equal(x, y)
    assert a.episode_returns == b.episode_returns
    assert a.total_steps == b.total_steps
    assert a.evaluate(2) == b.evaluate(2)


def _budget_original(variant, monkeypatch):
    import sys
    import types
    namespace = 'frozen_budget_' + variant
    package = types.ModuleType(namespace)
    package.__path__ = [str(STUDY / 'budget-sources' / variant / 'src/counterfactual_rl')]
    monkeypatch.setitem(sys.modules, namespace, package)
    env = importlib.import_module(namespace + '.envs.routing_budget')
    base_env = importlib.import_module(namespace + '.envs.cvrp')
    import counterfactual_rl.envs
    monkeypatch.setattr(counterfactual_rl.envs, 'routing_budget', env, raising=False)
    monkeypatch.setitem(sys.modules, 'counterfactual_rl.envs.routing_budget', env)
    monkeypatch.setitem(sys.modules, 'counterfactual_rl.envs.cvrp', base_env)
    for suffix in ('budget_oracle', 'oracle'):
        module = importlib.import_module(namespace + '.analysis.claim1.cvrp.' + suffix)
        monkeypatch.setitem(sys.modules, 'counterfactual_rl.analysis.claim1.cvrp.' + suffix, module)
    score = importlib.import_module(namespace + '.analysis.metrics')
    monkeypatch.setitem(sys.modules, 'counterfactual_rl.analysis.metrics', score)
    return namespace, env


@pytest.mark.parametrize('variant', ['budget', 'option_a'])
def test_budget_sources_and_recipe_coverage(variant):
    manifest = json.loads((STUDY / f'budget-{variant}-source-manifest.json').read_text())
    for row in manifest['sources'].values():
        data = base64.b64decode(json.loads((ROOT / row['exact_bytes']).read_text())['base64'])
        assert hashlib.sha256(data).hexdigest() == row['sha256']
    from experiments.learners.cvrp_budget.build import validate_config
    recipe = json.loads((ROOT / 'configs/experiment/cvrp_budget-logged-settings.json').read_text())
    rows = [r for r in recipe['rows'] if r['config']['learner_variant'] == variant]
    assert len(rows) == {'budget': 2128, 'option_a': 496}[variant]
    for row in rows:
        validate_config(row['config'])
    cfg = dict(rows[0]['config'], imaginary_historical_flag=True)
    with pytest.raises(ValueError):
        validate_config(cfg)
    if variant == 'budget':
        cfg = dict(rows[0]['config'], allow_stranding=True)
        with pytest.raises(ValueError):
            validate_config(cfg)


@pytest.mark.parametrize('windows,stranding,reward', [
    (False, False, 'stepwise'), (True, False, 'stepwise'),
    (False, True, 'stepwise'), (True, True, 'stepwise'),
    (False, False, 'terminal'), (True, False, 'terminal'),
    (False, True, 'terminal'), (True, True, 'terminal')])
def test_budget_option_a_dynamics_exact(windows, stranding, reward, monkeypatch):
    import jax
    import jax.numpy as jnp
    ns, old = _budget_original('option_a', monkeypatch)
    new = importlib.import_module('experiments.envs.cvrp_budget.option_a.routing_budget')
    spec = old.INSTANCES['small']
    kwargs = dict(node_xy=spec['xy'], demand=spec['demand'], capacity=spec['capacity'],
        budget_mult=.8, time_windows=windows, allow_stranding=stranding, reward_shape=reward)
    a, b = old.BudgetRoutingEnv(**kwargs), new.BudgetRoutingEnv(**kwargs)
    assert a._index == b._index
    for field in ('action_masks_np', 'state_features', 'rewards_np'):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    states = jnp.repeat(jnp.arange(a.n_states), a.n_actions)
    actions = jnp.tile(jnp.arange(a.n_actions), a.n_states)
    keys = jax.random.split(jax.random.PRNGKey(5), len(states))
    for x, y in zip(jax.tree.leaves(jax.vmap(a.step)(keys, states, actions)),
                    jax.tree.leaves(jax.vmap(b.step)(keys, states, actions))):
        np.testing.assert_array_equal(x, y)
    old_oracle = importlib.import_module(ns + '.analysis.claim1.cvrp.budget_oracle')
    new_oracle = importlib.import_module('experiments.learners.cvrp_budget.option_a._historical.budget_oracle')
    for x, y in zip(old_oracle.compute_oracle(a), new_oracle.compute_oracle(b)):
        np.testing.assert_array_equal(x, y)
    assert new_oracle.optimal_served(b) == new_oracle.brute_force_served(b)


@pytest.mark.parametrize('variant', ['budget', 'option_a'])
@pytest.mark.parametrize('algorithm', ['dqn-uniform', 'dqn', 'consequence-dqn'])
def test_budget_reduced_learning_exact(variant, algorithm, tmp_path, monkeypatch):
    import jax
    ns, _ = _budget_original(variant, monkeypatch)
    old = importlib.import_module(ns + '.agents.cvrp.dqn')
    old_cce = importlib.import_module(ns + '.agents.cvrp.consequence_dqn')
    from experiments.learners.cvrp_budget.build import build_agent, smoke_overrides
    rows = json.loads((ROOT / 'configs/experiment/cvrp_budget-logged-settings.json').read_text())['rows']
    cfg = next(r['config'] for r in rows if r['config']['learner_variant'] == variant)
    cfg.update(smoke_overrides(cfg))
    cfg.update(instance='small', capacity=6, budget_mult=.8, budget_units=None, algorithm=algorithm)
    if variant == 'option_a':
        cfg.update(time_windows=True, allow_stranding=True, reward_shape='terminal')
    monkeypatch.setattr(old, '__file__', str(tmp_path / 'old' / 'dqn.py'))
    cls = old_cce.CVRPConsequenceDQN if algorithm == 'consequence-dqn' else old.CVRPDQN
    a, b = cls(cfg), build_agent(cfg)
    assert a.config == b.config == cfg
    a.learn(verbose=False)
    monkeypatch.setenv('CCE_RUN_DIR', str(tmp_path / 'new'))
    b.learn(verbose=False)
    for x, y in zip(jax.tree.leaves(a.params), jax.tree.leaves(b.params)):
        np.testing.assert_array_equal(x, y)
    assert a.episode_returns == b.episode_returns
    assert a.evaluate(2) == b.evaluate(2)
