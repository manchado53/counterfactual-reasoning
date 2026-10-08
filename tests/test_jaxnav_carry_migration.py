"""The step-budget pilot retains its frozen collection and learning loop."""
import ast
import hashlib
import json
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'studies/jaxnav-historical/sources/carry'
RECIPE = ROOT / 'configs/experiment/jaxnav-carry-pilot-20260930.json'


def test_carry_sources_and_twenty_rows():
    from experiments.recipes import load_recipe
    from experiments.registry import get_environment
    hashes = json.loads((SOURCE.parent.parent/'carry-source-hashes.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == expected
    for row in json.loads(RECIPE.read_text())['rows']:
        assert load_recipe(RECIPE, row['task']) == row
        assert get_environment('jaxnav').evaluation_axis(row['config']) == 'env_steps'


def test_fixed_hold_class_is_frozen_source():
    old = ast.parse((SOURCE/'fixed/train_temporal_replay.py').read_text())
    new = ast.parse((ROOT/'experiments/learners/jaxnav/temporal_fixed.py').read_text())
    def cls(tree):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'TemporalAgent')
    assert ast.dump(cls(old)) == ast.dump(cls(new))


@pytest.mark.parametrize('name', ['dqn_vectorized.py', 'consequence_dqn_vectorized.py'])
def test_fixed_parent_is_frozen_carry_source(name):
    old = (SOURCE/'fixed/src/counterfactual_rl/agents/jax_nav'/name).read_text()
    old = old.replace('from ..shared.early_stopping', 'from experiments.common.early_stopping')
    new = (ROOT/'experiments/learners/jaxnav'/name).read_text()
    assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))


def test_pilot_science_loop_is_unchanged():
    # The outer launch/log paths move, and manifest budgets become explicit
    # config fields. Collection, updates, target cadence, epsilon and eval do not.
    old = ast.parse((SOURCE/'scripts/train_carry_pilot.py').read_text())
    main = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                and n.target.id == 'block')
    new = ast.parse((ROOT/'experiments/learners/jaxnav/carry_pilot.py').read_text())
    fn = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'learn_pilot')
    moved = next(n for n in fn.body if isinstance(n, ast.For))
    # Only checkpoint output suppression in explicit --smoke differs.
    class SaveCondition(ast.NodeTransformer):
        def visit_Compare(self, node):
            if ast.unparse(node) == "args.mode == 'train'":
                return ast.parse("cfg['save_step_checkpoints']", mode='eval').body
            return self.generic_visit(node)
    assert ast.dump(SaveCondition().visit(loop)) == ast.dump(moved)


@pytest.mark.parametrize('task', [0, 1, 4, 5, 8, 9, 12, 13, 16, 17])
def test_carry_cpu_training_and_reference_collector(tmp_path, monkeypatch, task):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    import jax
    import jax.numpy as jnp
    import numpy as np
    import time
    from experiments.recipes import load_recipe
    from experiments.registry import get_environment
    from experiments.learners.jaxnav.carry_pilot import learn_pilot
    row = load_recipe(RECIPE, task)
    adapter = get_environment('jaxnav')
    cfg = dict(row['config'], **adapter.smoke_overrides(row['config']))
    if cfg['collector_variant'] == 'fixed':
        from experiments.learners.jaxnav.dqn_vectorized import JaxNavDQNVectorized as DQN, _tree_where
        from experiments.learners.jaxnav.consequence_dqn_vectorized import JaxNavConsequenceDQNVectorized as CCE
    else:
        from experiments.learners.jaxnav.dqn_vectorized_legacy import JaxNavDQNVectorized as DQN, _tree_where
        from experiments.learners.jaxnav.consequence_dqn_vectorized_legacy import JaxNavConsequenceDQNVectorized as CCE
    cce = cfg['algorithm'] == 'consequence-dqn'
    code = ast.parse((SOURCE/cfg['collector_variant']/'train_temporal_replay.py').read_text())
    node = next(n for n in ast.walk(code) if isinstance(n, ast.ClassDef) and n.name == 'TemporalAgent')
    scope = dict(Parent=CCE if cce else DQN, cce=cce, prefix=9 if cce else 8, jax=jax, jnp=jnp,
                 np=np, _tree_where=_tree_where, time=time, Path=Path, json=json)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'frozen-carry', 'exec'), scope)
    Reference = scope['TemporalAgent']
    Reference.learn = learn_pilot  # loop equivalence proved above, only launch wiring moved
    traces = []
    for name, factory in [('old', Reference), ('new', adapter.build_agent)]:
        out = tmp_path/name
        out.mkdir()
        monkeypatch.setenv('CCE_RUN_DIR', str(out))
        agent = factory(dict(cfg))
        agent.learn(verbose=False)
        leaves = jax.tree.leaves((agent.params, agent.target_params, agent.opt_state, agent._key))
        rows = [json.loads(line) for line in (out/'eval.jsonl').read_text().splitlines()]
        assert [r['env_steps'] for r in rows] == [16, 32]
        assert agent.total_steps == 32
        traces.append(([np.asarray(x).copy() for x in leaves], rows))
    for a, b in zip(traces[0][0], traces[1][0]):
        np.testing.assert_array_equal(a, b)
    assert traces[0][1] == traces[1][1]
