"""Exact source comparison for real masked updates and all fixed-transfer arms."""
import json
import numpy as np
import pytest
import jax
from tests.test_doorkey_transfer_migration import source_import, same, ARMS, restore_source_namespace
from tests.test_doorkey_transfer_migration import ROOT


def test_legacy_kernels_are_path_only():
    source = ROOT/'studies/small-transfers-migration/sources/root/src/counterfactual_rl'
    port = ROOT/'experiments/learners/connect_four/legacy'
    for name in ('config.py','dqn.py','consequence_dqn.py','policies.py','opponent.py','opponent_mcts.py'):
        text = (source/'agents/connect_four'/name).read_text()
        text = text.replace('from ..chess.array_buffer','from .array_buffer')
        text = text.replace('from ..shared.','from .shared.')
        text = text.replace('from counterfactual_rl.utils.action_selection','from .action_selection')
        text = text.replace('from counterfactual_rl.analysis.metrics','from .metrics')
        assert (port/name).read_text() == text
    assert (port/'array_buffer.py').read_bytes() == (source/'agents/chess/array_buffer.py').read_bytes()
    assert (port/'action_selection.py').read_bytes() == (source/'utils/action_selection.py').read_bytes()


def test_seed_scope_is_protocol_specific():
    from experiments.learners.connect_four.build import validate_config,resolved_config
    fixed = resolved_config('uniform',0)
    fixed['seed'] = 9
    with pytest.raises(ValueError,match='seed 0 or 1'):
        validate_config(fixed)
    rows=json.loads((ROOT/'configs/experiment/connect_four-claim2-main-legacy.json').read_text())['rows']
    assert {r['seed'] for r in rows} == set(range(10))
    for row in rows:
        validate_config(row['config'])


@pytest.mark.parametrize('opponent',['random','rule_based','mcts'])
@pytest.mark.parametrize('arm',ARMS)
def test_legacy_source_and_smoke(opponent,arm,tmp_path,monkeypatch):
    from experiments.learners.connect_four.build import build_agent,smoke_overrides
    from experiments.learners.connect_four.legacy.build import ENV_INFO
    from experiments.learners.connect_four.legacy.shared.metrics import MetricsLogger
    name = {'random':'main','rule_based':'rulebased','mcts':'mcts-v2'}[opponent]
    rows = json.loads((ROOT/f'configs/experiment/connect_four-claim2-{name}-legacy.json').read_text())['rows']
    cfg = next(r['config'] for r in rows if r['arm']==arm and r['seed']==0)
    cfg.update(smoke_overrides(cfg))
    module = 'consequence_dqn' if arm.startswith('cce') else 'dqn'
    cls = 'Connect4ConsequenceDQN' if arm.startswith('cce') else 'Connect4DQN'
    old = source_import('root','agents.connect_four.'+module)
    reference = getattr(old,cls)(ENV_INFO.copy(),cfg)
    agent = build_agent(cfg)
    same(agent.params,reference.params)
    initial = [np.array(x) for x in jax.tree.leaves(agent.params)]
    # Only redirect the untouched source's output logger; no learner patch.
    import sys
    monkeypatch.setattr(sys.modules['counterfactual_rl.agents.shared.metrics'],'MetricsLogger',MetricsLogger)
    if module == 'dqn':
        monkeypatch.setattr(old,'MetricsLogger',MetricsLogger)
    for which,obj in [('old',reference),('new',agent)]:
        out=tmp_path/which;out.mkdir()
        monkeypatch.setenv('CCE_RUN_DIR',str(out))
        np.random.seed(17)
        obj.learn(verbose=False)
    assert agent.config == cfg
    same(agent.params,reference.params)
    same(agent.target_params,reference.target_params)
    same(agent.opt_state,reference.opt_state)
    same(agent._key,reference._key)
    same(agent.buffer.td_magnitudes[:len(agent.buffer)],reference.buffer.td_magnitudes[:len(reference.buffer)])
    same(agent.episode_returns,reference.episode_returns)
    same(agent.episode_lengths,reference.episode_lengths)
    assert agent.total_steps == reference.total_steps == 32
    assert any(not np.array_equal(x,y) for x,y in zip(initial,jax.tree.leaves(agent.params)))
    if arm.startswith('cce'):
        assert agent.q_update_count == reference.q_update_count == 8
        same(agent.buffer.consequence_scores[:len(agent.buffer)],reference.buffer.consequence_scores[:len(reference.buffer)])
    before = np.array(agent._key)
    same(list(agent.evaluate(8,seed=9).values()),list(reference.evaluate(8,seed=9).values()))
    same(before,agent._key)
    records = [json.loads(line) for line in (tmp_path/'new/eval.jsonl').read_text().splitlines()]
    assert [r['env_steps'] for r in records] == [16,32]
    assert [r['eval_seed'] for r in records] == [0,1]
    assert agent.evaluation_identity()['eval_opponent'] == opponent
    initial_numpy = json.loads((tmp_path/'new/numpy_initial_state.json').read_text())
    expected_numpy = np.random.RandomState(17).get_state()
    np.testing.assert_array_equal(initial_numpy['keys'],expected_numpy[1])
    assert initial_numpy['position'] == expected_numpy[2]


@pytest.mark.parametrize('arm', ARMS)
def test_fixed_source_and_smoke(arm,tmp_path,monkeypatch):
    from experiments.learners.connect_four.build import resolved_config,smoke_overrides,build_agent
    source = source_import('connect-four-fixed','agents.connect_four.transfer')
    cfg = resolved_config(arm,0)
    cfg.update(smoke_overrides(cfg))
    agent = build_agent(cfg)
    reference = source.Trainer(arm,0,n_envs=4,capacity=256)
    same(agent.params,reference.params)
    initial = [np.array(x) for x in jax.tree.leaves(agent.params)]
    np.random.seed(0)
    for _ in range(2):
        reference.learn_block(32)
    monkeypatch.setenv('CCE_RUN_DIR',str(tmp_path))
    agent.learn(verbose=False)
    same(agent.params,reference.params)
    same(agent.target_params,reference.target_params)
    same(agent.opt_state,reference.opt_state)
    same(agent.key,reference.key)
    same(agent.carry,reference.carry)
    assert agent.diagnostics() == reference.diagnostics()
    assert agent.evaluate() == reference.evaluate()
    assert agent.config == cfg and agent.steps == 256 and agent.updates == 16
    assert any(not np.array_equal(x,y) for x,y in zip(initial,jax.tree.leaves(agent.params)))
    before = np.array(agent.key)
    numpy_before = np.random.get_state()
    agent.evaluate()
    same(before,agent.key)
    same(numpy_before[1],np.random.get_state()[1])
    if agent.cce:
        same(agent.buffer.consequence_scores,reference.buffer.consequence_scores)
        assert np.all(agent.buffer.scored_at[:len(agent.buffer)] >= 0)
    from experiments.learners.connect_four.transfer import Trainer
    restored = Trainer.load(tmp_path/'last.pkl')
    same(agent.params,restored.params)
    assert restored.evaluate() == agent.evaluate()
    assert len((tmp_path/'eval.jsonl').read_text().splitlines()) == 2
