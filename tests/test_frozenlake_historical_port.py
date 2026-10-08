"""Historical config recovery and old/new graded learner checks, not old-job replay."""
import json
import os
from pathlib import Path
import pytest


def test_recovery_does_not_fill_missing_defaults(tmp_path):
    from experiments.recover_frozenlake import recover
    runs = tmp_path / 'runs'
    (runs/'1').mkdir(parents=True)
    (runs/'1'/'metrics.log').write_text('# map_name: 8x8\n# seed: 0\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'1': {'seed': 0}, '2': {'seed': 1}}))
    result = recover(manifest, runs)
    assert result['rows'] == []
    assert len(result['blocked']) == 2
    assert 'unresolved' in result['blocked'][0]['reason']
    assert result['blocked'][1]['reason'] == 'metrics.log missing'


@pytest.mark.parametrize('arm', ['uniform', 'per', 'mul', 'only'])
@pytest.mark.parametrize('collector', ['old', 'sequential'])
def test_graded_old_new_training(tmp_path, monkeypatch, arm, collector):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')
    import jax
    import numpy as np
    from experiments.learners.frozenlake.build import build_agent, smoke_overrides
    from counterfactual_rl.agents.frozen_lake import dqn as original_logger
    from counterfactual_rl.agents.frozen_lake.dqn_vectorized import FrozenLakeDQNVectorized as OldDQN
    from counterfactual_rl.agents.frozen_lake.consequence_dqn_vectorized import FrozenLakeConsequenceDQNVectorized as OldCCE
    if collector == 'sequential':
        from counterfactual_rl.agents.frozen_lake.dqn import FrozenLakeDQN as OldDQN
        from counterfactual_rl.agents.frozen_lake.consequence_dqn import FrozenLakeConsequenceDQN as OldCCE
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root/'configs/experiment/frozenlake-graded-coarse-logged.json').read_text())['rows']
    algorithm = {'uniform': 'dqn-uniform', 'per': 'dqn'}.get(arm, 'consequence-dqn')
    def matches(c):
        if c['algorithm'] != algorithm or c['slip_prob'] != .333:
            return False
        if arm == 'only': return c['mu'] == 1 and c['priority_mixing'] == 'additive'
        if arm == 'mul': return c['priority_mixing'] == 'multiplicative'
        if arm == 'add': return c['priority_mixing'] == 'additive' and c['mu'] != 1
        return True
    cfg = next(r['config'] for r in rows if matches(r['config']))
    cfg = dict(cfg, **smoke_overrides(cfg))
    cfg.update(vectorized=collector != 'sequential', collector_variant=collector)
    monkeypatch.setattr(original_logger, '__file__', str(tmp_path/'reference.py'))
    monkeypatch.setenv('SLURM_JOB_ID', 'old')
    monkeypatch.setenv('CCE_RUN_DIR', str(tmp_path/'new'))
    traces = []
    for factory in [OldCCE if algorithm == 'consequence-dqn' else OldDQN, build_agent]:
        agent = factory(dict(cfg))
        assert agent.config == cfg
        agent.learn(verbose=False)
        state = jax.tree.leaves((agent.params, agent.target_params, agent.opt_state, agent._key))
        _, indices, weights = agent.buffer.sample(cfg['batch_size'])
        traces.append(([np.asarray(x).copy() for x in state],
                       agent.episode_returns, agent.episode_lengths,
                       np.asarray(indices), np.asarray(weights), agent.evaluate(4)))
    for a, b in zip(traces[0][0], traces[1][0]): np.testing.assert_array_equal(a,b)
    for a, b in zip(traces[0][1:5], traces[1][1:5]): np.testing.assert_array_equal(a,b)
    assert traces[0][-1] == traces[1][-1]
