"""CPU source comparisons; source files are independent, pinned, unmodified bytes."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT/'studies/small-transfers-migration/sources'
ARMS = ('uniform','per','cce_mul','cce_add','cce_only')


@pytest.fixture(autouse=True)
def restore_source_namespace():
    saved = {k:v for k,v in sys.modules.items() if k == 'counterfactual_rl' or k.startswith('counterfactual_rl.')}
    yield
    for key in list(sys.modules):
        if key == 'counterfactual_rl' or key.startswith('counterfactual_rl.'):
            del sys.modules[key]
    sys.modules.update(saved)


def source_import(label, module):
    # Give the untouched source its original package name. New code does not
    # import that namespace, so no source rewrite or private worktree is needed.
    for name in list(sys.modules):
        if name == 'counterfactual_rl' or name.startswith('counterfactual_rl.'):
            del sys.modules[name]
    package = types.ModuleType('counterfactual_rl')
    package.__path__ = [str(SOURCES/label/'src/counterfactual_rl')]
    sys.modules['counterfactual_rl'] = package
    return importlib.import_module('counterfactual_rl.'+module)


def same(a,b):
    left,right = jax.tree.leaves(a),jax.tree.leaves(b)
    assert len(left) == len(right)
    for x,y in zip(left,right):
        np.testing.assert_array_equal(x,y)


def test_source_hashes():
    manifest = json.loads((ROOT/'studies/small-transfers-migration/source-manifest.json').read_text())
    for label, record in manifest.items():
        for name, digest in record['files'].items():
            assert hashlib.sha256((SOURCES/label/name).read_bytes()).hexdigest() == digest, (label,name)
    extra = json.loads((ROOT/'studies/small-transfers-migration/extra-source-manifest.json').read_text())
    for name,record in extra['files'].items():
        assert hashlib.sha256((SOURCES/'root'/name).read_bytes()).hexdigest() == record['sha256']


def test_resolved_controls_and_source_pins():
    from experiments.learners.doorkey.source_recipe import RECIPE
    assert RECIPE == json.loads((SOURCES/'doorkey-fixed/src/counterfactual_rl/agents/doorkey/source_recipe.json').read_text())
    from experiments.recipes import load_recipe
    from experiments.registry import get_environment
    for pattern in ('doorkey-*.json','connect_four-*.json'):
        for path in (ROOT/'configs/experiment').glob(pattern):
            manifest = json.loads(path.read_text())
            adapter = get_environment(manifest['environment'])
            for row in manifest['rows']:
                assert load_recipe(path,row['task']) == row
                config = row['config']
                smoke = dict(config,**adapter.smoke_overrides(config))
                adapter.validate_config(smoke)
                if config['protocol'] == 'fixed-jaxnav-transfer':
                    assert config['blocks'] == 2000 and config['eval_every_blocks'] == 50
                    assert config['training_budget_kind'] == 'env_steps' and config['early_stop_win_rate'] is None
                    assert config['eval_episodes'] == smoke['eval_episodes'] == 1000
                    assert config['hidden_dim'] == smoke['hidden_dim'] == 128
                    for key in ('gamma','alpha','batch_size','n_steps_per_update','target_update_freq','fixed_eval_seed','PER_parameters'):
                        assert smoke[key] == config[key]
                    bad = dict(config, gamma=.5)
                    with pytest.raises(ValueError):
                        adapter.validate_config(bad)
    # Pure path changes only in transition and fixed ConnectFour learner kernels.
    pairs = [('doorkey-fixed','src/counterfactual_rl/envs/doorkey.py','experiments/envs/doorkey.py'),
             ('andon','src/counterfactual_rl/envs/doorkey.py','experiments/envs/doorkey_legacy.py'),
             ('connect-four-fixed','src/counterfactual_rl/envs/connect_four_transfer.py','experiments/envs/connect_four_transfer.py'),
             ('connect-four-fixed','src/counterfactual_rl/agents/connect_four/transfer.py','experiments/learners/connect_four/transfer.py'),
             ('doorkey-fixed','src/counterfactual_rl/agents/doorkey/adapter.py','experiments/learners/doorkey/adapter.py')]
    for label,original,new in pairs:
        text = (SOURCES/label/original).read_text()
        text = text.replace('from counterfactual_rl.envs','from experiments.envs')
        text = text.replace('from counterfactual_rl.agents.shared.','from ._source_shared.')
        assert (ROOT/new).read_text() == text


@pytest.mark.parametrize('arm', ARMS)
def test_fixed_source_and_smoke(arm,tmp_path,monkeypatch):
    from experiments.learners.doorkey.build import resolved_config,smoke_overrides,build_agent
    old = source_import('doorkey-fixed','agents.doorkey.train')
    cfg = resolved_config(arm,0)
    cfg.update(smoke_overrides(cfg))
    agent = build_agent(cfg)
    reference = old.Trainer(arm,0)
    same(agent.params,reference.params)
    same(agent.opt_state,reference.opt_state)
    # The original collector and migrated collector see identical fresh keys.
    keys = jax.random.split(jax.random.PRNGKey(71),4*64).reshape(4,64,2)
    carry = old.initial_carry(4)
    from experiments.learners.doorkey.adapter import initial_carry
    a = agent.task.collect(agent.params,jnp.float32(1.),initial_carry(4),keys)
    b = reference.task.collect(reference.params,jnp.float32(1.),carry,keys)
    same(a,b)
    assert np.asarray(a[1]['truncated']).any()
    assert not np.asarray(a[1]['done'])[np.asarray(a[1]['truncated'])].any()
    # Compare real TD updates, target handling, returns, and CCE scores.
    states = jnp.asarray(a[1]['s']).reshape(-1)[:128]
    actions = jnp.asarray(a[1]['a']).reshape(-1)[:128]
    rewards = jnp.asarray(a[1]['r']).reshape(-1)[:128]
    next_s = jnp.asarray(a[1]['next_s']).reshape(-1)[:128]
    done = jnp.asarray(a[1]['done']).reshape(-1)[:128]
    batch = (states,actions,rewards,next_s,done,jnp.linspace(.1,1.,128))
    same(agent.update_fn(agent.params,agent.target,agent.opt_state,*batch),
         reference.update_fn(reference.params,reference.target,reference.opt_state,*batch))
    ra = agent.task.branch_returns(agent.params,agent.target,states[:8])
    rb = reference.task.branch_returns(reference.params,reference.target,states[:8])
    same(ra,rb)
    assert agent.evaluate() == reference.evaluate()
    before = [np.array(x) for x in jax.tree.leaves(agent.params)]
    monkeypatch.setenv('CCE_RUN_DIR',str(tmp_path))
    agent.learn(verbose=False)
    assert agent.config == cfg and agent.steps == 256 and agent.updates == 16
    assert any(not np.array_equal(x,y) for x,y in zip(before,jax.tree.leaves(agent.params)))
    before_rng = agent.rng_state()
    agent.evaluate()
    assert agent.rng_state() == before_rng
    if agent.cce:
        assert np.all(agent.buffer.scored_at[:len(agent.buffer)] >= 0)
    restored = build_agent(cfg)
    restored.load(tmp_path/'last.pkl')
    same(agent.params,restored.params)
    assert restored.evaluate() == agent.evaluate()
    assert len((tmp_path/'eval.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('layout', ['6x6','6x6_lava','12x12','16x16','12x12_lava','16x16_lava'])
@pytest.mark.parametrize('arm', ARMS)
def test_legacy_source_and_smoke(layout,arm,tmp_path,monkeypatch):
    from experiments.learners.doorkey.build import build_agent,smoke_overrides
    if layout in ('12x12','16x16') and arm not in ('uniform','per'):
        pytest.skip('Source headroom study defines uniform/PER only')
    recipe = ('doorkey-large-lava-cli-legacy.json' if layout in ('12x12_lava','16x16_lava') else
              'doorkey-headroom-legacy.json' if layout in ('12x12','16x16') else
              'doorkey-lava-claim2-legacy.json' if layout.endswith('lava') else 'doorkey-claim2-legacy.json')
    rows = json.loads((ROOT/'configs/experiment'/recipe).read_text())['rows']
    cfg = next(r['config'] for r in rows if r['arm']==arm and r['seed']==0 and r['config']['layout_name']==layout)
    cfg.update(smoke_overrides(cfg))
    module = 'consequence_dqn' if arm.startswith('cce') else 'dqn'
    name = 'DoorKeyConsequenceDQN' if arm.startswith('cce') else 'DoorKeyDQN'
    if cfg['vectorized']:
        module += '_vectorized'
        name += 'Vectorized'
    old = source_import('andon','agents.doorkey.'+module)
    reference = getattr(old,name)(cfg)
    agent = build_agent(cfg)
    same(agent.params,reference.params)
    same(agent.env.next_states,reference.env.next_states)
    same(agent.env.rewards,reference.env.rewards)
    same(agent.env.dones,reference.env.dones)
    if cfg['vectorized']:
        agent._build_collect_fn()
        reference._build_collect_fn()
    # Full learn() paths use identical seeds; compare params/replay and eval RNG.
    old_out = tmp_path/'source'; old_out.mkdir()
    new_out = tmp_path/'port'; new_out.mkdir()
    monkeypatch.setenv('CCE_RUN_DIR',str(new_out))
    agent.learn(verbose=False)
    # Source logger has no output override. Redirect its logger factory only.
    from experiments.learners.doorkey.legacy.dqn import _MetricsLogger
    monkeypatch.setenv('CCE_RUN_DIR',str(old_out))
    monkeypatch.setattr(sys.modules['counterfactual_rl.agents.doorkey.dqn'],'_MetricsLogger',_MetricsLogger)
    if hasattr(old,'_MetricsLogger'):
        monkeypatch.setattr(old,'_MetricsLogger',_MetricsLogger)
    reference.learn(verbose=False)
    same(agent.params,reference.params)
    same(agent.target_params,reference.target_params)
    same(agent._key,reference._key)
    assert agent.config == cfg
    assert agent.episode_returns == reference.episode_returns
    assert agent.episode_lengths == reference.episode_lengths
    assert agent.total_steps == reference.total_steps > 0
    assert agent.evaluate(2) == reference.evaluate(2)
