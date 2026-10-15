import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from counterfactual_rl.replay.buffers import PrioritizedReplayBuffer
from counterfactual_rl.replay.consequence_buffers import ConsequenceReplayBuffer
from counterfactual_rl.replay.rank_buffer import SelectionReplayBuffer
from training.learners.four_rooms.build import config_for, validate_config, smoke_overrides, build_agent


def test_proportional_priorities_match_existing_per_and_not_ranks():
    buffer=SelectionReplayBuffer(capacity=4,selection='proportional_per',
        normalise_priorities='none',priority_mixing='additive',mu=0,eps=.01,beta=.25)
    reference=PrioritizedReplayBuffer(capacity=4,eps=.01,beta=.25)
    batch=dict(s=np.arange(4),a=np.zeros(4),r=np.zeros(4),done=np.zeros(4))
    buffer.add_batch(batch);reference.add_batch(batch)
    td=np.array([0.,1.,2.,100.])
    buffer.update_priorities(np.arange(4),td);reference.update_priorities(np.arange(4),td)
    expected=reference.priorities[:4]/reference.priorities[:4].sum()
    np.testing.assert_allclose(buffer._compute_priorities(),expected,rtol=1e-14)
    # CCE scores must have no effect on this arm.
    buffer.update_consequence_scores(np.arange(4),[1000.,0.,500.,1.],q_update_count=1)
    np.testing.assert_allclose(buffer._compute_priorities(),expected,rtol=1e-14)
    _,indices,weights=buffer.sample(4)
    np.testing.assert_allclose(weights,1/(4*expected[indices]))
    buffer.update_priorities(np.arange(4),np.zeros(4))
    np.testing.assert_allclose(buffer._compute_priorities(),np.full(4,.25))
    assert SelectionReplayBuffer.sample is ConsequenceReplayBuffer.sample
    with pytest.raises(ValueError):
        SelectionReplayBuffer(selection='proportional_per',normalise_priorities='rank')


def test_only_replay_selection_config_changes_and_one_full_run():
    root=Path(__file__).parents[1]
    rows=json.loads((root/'experiments/four_rooms/proportional-per-2x/training.json').read_text())['rows']
    assert len(rows)==1 and rows[0]['seed']==0 and rows[0]['task']==0
    c=rows[0]['config'];validate_config(c)
    baseline=config_for('rank_per',0,2)
    changed={k for k in c if c[k]!=baseline[k]}
    assert changed=={'arm','normalise_priorities','priority_mixing','mu'}
    assert c['blocks']*c['n_envs']*c['collect_steps']==524288


def test_same_network_eval_bank_and_no_cce_scoring():
    configs=[dict(config_for(a,0,2),**smoke_overrides({})) for a in ('rank_per','proportional_per')]
    agents=[build_agent(c) for c in configs]
    rank,prop=agents
    np.testing.assert_array_equal(rank.bank,prop.bank)
    assert rank.evaluation_identity()==prop.evaluation_identity()
    for a,b in zip(jax.tree.leaves(rank.params),jax.tree.leaves(prop.params)):
        np.testing.assert_array_equal(a,b)
    keys=jax.random.split(jax.random.PRNGKey(73),128).reshape(4,32,2)
    outputs=[]
    for agent in agents:
        _,out=agent.collect_fn(agent.params,jnp.float32(1),agent.states,keys)
        outputs.append(out)
    for a,b in zip(*outputs): np.testing.assert_array_equal(a,b)
    def forbidden(*args,**kwargs): raise AssertionError('Proportional PER called CCE branches')
    prop.branch_fn=forbidden
    prop.insert(outputs[1]);prop.update()
    assert prop.updates==1 and prop.scored==0
    assert prop.buffer.normalise_priorities=='none' and prop.buffer.mu==0
    assert np.isfinite(prop.last_loss)


def test_single_control_report_uses_common_observed_horizon(tmp_path):
    from analysis.studies.four_rooms_single import report,load_seed_zero
    reference=tmp_path/'reference';control=tmp_path/'control';out=tmp_path/'out'
    for arm in ('uniform','rank_per','cce','proportional_per'):
        folder=(control if arm=='proportional_per' else reference)/arm;folder.mkdir(parents=True)
        (folder/'attempt.json').write_text(json.dumps(dict(smoke=False,status='running',resolved_config=config_for(arm,0,2))))
        steps=[0,10] if arm=='proportional_per' else [0,10,20]
        rows=[dict(env_steps=s,success_rate=s/100,successful_length=None if s==0 else 80,eval_bank_hash='shared') for s in steps]
        (folder/'eval.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report(control,reference,out)
    status=json.loads((out/'status.json').read_text())
    assert status['common_horizon']==10
    assert 'One seed only' in status['inference']
    assert len((out/'comparison.csv').read_text().splitlines())==5
    assert not (out/'probabilities.json').exists()
    p=control/'proportional_per'/'eval.jsonl'
    p.write_text(p.read_text().replace('shared','other-bank'))
    with pytest.raises(ValueError,match='evaluation banks'): load_seed_zero([reference,control])
