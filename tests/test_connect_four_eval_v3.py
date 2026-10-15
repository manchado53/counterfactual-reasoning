import copy
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from training.learners.connect_four import bank_v3 as bank
from training.learners.connect_four.minimax_eval import root_scores, tied_action, won, heuristic
from training.learners.connect_four.protocol_v3 import config_for, validate, ARMS
from training.learners.connect_four.evaluation_v3 import Evaluator
from training.learners.connect_four.transfer_v2 import Agent
from training.envs import connect_four_transfer as task


def test_bank_frozen_balanced_paired_and_reproducible(tmp_path):
    data,h = bank.load()
    assert bank.generate() == data
    assert bank.verify(data) == h
    assert len({tuple(np.array(c['board']).ravel()) for c in data['cases']}) == 51
    p=tmp_path/'bank.json'
    p.write_bytes(bank.canonical(data))
    assert bank.load(h,p)[1]==h
    data['cases'][0]['seed'] += 1
    p.write_bytes(bank.canonical(data))
    with pytest.raises(ValueError,match='hash'):
        bank.load(h,p)


def test_bank_rejects_bad_pairing():
    data,_=bank.load()
    data['cases'][500]['opening_id']=1
    with pytest.raises(AssertionError):
        bank.verify(data)


def board_from(moves):
    board=np.zeros((6,7),np.int8)
    for ply,col in enumerate(moves):
        bank.play(board,col,1+ply%2)
    # Root perspective: next mover is +1.
    return np.where(board==0,0,np.where(board==1+len(moves)%2,1,-1)).astype(np.int8).ravel()


@pytest.mark.parametrize('depth',range(1,6))
def test_minimax_wins_blocks_and_legal(depth):
    board=board_from([0,1,0,1,0,2])
    before=board.copy()
    scores=root_scores(board,depth)
    assert np.argmax(scores)==0
    np.testing.assert_array_equal(board,before)
    if depth>=2:
        board=board_from([1,0,1,0,2,0])
        assert np.argmax(root_scores(board,depth))==0
    board=board_from([0]*6)
    scores=root_scores(board,depth)
    assert scores[0]==-1000000
    assert tied_action(scores,5,6)!=0


def test_seeded_ties_repeat_without_global_rng():
    scores=np.zeros(7,dtype=np.int64)
    before=pickle.dumps(np.random.get_state())
    choices=[tied_action(scores,seed,8) for seed in range(50)]
    assert len(set(choices))>1
    assert choices==[tied_action(scores,seed,8) for seed in range(50)]
    assert pickle.dumps(np.random.get_state())==before


def brute(board, player, depth):
    if not np.any(board[:7]==0):
        return 0
    if depth==0:
        return heuristic(board,player)
    values=[]
    for col in np.flatnonzero(board[:7]==0):
        k=np.flatnonzero(board.reshape(6,7)[:,col]==0)[-1]*7+col
        board[k]=player
        values.append(100000+depth if won(board,player) else -brute(board,-player,depth-1))
        board[k]=0
    return max(values)


def test_alpha_beta_exact_root_ties_match_exhaustive_search():
    for moves in ([],[3,2,3,4],[1,0,1,0,2,0]):
        b=board_from(moves)
        for depth in (1,2,3):
            expected=np.full(7,-1000000)
            for col in np.flatnonzero(b[:7]==0):
                k=np.flatnonzero(b.reshape(6,7)[:,col]==0)[-1]*7+col
                b[k]=1
                expected[col]=100000+depth if won(b,1) else -brute(b,-1,depth-1)
                b[k]=0
            np.testing.assert_array_equal(root_scores(b,depth),expected)


def test_matched_recipe_and_training_unchanged():
    from training.learners.connect_four.protocol_v2 import config_for as v2
    from training.learners.connect_four.build import validate_config, smoke_overrides
    rows=json.loads(Path('experiments/connect_four/fixed-eval-v3/training.json').read_text())['rows']
    assert len(rows)==15
    assert {(r['arm'],r['seed']) for r in rows}=={(a,s) for a in ARMS for s in range(5)}
    for row in rows:
        c=row['config']
        validate_config(c)
        old=v2(c['arm'],c['seed'])
        assert {k:v for k,v in c.items() if k not in ('protocol','eval_bank_hash')} == {k:v for k,v in old.items() if k not in ('protocol','eval_bank_hash')}
        smoke=dict(c,**smoke_overrides(c))
        validate(smoke)
    replay={'arm','score_on_insert','decay_enabled','normalise_priorities'}
    for seed in range(5):
        configs=[{k:v for k,v in config_for(a,seed).items() if k not in replay} for a in ARMS]
        assert configs[0]==configs[1]==configs[2]


class FixedNetwork:
    def apply(self,params,obs):
        return jnp.broadcast_to(params,obs.shape[:-1]+(7,))


def test_full_bank_identity_checkpoint_and_rng_isolation(tmp_path):
    from training.learners.connect_four.build import build_agent
    agent=build_agent(config_for('uniform',0))
    assert isinstance(agent,Agent)
    before=agent.rng_identity()
    first=agent.evaluate()
    assert agent.rng_identity()==before
    paths=agent.evaluator.last_actions.copy()
    agent.steps=500000  # The same saved policy at a later training point.
    agent.key=jax.random.PRNGKey(981)  # Even unrelated training RNG changes cannot affect eval.
    later_rng=agent.rng_identity()
    second=agent.evaluate()
    assert second==first
    np.testing.assert_array_equal(paths,agent.evaluator.last_actions)
    assert agent.rng_identity()==later_rng
    p=tmp_path/'agent.pkl'
    agent.save(p)
    with p.open('rb') as f:
        saved=pickle.load(f)
    assert agent.evaluator.evaluate(saved['params'])==first
    assert agent.rng_identity()==later_rng
    assert before!=later_rng
    assert first['by_suite']['empty']['cases']==500
    assert first['by_suite']['opening']['cases']==500
    assert first['by_suite']['empty']['unique_start_positions']==1
    assert first['by_suite']['opening']['unique_start_positions']==50
    for result in first['by_suite_role_depth'].values():
        assert result['cases']==50
        assert result['wins']+result['draws']+result['losses']==50
        assert result['score_rate']==pytest.approx((result['wins']+.5*result['draws'])/50)
    # Restore bank identity from disk on every call, not merely a cached hash label.
    assert saved['eval_bank_hash']==bank.load()[1]


def test_same_actions_same_foe_even_when_q_values_change():
    evaluator=Evaluator(FixedNetwork(),bank.load()[1])
    a=jnp.arange(7,dtype=jnp.float32)
    first=evaluator.evaluate(a)
    paths=evaluator.last_actions.copy()
    second=evaluator.evaluate(a*3+50)
    assert first==second
    np.testing.assert_array_equal(paths,evaluator.last_actions)


def test_search_starts_in_fresh_processes():
    import subprocess
    import sys
    code = "from training.learners.connect_four.minimax_eval import root_scores; import numpy as np; print(root_scores(np.zeros(42,np.int8),5).tolist())"
    first = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=60)
    second = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=60)
    assert first.returncode == second.returncode == 0, first.stderr+second.stderr
    assert first.stdout == second.stdout
