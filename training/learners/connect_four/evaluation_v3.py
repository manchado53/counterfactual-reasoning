"""Paired checkpoint tests: fixed board/role/foe/seed, no training randomness."""
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
from training.envs import connect_four_transfer as task
from .bank_v3 import BANK_PATH, SUITES, load
from .minimax_eval import batch_scores, tied_action


class Evaluator:
    bank_path = BANK_PATH

    def __init__(self, network, expected_hash):
        self.bank, self.hash = load(expected_hash)
        cases = self.bank['cases']
        self.roles = np.array([c['role'] for c in cases],np.int32)
        self.depths = np.array([c['depth'] for c in cases],np.int64)
        self.seeds = np.array([c['seed'] for c in cases],np.uint32)
        self.suites = np.array([c['suite'] for c in cases])
        self.lengths = np.array([len(c['opening']) for c in cases])
        moves = np.full((1000,4),-1,np.int32)
        for i,c in enumerate(cases):
            moves[i,:len(c['opening'])] = c['opening']
        def start(seed, role, moves):
            s = task.ENV.init(jax.random.PRNGKey(seed))
            learner = jnp.where(role == 0,s.current_player,1-s.current_player)
            def tick(s,a):
                return jax.lax.cond(a>=0,lambda s:task.ENV.step(s,a),lambda s:s,s),None
            s,_ = jax.lax.scan(tick,s,moves)
            return s,learner
        self.states,self.players = jax.jit(jax.vmap(start))(
            jnp.asarray(self.seeds),jnp.asarray(self.roles),jnp.asarray(moves))
        self.observe = jax.jit(jax.vmap(task.observe))
        self.predict = jax.jit(network.apply)
        def advance(s,a):
            return jax.lax.cond(s.terminated | s.truncated,lambda s:s,lambda s:task.ENV.step(s,a),s)
        self.advance = jax.jit(jax.vmap(advance))
        obs = np.asarray(self.observe(self.states,self.players)).reshape(1000,6,7,2)
        assert not np.asarray(self.states.terminated).any()
        for i,c in enumerate(cases):
            b = np.asarray(c['board'])
            np.testing.assert_array_equal(obs[i,:,:,0],b==1+c['role'])
            np.testing.assert_array_equal(obs[i,:,:,1],b==2-c['role'])
        self.last_actions = None

    def evaluate(self, params):
        load(self.hash)  # Verify actual bank bytes at every checkpoint.
        states = self.states
        paths = np.full((1000,42),-1,np.int32)
        for ply in range(42):
            done = np.asarray(states.terminated | states.truncated)
            if done.all():
                break
            legal = np.asarray(states.legal_action_mask)
            turn = np.asarray(states.current_player) == np.asarray(self.players)
            q = np.asarray(self.predict(params,self.observe(states,self.players)))
            assert np.isfinite(q).all()
            actions = np.argmax(np.where(legal,q,-np.inf),axis=1).astype(np.int32)
            obs = np.asarray(states.observation,dtype=np.int8)
            boards = (obs[:,:,:,0]-obs[:,:,:,1]).reshape(1000,42).copy()
            foe_turn = ~done & ~turn
            scores = batch_scores(boards,self.depths,foe_turn)
            for i in np.flatnonzero(foe_turn):
                actions[i] = tied_action(scores[i],self.seeds[i],self.lengths[i]+ply)
            assert legal[np.flatnonzero(~done),actions[~done]].all()
            paths[~done,ply] = actions[~done]
            states = self.advance(states,jnp.asarray(actions))
        assert np.asarray(states.terminated | states.truncated).all()
        rewards = np.asarray(states.rewards)[np.arange(1000),np.asarray(self.players)]
        self.last_actions = paths.copy()
        self.last_rewards = rewards.copy()
        def stats(pick):
            r = rewards[pick]
            ids = np.flatnonzero(pick)
            unique_paths = {tuple(self.bank['cases'][i]['opening'])+tuple(paths[i][paths[i]>=0]) for i in ids}
            starts = {tuple(np.array(self.bank['cases'][i]['board']).ravel()) for i in ids}
            return dict(cases=len(r),wins=int((r==1).sum()),draws=int((r==0).sum()),losses=int((r==-1).sum()),
                win_rate=float(np.mean(r==1)),draw_rate=float(np.mean(r==0)),loss_rate=float(np.mean(r==-1)),
                score_rate=float(np.mean((r.astype(np.float64)+1)/2)),unique_start_positions=len(starts),unique_game_paths=len(unique_paths))
        return dict(eval_bank_hash=self.hash,cases=1000,independent_training_seeds=1,
            by_suite={s:stats(self.suites==s) for s in SUITES},
            by_suite_role_depth={f'{s}:{r}:{d}':stats((self.suites==s)&(self.roles==r)&(self.depths==d))
                               for s in SUITES for r in (0,1) for d in range(1,6)},
            outcome_hash=hashlib.sha256(rewards.tobytes()+paths.tobytes()).hexdigest())
