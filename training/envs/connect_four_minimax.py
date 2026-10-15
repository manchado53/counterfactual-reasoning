"""Depth-3 minimax learner-turn dynamics with fresh, explicit tie keys.

64-bit bitboards speed up exact depth-3 search. Network/learning arrays remain
float32. CPU minimax_eval.py is the independent reference used in tests.
"""
import jax
import jax.numpy as jnp
import numpy as np
from . import connect_four_transfer as base

ENV, Carry, Step = base.ENV, base.Carry, base.Step
legal_mask, observe = base.legal_mask, base.observe
U = jnp.uint64
POSITIONS = np.array([c*7+5-r for r in range(6) for c in range(7)],dtype=np.uint64)
MASKS = np.array([sum(1 << (int(k)%7*7+5-int(k)//7) for k in w)
                  for w in np.asarray(base.WINDOWS)],dtype=np.uint64)
FULL = np.uint64(sum(63 << (7*c) for c in range(7)))
CENTER = np.uint64(63 << 21)
COLS = jnp.arange(7,dtype=jnp.int32)


def bits(state):
    obs = state.observation.reshape(42,2)
    masks = U(1) << POSITIONS
    return jnp.sum(jnp.where(obs[:,0],masks,U(0)),dtype=np.uint64), jnp.sum(jnp.where(obs[:,1],masks,U(0)),dtype=np.uint64)


def won(b):
    value = jnp.bool_(False)
    for shift in (1,7,6,8):
        pairs = b & (b >> U(shift))
        value |= (pairs & (pairs >> U(2*shift))) != 0
    return value


def place(mine, other, col):
    shift = U(col)*U(7)
    height = jax.lax.population_count(((mine|other) >> shift) & U(63))
    return mine | (U(1) << (shift+height)), height < 6


def heuristic(mine,other):
    own = jax.lax.population_count(mine & MASKS).astype(jnp.int32)
    opp = jax.lax.population_count(other & MASKS).astype(jnp.int32)
    weights = jnp.array([0,1,4,16,16],jnp.int32)
    score = jnp.where(opp==0,weights[own],0).sum(dtype=jnp.int32)
    score -= jnp.where(own==0,weights[opp],0).sum(dtype=jnp.int32)
    score += 3*(jax.lax.population_count(mine&CENTER).astype(jnp.int32)
                -jax.lax.population_count(other&CENTER).astype(jnp.int32))
    return score


def depth1(mine,other):
    def action(col):
        nxt,legal = place(mine,other,col)
        value = jnp.where(won(nxt),100001,jnp.where((nxt|other)==FULL,0,heuristic(nxt,other)))
        return jnp.where(legal,value,-1000000)
    return jax.vmap(action)(COLS).max()


def depth2(mine,other):
    def action(col):
        nxt,legal = place(mine,other,col)
        value = jnp.where(won(nxt),100002,jnp.where((nxt|other)==FULL,0,-depth1(other,nxt)))
        return jnp.where(legal,value,-1000000)
    return jax.vmap(action)(COLS).max()


def search_scores(state):
    with jax.enable_x64(True):
        mine,other = bits(state)
        def action(col):
            nxt,legal = place(mine,other,col)
            value = jnp.where(won(nxt),100003,jnp.where((nxt|other)==FULL,0,-depth2(other,nxt)))
            return jnp.where(legal,value,-1000000)
        return jax.vmap(action)(COLS)

def opponent(state,key):
    scores = search_scores(state)
    best = (scores==scores.max()) & legal_mask(state)
    return jax.random.categorical(key,jnp.where(best,jnp.float32(0),jnp.float32(-jnp.inf))).astype(jnp.int32)


def reset(key,role):
    init,foe_key = jax.random.split(key)
    state = ENV.init(init)
    learner = jnp.where(role==0,state.current_player,1-state.current_player)
    state = jax.lax.cond(role==1,lambda s:ENV.step(s,opponent(s,foe_key)),lambda s:s,state)
    return state,learner


def step(state,learner,action,key):
    def live(s):
        first = ENV.step(s,action)
        ended = first.terminated | first.truncated
        last = jax.lax.cond(ended,lambda s:s,lambda s:ENV.step(s,opponent(s,key)),first)
        reward = first.rewards[learner]+jnp.where(ended,jnp.float32(0),last.rewards[learner])
        return Step(last,reward,last.terminated|last.truncated,jnp.where(ended,1,2).astype(jnp.int32))
    return jax.lax.cond(state.terminated|state.truncated,
        lambda s:Step(s,jnp.float32(0),jnp.bool_(True),jnp.int32(0)),live,state)


def initial_carry(keys):
    n = keys.shape[0]
    ids = jnp.arange(n,dtype=jnp.int32)
    role = ids%2
    state,learner = jax.vmap(reset)(keys,role)
    return Carry(state,learner,role,jnp.zeros(n,bool),jnp.zeros(n,jnp.float32),
        jnp.zeros(n,jnp.int32),role,ids,jnp.int32(n))


def build_collect(network):
    def collect(params,epsilon,carry,keys):
        def tick(c,ks):
            split = jax.vmap(lambda k:jax.random.split(k,4))(ks)
            ids = c.next_id+jnp.cumsum(c.pending.astype(jnp.int32),dtype=jnp.int32)-1
            roles = ids%2
            fresh,players = jax.vmap(reset)(split[:,0],roles)
            def choose(new,old):
                shape = (len(c.pending),)+(1,)*(old.ndim-1)
                return jnp.where(c.pending.reshape(shape),new,old)
            state = jax.tree.map(choose,fresh,c.state)
            learner = jnp.where(c.pending,players,c.learner)
            role = jnp.where(c.pending,roles,c.role)
            ep_ids = jnp.where(c.pending,ids,c.episode_id)
            returns = jnp.where(c.pending,0.,c.returns)
            lengths = jnp.where(c.pending,0,c.length)
            board_lengths = jnp.where(c.pending,role,c.board_length)
            obs = jax.vmap(observe)(state,learner)
            mask = jax.vmap(legal_mask)(state)
            q = network.apply(params,obs)
            greedy = jnp.argmax(jnp.where(mask,q,-jnp.inf),-1)
            random = jax.vmap(lambda k,m:jax.random.categorical(k,jnp.where(m,jnp.float32(0),jnp.float32(-jnp.inf))))(split[:,1],mask)
            explore = jax.vmap(lambda k:jax.random.uniform(k,dtype=jnp.float32))(split[:,2])<epsilon
            actions = jnp.where(explore,random,greedy).astype(jnp.int32)
            result = jax.vmap(step)(state,learner,actions,split[:,3])
            returns += result.reward
            lengths += 1
            board_lengths += result.board_moves
            next_c = Carry(result.state,learner,role,result.done,returns,lengths,board_lengths,
                ep_ids,c.next_id+c.pending.sum(dtype=jnp.int32))
            row = dict(s=obs,a=actions,r=result.reward,s_next=jax.vmap(observe)(result.state,learner),
                done=result.done,mask=mask,next_mask=jax.vmap(legal_mask)(result.state),
                state=state,learner=learner,role=role,episode_id=ep_ids,
                episode_return=returns,episode_length=lengths,board_length=board_lengths,
                board_moves=result.board_moves+jnp.where(c.pending,role,0),started=c.pending)
            return next_c,row
        return jax.lax.scan(tick,carry,keys)
    return jax.jit(collect)
