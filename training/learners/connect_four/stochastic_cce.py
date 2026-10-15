"""Empirical return distributions for a stochastic depth-3 minimax foe."""
import jax
import jax.numpy as jnp
import numpy as np
from training.envs import connect_four_minimax as task


def consequence(returns,q,mask,actions):
    """Mean legal-action-weighted empirical Wasserstein-1 return distance."""
    values = np.sort(np.asarray(returns,dtype=np.float64),axis=-1)
    actions = np.asarray(actions)
    masks = np.asarray(mask)
    probs = np.asarray(jax.nn.softmax(jnp.where(mask,q.astype(jnp.float32)/.5,-jnp.inf),-1),dtype=np.float64)
    taken = values[np.arange(len(actions)),actions]
    distances = np.mean(np.abs(values-taken[:,None,:]),axis=-1)
    alternatives = masks & (np.arange(7)[None,:] != actions[:,None])
    weights = np.where(alternatives,probs,0.)
    denominator = weights.sum(-1)
    weighted = np.divide((weights*distances).sum(-1),denominator,
                         out=np.zeros(len(actions)),where=denominator>0)
    fallback = np.divide(np.where(alternatives,distances,0.).sum(-1),alternatives.sum(-1),
                         out=np.zeros(len(actions)),where=alternatives.sum(-1)>0)
    return np.where(denominator>0,weighted,fallback)


def build_rollouts(network,horizon=60):
    def rollout(params,target,state,learner,action,key):
        def legal_branch(_):
            first = task.step(state,learner,action,jax.random.fold_in(key,0))
            def live(c):
                _,_,_,done,ply = c
                return ~done & (ply<horizon)
            def tick(c):
                s,total,discount,done,ply = c
                q = network.apply(params,task.observe(s,learner))
                a = jnp.argmax(jnp.where(task.legal_mask(s),q,-jnp.inf)).astype(jnp.int32)
                result = task.step(s,learner,a,jax.random.fold_in(key,ply))
                return result.state,total+discount*result.reward,discount*jnp.float32(.99),result.done,ply+1
            last,total,discount,done,_ = jax.lax.while_loop(live,tick,
                (first.state,first.reward,jnp.float32(.99),first.done,jnp.int32(1)))
            q = network.apply(target,task.observe(last,learner))
            tail = jnp.max(jnp.where(task.legal_mask(last),q,-jnp.inf))
            return total+discount*jnp.where(done,jnp.float32(0),tail),done
        return jax.lax.cond(task.legal_mask(state)[action],legal_branch,
            lambda _:(jnp.float32(0),jnp.bool_(True)),None)
    # One (K,2) key bank per state, shared across all counterfactual actions.
    paths = jax.vmap(rollout,in_axes=(None,None,None,None,None,0))
    actions = jax.vmap(paths,in_axes=(None,None,None,None,0,None))
    return jax.jit(jax.vmap(actions,in_axes=(None,None,0,0,None,0)))
