"""Fail closed if the training device's search differs from the CPU test foe."""
import jax
import jax.numpy as jnp
import numpy as np
from training.envs import connect_four_minimax as task
from .minimax_eval import root_scores


def verify_search(evaluator):
    ids=np.array([0]+list(range(500,550)))
    states=jax.tree.map(lambda x:x[ids],evaluator.states)
    actual=np.asarray(jax.jit(jax.vmap(task.search_scores))(states))
    observations=np.asarray(states.observation,dtype=np.int8)
    for obs,scores in zip(observations,actual):
        board=(obs[:,:,0]-obs[:,:,1]).reshape(42).copy()
        np.testing.assert_array_equal(scores,root_scores(board,3))
    return dict(depth=3,boards_checked=len(ids),exact_cpu_match=True,backend=jax.default_backend())
