"""Bounded independent-process reference probe; emits JSON to stdout.

Reference mode imports the sealed runtime through PYTHONPATH. Migrated mode calls
the public build/learn path. Both use the same explicitly reduced recipe. No jobs.
The parent must set a fresh CCE_RUN_DIR and a 180-second process timeout.
"""
import argparse
import hashlib
import json
import os

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np


def digest(tree):
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        value = np.asarray(leaf)
        h.update(str((value.shape, value.dtype)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['reference', 'migrated'])
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    with open(args.recipe) as stream:
        cfg = json.load(stream)['rows'][args.task]['config']
    if args.mode == 'migrated':
        from experiments.learners.frozenlake.build import build_agent
        agent = build_agent(cfg)
        agent.learn(verbose=False)
        carry = agent.collector_state
    else:
        from counterfactual_rl.agents.frozen_lake.transfer import TransferCCE, TransferDQN
        # This loop follows the sealed train_lake_transfer.py, including update
        # phase, target sync at update zero, and episode-clock epsilon decay.
        np.random.seed(cfg['seed'])
        agent = (TransferCCE if cfg['algorithm'] == 'consequence-dqn' else TransferDQN)(cfg)
        agent._build_collect_fn()
        agent._key, key = jax.random.split(agent._key)
        states = jax.vmap(lambda k: agent.env.reset(k)[1])(jax.random.split(key, cfg['n_envs']))
        carry = agent._initial_collect_carry(states)
        n = cfg['n_envs'] * cfg['collect_steps']
        for block in range(1, cfg['blocks'] + 1):
            agent._key, key = jax.random.split(agent._key)
            keys = jax.random.split(key, n).reshape(cfg['n_envs'], cfg['collect_steps'], 2)
            carry, out = agent._collect_fn(agent.params, jnp.float32(agent.epsilon), carry, keys)
            jax.block_until_ready(out[4])
            agent._add_chunk_to_buffer(out)
            ended = np.asarray(out[7], dtype=bool)
            agent.episode_returns.extend(np.asarray(out[5])[ended].tolist())
            agent.episode_lengths.extend(np.asarray(out[6])[ended].tolist())
            episode = agent._current_episode = len(agent.episode_returns)
            start_update = agent.total_steps // agent.n_steps_per_update
            agent.total_steps += n
            end_update = agent.total_steps // agent.n_steps_per_update
            freq = max(1, agent.target_update_freq // agent.n_steps_per_update)
            for update in range(start_update, end_update):
                agent._update()
                if update % freq == 0:
                    agent._update_target_network()
            decay = min(1., episode / max(1, agent.epsilon_decay_episodes))
            agent.epsilon = agent.epsilon_start + (agent.epsilon_end-agent.epsilon_start) * decay
            if block % cfg['eval_every_blocks'] == 0 or block == cfg['blocks']:
                agent.evaluate(cfg['eval_episodes'])
    replay = agent.buffer
    result = dict(params=digest(agent.params), target=digest(agent.target_params),
                  optimizer=digest(agent.opt_state), carry=digest(carry), key=digest(agent._key),
                  replay=digest(replay.buffer[:len(replay)]),
                  epsilon=agent.epsilon, steps=agent.total_steps, updates=agent.q_update_count,
                  returns=agent.episode_returns, lengths=agent.episode_lengths,
                  eval=agent.evaluate(cfg['eval_episodes']))
    for name in ('priorities', 'td_priorities', 'consequence_scores', 'scored_at'):
        if hasattr(replay, name):
            result[name] = digest(getattr(replay, name)[:len(replay)])
    if hasattr(agent, '_score_key'):
        result['score_key'] = digest(agent._score_key)
        result['scored_arrivals'] = agent._scored_arrivals
    # Probe actual sampled indices and weights after training, without assuming
    # that equal priority arrays imply equal host RNG consumption.
    _, indices, weights = replay.sample(cfg['batch_size'])
    result['sample_indices'] = np.asarray(indices).tolist()
    result['sample_weights'] = np.asarray(weights).tolist()
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
