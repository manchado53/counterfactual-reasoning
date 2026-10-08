"""Historical matched-step pilot; budget and episode epsilon stay distinct."""
import json
import time
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .dqn import _MetricsLogger


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def learn_pilot(agent, verbose=False):
    cfg = agent.config
    variant = cfg['collector_variant']
    cce = cfg['algorithm'] == 'consequence-dqn'
    run = Path(os.environ['CCE_RUN_DIR'])
    np.random.seed(cfg['seed'])
    agent.metrics_logger = _MetricsLogger(cfg, cfg['n_episodes'], None, cfg['eval_episodes'])
    agent.metrics_logger.timer.start('total')
    print('CONFIG: ' + json.dumps(cfg), flush=True)
    print(f'Run dir: {run}', flush=True)
    print('DEVICE: ' + str(jax.devices()), flush=True)
    agent._build_collect_fn()
    agent._key, init_key = jax.random.split(agent._key)
    states = jax.vmap(lambda k: agent.env.reset(k)[1])(jax.random.split(init_key, cfg['n_envs']))
    if variant == 'fixed':
        states = agent._initial_collect_carry(states)
    n = cfg['n_envs'] * cfg['collect_steps']
    blocks = cfg['blocks']
    every = cfg['eval_every_blocks']
    agent._current_episode = 0
    started = time.monotonic()
    ghost_n = 0
    for block in range(1, blocks + 1):
        agent._key, key = jax.random.split(agent._key)
        keys = jax.random.split(key, n).reshape(cfg['n_envs'], cfg['collect_steps'], 2)
        states, outputs = agent._collect_fn(agent.params, jnp.float32(agent.epsilon), states, keys)
        jax.block_until_ready(outputs[4])
        agent._add_chunk_to_buffer(outputs)
        ended = np.asarray(outputs[7]).astype(bool)
        agent.episode_returns.extend(np.asarray(outputs[5])[ended].tolist())
        agent.episode_lengths.extend(np.asarray(outputs[6])[ended].tolist())
        episode = agent._current_episode = len(agent.episode_returns)
        if cce:
            ghost_n += int(np.asarray(outputs[8].done).sum())
            if variant == 'fixed':
                assert ghost_n == 0, 'Transition stepped an ended episode'
        prev = agent.total_steps
        agent.total_steps += n
        q_start = prev // agent.n_steps_per_update
        q_end = agent.total_steps // agent.n_steps_per_update
        freq = max(1, agent.target_update_freq // agent.n_steps_per_update)
        for update in range(q_start, q_end):
            agent._update()
            if update % freq == 0:
                agent._update_target_network()
        decay = min(1., episode / max(1, agent.epsilon_decay_episodes))
        agent.epsilon = agent.epsilon_start + (agent.epsilon_end - agent.epsilon_start) * decay
        if block % every == 0 or block == blocks:
            before_key = np.asarray(agent._key).copy()
            metrics = agent.evaluate(cfg['eval_episodes'])
            np.testing.assert_array_equal(before_key, np.asarray(agent._key))
            assert np.isfinite(metrics['win_rate'])
            metrics.update(env_steps=agent.total_steps, block=block)
            agent.metrics_logger.log_eval(episode, q_end, agent.epsilon, metrics)
            agent._write_qdiag(episode, q_end)
            if cce:
                agent._log_priority_diagnostics(episode, metrics)
            agent.save(str(run / 'last.pkl'))
            if cfg['save_step_checkpoints']:
                agent.save(str(run / f'ckpt_steps_{agent.total_steps:09d}.pkl'))
            progress = dict(block=block, blocks=blocks, env_steps=agent.total_steps,
                            episode=episode, epsilon=agent.epsilon, win_rate=metrics['win_rate'],
                            seconds=time.monotonic() - started, terminal_start_transitions=ghost_n)
            dump(run / 'progress.json', progress)
            print('PROGRESS: ' + json.dumps(progress), flush=True)
