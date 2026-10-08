"""Matched-step old/fixed collector pilot. Never resumes or overwrites a run."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time


def temporal_class(path, parent, cce):
    """Reuse the exact frozen collector class, not another handwritten copy."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from counterfactual_rl.agents.jax_nav.dqn_vectorized import _tree_where
    tree = ast.parse(Path(path).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    cls = next(n for n in main.body if isinstance(n, ast.ClassDef) and n.name == 'TemporalAgent')
    scope = dict(Parent=parent, cce=cce, prefix=9 if cce else 8, jax=jax, jnp=jnp,
                 np=np, _tree_where=_tree_where, time=time, Path=Path, json=json)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), scope)
    return scope['TemporalAgent']


def bank_hash(agent):
    import jax
    import numpy as np
    h = hashlib.sha256()
    for leaf in jax.tree.leaves((agent._fixed_eval_init_states, agent._fixed_eval_step_keys)):
        a = np.asarray(leaf)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', -1)))
    p.add_argument('--mode', choices=['smoke', 'train'], required=True)
    args = p.parse_args()
    from carry_pilot_suite import verify, dump, validate_smokes
    work = args.work.resolve()
    verify(work)
    if args.mode == 'train':
        validate_smokes(work)
    manifest = json.loads((work / 'manifest.json').read_text())
    row = manifest['rows'][args.task]
    assert row['task'] == args.task and args.task >= 0
    cfg = row['config'].copy()
    variant = row['variant']
    runtime = work / variant
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jaxmarl
    from counterfactual_rl.agents.jax_nav import dqn as base
    from counterfactual_rl.agents.jax_nav.dqn_vectorized import JaxNavDQNVectorized
    from counterfactual_rl.agents.jax_nav.consequence_dqn_vectorized import JaxNavConsequenceDQNVectorized
    assert Path(base.__file__).resolve().is_relative_to(runtime / 'src')
    assert Path(jaxmarl.__file__).resolve().is_relative_to(work / 'vendor')
    assert jax.default_backend() == 'gpu'
    run = Path(base.__file__).parent / 'runs' / os.environ['SLURM_JOB_ID']
    assert not run.exists(), f'Refusing reused output: {run}'
    cce = cfg['algorithm'] == 'consequence-dqn'
    Parent = JaxNavConsequenceDQNVectorized if cce else JaxNavDQNVectorized
    Agent = temporal_class(runtime / 'train_temporal_replay.py', Parent, cce)
    agent = Agent(cfg)
    assert agent.config == cfg
    np.random.seed(cfg['seed'])
    agent.metrics_logger = base._MetricsLogger(cfg, cfg['n_episodes'], None, cfg['eval_episodes'])
    agent.metrics_logger.timer.start('total')
    print('CONFIG: ' + json.dumps(cfg), flush=True)
    print('PILOT: ' + json.dumps(row), flush=True)
    print(f'Run dir: {run}', flush=True)
    print('DEVICE: ' + str(jax.devices()), flush=True)
    dump(run / 'provenance.json', dict(**row, mode=args.mode, manifest=str(work / 'manifest.json')))
    agent._build_collect_fn()
    agent._key, init_key = jax.random.split(agent._key)
    states = jax.vmap(lambda k: agent.env.reset(k)[1])(jax.random.split(init_key, cfg['n_envs']))
    if variant == 'fixed':
        states = agent._initial_collect_carry(states)
    n = cfg['n_envs'] * cfg['collect_steps']
    blocks = 8 if args.mode == 'smoke' else manifest['blocks']
    every = 4 if args.mode == 'smoke' else manifest['eval_every_blocks']
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
            if args.mode == 'train':
                agent.save(str(run / f'ckpt_steps_{agent.total_steps:09d}.pkl'))
            progress = dict(block=block, blocks=blocks, env_steps=agent.total_steps,
                            episode=episode, epsilon=agent.epsilon, win_rate=metrics['win_rate'],
                            seconds=time.monotonic() - started, terminal_start_transitions=ghost_n)
            dump(run / 'progress.json', progress)
            print('PROGRESS: ' + json.dumps(progress), flush=True)
    qrows = [json.loads(x) for x in (run / 'qdiag.jsonl').read_text().splitlines()]
    assert qrows and all(r.get('loss') is not None and np.isfinite(r['loss']) for r in qrows)
    if variant == 'fixed':
        assert max(agent.episode_lengths) > cfg['collect_steps'], 'Full lengths not retained'
        assert max(agent.episode_lengths) <= cfg['max_steps']
    if cce:
        assert np.all(agent.buffer.scored_at[:len(agent.buffer)] >= 0)
        assert np.isfinite(agent.buffer.consequence_scores[:len(agent.buffer)]).all()
    if args.mode == 'smoke':
        check = Parent(cfg)
        check.load(str(run / 'last.pkl'))
        reloaded = check.evaluate(cfg['eval_episodes'])
        for name in reloaded:
            np.testing.assert_allclose(reloaded[name], metrics[name], rtol=1e-6, atol=1e-6, equal_nan=True)
        assert bank_hash(check) == bank_hash(agent)
    record = dict(passed=True, task=args.task, arm=row['arm'], variant=variant, seed=row['seed'],
                  run=str(run), env_steps=agent.total_steps, update_attempts=q_end,
                  eval_bank_hash=bank_hash(agent), fixed_eval_n=cfg['eval_episodes'],
                  score_coverage=1. if cce else None, max_episode_length=max(agent.episode_lengths),
                  seconds=time.monotonic() - started)
    dump(run / 'complete.json', record)
    if args.mode == 'smoke':
        dump(work / f'smoke-{args.task}.json', record)
    agent.metrics_logger.timer.stop('total')
    agent.metrics_logger.close()
    print('COMPLETE: ' + json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
