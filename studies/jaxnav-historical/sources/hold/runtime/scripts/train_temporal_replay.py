"""Hold exploration only; preserve parent PER/CCE training and evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from temporal_replay_suite import verify, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'train'], required=True)
    parser.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', -1)))
    args = parser.parse_args()
    work = args.work.resolve()
    verify(work)
    spec = json.loads((work / 'manifest.json').read_text())['rows'][args.task]
    assert args.task == spec['task'] and 0 <= args.task < 40
    cfg = spec['config'].copy()
    if args.mode == 'smoke':
        assert spec['seed'] == 0
        cfg.update(n_episodes=1000, n_checkpoints=1, n_buffer_snapshots=0)
    print('CONFIG: ' + json.dumps(cfg), flush=True)
    print('SOURCE: ' + json.dumps({'baseline': spec['parent_log'], 'hashes': str(work / 'hashes.json')}), flush=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from counterfactual_rl.agents.jax_nav import dqn as base_module
    from counterfactual_rl.agents.jax_nav.dqn_vectorized import JaxNavDQNVectorized, _tree_where
    from counterfactual_rl.agents.jax_nav.consequence_dqn_vectorized import JaxNavConsequenceDQNVectorized
    import jaxmarl

    assert Path(base_module.__file__).resolve().is_relative_to(work / 'runtime/src')
    assert Path(jaxmarl.__file__).resolve().is_relative_to(work / 'runtime/vendor')
    run = Path(base_module.__file__).parent / 'runs' / os.environ['SLURM_JOB_ID']
    assert not run.exists(), f'Refusing nonempty/reused run path: {run}'
    print('DEVICE: ' + str(jax.devices()), flush=True)
    assert jax.default_backend() == 'gpu', 'GPU job must not silently use CPU'
    cce = cfg['algorithm'] == 'consequence-dqn'
    Parent = JaxNavConsequenceDQNVectorized if cce else JaxNavDQNVectorized
    prefix = 9 if cce else 8

    class TemporalAgent(Parent):
        def _build_collect_fn(self):
            env, net, seed, hold = self.env, self.network, self.config['seed'], self.config['explore_hold_steps']

            def collect(params, epsilon, init_state, step_keys):
                def step(carry, key):
                    state, obs, ended, total, length = carry
                    kr, _, _, ke = jax.random.split(key, 4)
                    new_obs, new_state = env.reset(kr)
                    state = _tree_where(ended, new_state, state)
                    obs = jnp.where(ended, new_obs, obs)
                    total = jnp.where(ended, jnp.float32(0), total)
                    length = jnp.where(ended, jnp.int32(0), length)
                    # Exact random-key rule of uniform array 278653. The draw is
                    # fixed inside a block; epsilon can change at chunk boundaries.
                    bits = jax.lax.bitcast_convert_type(state.goal[0], jnp.uint32)
                    k = jax.random.PRNGKey(seed)
                    k = jax.random.fold_in(jax.random.fold_in(k, bits[0]), bits[1])
                    ka, kx = jax.random.split(jax.random.fold_in(k, state.step // hold))
                    draw = jax.random.uniform(kx)
                    random_action = jax.random.randint(ka, (), 0, self.n_actions)
                    explore = draw < epsilon
                    action = jnp.where(explore, random_action, jnp.argmax(net.apply(params, obs)))
                    ob, ns, reward, done, _ = env.step(ke, state, action)
                    total = total + reward
                    length = length + jnp.int32(1)
                    standard = (obs, action, reward, ob, done,
                                jnp.where(done, total, 0.), jnp.where(done, length, 0), done)
                    if cce:
                        standard += (state,)  # Parent CCE consumes slot 8 unchanged.
                    return (ns, ob, done, total, length), standard + (explore, state.goal[0], state.step, draw)

                carry = (init_state, env.get_obs(init_state), jnp.bool_(False), jnp.float32(0), jnp.int32(0))
                (state, _, _, _, _), outputs = jax.lax.scan(step, carry, step_keys)
                return state, outputs
            self._collect_fn = jax.jit(jax.vmap(collect, in_axes=(None, None, 0, 0)))

        def _add_chunk_to_buffer(self, outputs):
            start = time.monotonic()
            super()._add_chunk_to_buffer(outputs[:prefix])
            explore = np.asarray(outputs[prefix])
            self._explore_n = getattr(self, '_explore_n', 0) + int(explore.sum())
            self._collect_n = getattr(self, '_collect_n', 0) + explore.size
            self._chunk_n = getattr(self, '_chunk_n', 0) + 1
            row = dict(episode=self._current_episode, chunk=self._chunk_n, epsilon=float(self.epsilon),
                       hold=self.config['explore_hold_steps'], explore_fraction=float(explore.mean()),
                       cumulative_explore_n=self._explore_n, cumulative_transition_n=self._collect_n,
                       scoring_insert_seconds=time.monotonic() - start)
            if cce:
                n = len(self.buffer)
                row['scored_frac'] = float(np.mean(self.buffer.scored_at[:n] >= 0))
                assert row['scored_frac'] == 1., 'CCE lost score-on-insert coverage'
            with (Path(self.metrics_logger.dir) / 'exploration.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')

    def same_tree(a, b):
        assert jax.tree.structure(a) == jax.tree.structure(b)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=1e-6)

    def bank_hash(agent):
        h = hashlib.sha256()
        for leaf in jax.tree.leaves((agent._fixed_eval_init_states, agent._fixed_eval_step_keys)):
            a = np.asarray(leaf)
            h.update(str((a.shape, a.dtype)).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    if args.mode == 'smoke':
        check = TemporalAgent(cfg)
        assert check.config == cfg
        check._build_collect_fn()
        states = jax.vmap(lambda k: check.env.reset(k)[1])(jax.random.split(jax.random.PRNGKey(471), 8))
        pieces = []
        for idx, epsilon in enumerate([1., 1., .8, .2]):
            keys = jax.random.split(jax.random.PRNGKey(100 + idx), 8 * 32).reshape(8, 32, 2)
            states, out = check._collect_fn(check.params, jnp.float32(epsilon), states, keys)
            jax.block_until_ready(out[4])
            out = jax.device_get(out)
            assert np.array_equal(out[prefix], out[prefix + 3] < np.float32(epsilon))
            assert ((out[1] >= 0) & (out[1] < 15)).all()
            pieces.append(out)
        actions = np.concatenate([p[1] for p in pieces[:2]], axis=1)
        goals = np.concatenate([p[prefix + 1] for p in pieces[:2]], axis=1)
        steps = np.concatenate([p[prefix + 2] for p in pieces[:2]], axis=1)
        same = (goals[:, 1:] == goals[:, :-1]).all(-1) & (steps[:, 1:] // spec['hold'] == steps[:, :-1] // spec['hold'])
        assert np.all(actions[:, 1:][same] == actions[:, :-1][same])
        # Both forced timeout/reset and normal chunk continuation retain parent behavior.
        original = Parent(cfg)
        original._build_collect_fn()
        states = states.replace(step=jnp.full_like(states.step, 199))
        keys = jax.random.split(jax.random.PRNGKey(531), 8 * 32).reshape(8, 32, 2)
        _, a = check._collect_fn(check.params, jnp.float32(0), states, keys)
        _, b = original._collect_fn(original.params, jnp.float32(0), states, keys)
        same_tree(a[:prefix], b)
        if cce:
            # Check stored simulator states really produce the corresponding observations.
            obs_from_state = jax.vmap(jax.vmap(check.env.get_obs))(a[8])
            np.testing.assert_allclose(np.asarray(obs_from_state), np.asarray(a[0]), rtol=1e-6, atol=1e-6)
        print('PASS collector: hold persistence, changing epsilon, resets, parent greedy equivalence, CCE state contract', flush=True)
        del check, original

    agent = TemporalAgent(cfg)
    assert agent.config == cfg, 'Resolved config differs from frozen recipe'
    started = time.monotonic()
    agent.learn(verbose=True)
    elapsed = time.monotonic() - started
    assert Path(agent.metrics_logger.dir) == run
    rows = [json.loads(x) for x in (run / 'eval.jsonl').read_text().splitlines()]
    assert rows[-1]['episode'] >= cfg['n_episodes']
    assert all(b['episode'] > a['episode'] for a, b in zip(rows, rows[1:]))
    assert all(np.isfinite(r['win_rate']) for r in rows)
    qrows = [json.loads(x) for x in (run / 'qdiag.jsonl').read_text().splitlines()]
    assert qrows and all(r.get('loss') is not None and np.isfinite(r['loss']) for r in qrows)
    if cce:
        assert np.all(agent.buffer.scored_at[:len(agent.buffer)] >= 0)
        assert np.isfinite(agent.buffer.consequence_scores[:len(agent.buffer)]).all()
    if args.mode == 'smoke':
        before_key = np.asarray(agent._key).copy()
        before = agent.evaluate(1000)
        check = Parent(cfg)
        check.load(str(run / 'last.pkl'))
        after = check.evaluate(1000)
        for name in before:
            np.testing.assert_allclose(before[name], after[name], rtol=1e-6, atol=1e-6, equal_nan=True)
        assert np.array_equal(before_key, np.asarray(agent._key))
        assert bank_hash(agent) == bank_hash(check)
        dump(work / f'smoke-{args.task}.json', dict(passed=True, task=args.task, arm=spec['arm'], hold=spec['hold'],
             episodes=rows[-1]['episode'], seconds=elapsed, fixed_eval_n=1000, eval_bank_hash=bank_hash(agent),
             run=str(run), device=str(jax.devices()), score_coverage=1. if cce else None,
             rough_250k_hours=elapsed / rows[-1]['episode'] * 250000 / 3600))
        print('SMOKE PASS: training, finite loss, coverage, 1000-map evaluation and checkpoint round trip', flush=True)
    else:
        dump(run / 'complete.json', dict(task=args.task, arm=spec['arm'], hold=spec['hold'], seed=spec['seed'],
             episodes=rows[-1]['episode'], seconds=elapsed, eval_bank_hash=bank_hash(agent)))


if __name__ == '__main__':
    main()
