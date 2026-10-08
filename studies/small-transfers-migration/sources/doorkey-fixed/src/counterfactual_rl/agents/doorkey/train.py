"""Executable fixed JaxNav-recipe transfer; fresh runs only, no job submission."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .adapter import DoorKeyAdapter, initial_carry, ddqn_targets, weighted_scores
from ..shared.buffers import PrioritizedReplayBuffer
from ..shared.consequence_buffers import ConsequenceReplayBuffer

ARMS = ('uniform', 'per', 'cce_mul', 'cce_add', 'cce_only')
HERE = Path(__file__).resolve().parent
RECIPE = json.loads((HERE / 'source_recipe.json').read_text())


def config_for(arm, seed):
    row = next(r for r in RECIPE['rows'] if r['arm'] == arm and r['seed'] == seed)
    expected = dict(gamma=.99, alpha=.0003, hidden_dim=128, n_layers=3,
        batch_size=128, buffer_capacity=100000, target_update_freq=2000,
        n_steps_per_update=16, double_dqn=True, n_envs=256, collect_steps=32,
        epsilon_start=1., epsilon_end=.05, epsilon_decay_episodes=62500,
        fixed_eval_seed=20260923,
        PER_parameters=dict(eps=.01, beta=.25, maximum_priority=1.))
    for name,value in expected.items():
        assert row['config'][name] == value, ('source recipe drift', name)
    cfg = dict(expected, seed=seed, arm=arm, cce_enabled=arm.startswith('cce'),
               optimizer='adam', gradient_clip=10., loss='importance-weighted-squared-td',
               target_copy_phase='after attempts 0,125,250,...',
               importance_weights='1/(N*p), no batch rescale')
    if cfg['cce_enabled']:
        cce = dict(consequence_metric='wasserstein', consequence_aggregation='weighted_mean',
            cf_horizon=60, cf_n_rollouts=1, cf_gamma=.99, cf_rollout_temperature=0,
            cf_bootstrap=True, score_on_insert=True, cf_score_chunk=1280,
            normalise_priorities='rank', cf_action_prob_temperature=.5,
            mu=1. if arm == 'cce_only' else .5, mu_c=1., mu_delta=1.,
            priority_mixing='multiplicative' if arm == 'cce_mul' else 'additive')
        for name,value in cce.items():
            assert row['config'][name] == value, ('source CCE drift', name)
        cfg.update(cce, refresh_enabled=False, drift_probe_enabled=False)
    cfg.update(environment='custom-full-state-doorkey', layout_name='6x6', slip_prob=0.,
               random_start='fixed', max_episode_steps=50, eval_max_steps=50,
               explore_hold_steps=1, eval_episodes=1000, observation='one-hot-state-id',
               timeout_bootstrap=True, blocks=2000, eval_every_blocks=50)
    # All historical/inert keys remain in source_recipe.json, not runtime config.
    return cfg


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes():
    root = HERE.parents[3]
    files = list(HERE.glob('*.py')) + [HERE / 'source_recipe.json',
        root / 'src/counterfactual_rl/envs/doorkey.py']
    files += [HERE.parent / 'shared' / name for name in
              ('buffers.py', 'consequence_buffers.py', 'draw_log.py')]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def append(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, allow_nan=False) + '\n')


class Trainer:
    def __init__(self, arm, seed):
        self.arm, self.seed = arm, seed
        self.config = config_for(arm, seed)
        self.task = DoorKeyAdapter()
        self.key, init = jax.random.split(jax.random.PRNGKey(seed))
        self.params = self.task.net.init(init, jnp.int32(0))
        self.target = jax.tree.map(jnp.copy, self.params)
        self.optimizer = optax.chain(optax.clip_by_global_norm(10.), optax.adam(.0003))
        self.opt_state = self.optimizer.init(self.params)
        self.cce = arm.startswith('cce')
        if self.cce:
            self.buffer = ConsequenceReplayBuffer(capacity=100000, eps=.01, beta=.25,
                max_priority=1., mu=1. if arm == 'cce_only' else .5,
                priority_mixing='multiplicative' if arm == 'cce_mul' else 'additive',
                mu_c=1., mu_delta=1., normalise_priorities='rank')
        else:
            self.buffer = PrioritizedReplayBuffer(capacity=100000, eps=.01, beta=.25,
                                                  max_priority=1., uniform=arm == 'uniform')
        self.carry = initial_carry(256)
        self.steps = self.episodes = self.updates = self.attempts = 0
        self.epsilon = 1.
        self.max_episode_length = 0
        self.eval_keys = jax.random.split(jax.random.PRNGKey(20260923), 1000)
        self.eval_bank_hash = hashlib.sha256(np.asarray(self.eval_keys).tobytes() +
            b'doorkey-6x6-fixed-east-slip0-cap50-greedy-lowest-index').hexdigest()
        self.update_fn = jax.jit(self._update)
        self.q_fn = jax.jit(self.task.net.apply)

    def _update(self, params, target, opt_state, s, a, r, ns, done, weights):
        next_online = self.task.net.apply(params, ns)
        next_target = self.task.net.apply(target, ns)
        targets = jax.lax.stop_gradient(ddqn_targets(next_online, next_target, r, done))
        def loss_fn(p):
            q = self.task.net.apply(p, s)
            qa = jnp.take_along_axis(q, a[:, None], axis=1)[:, 0]
            td = targets - qa
            loss = jnp.mean(weights * td ** 2)
            stats = jnp.stack([loss, jnp.mean(qa), jnp.max(jnp.abs(q)),
                               jnp.mean(targets), jnp.max(jnp.abs(targets)),
                               jnp.mean(jnp.abs(td)), jnp.max(jnp.abs(td))])
            return loss, (td, stats)
        (_, (td, stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, td, stats

    def insert(self, rows):
        data = {key: np.asarray(rows[name]).reshape(-1) for key, name in
                [('s','s'), ('a','a'), ('r','r'), ("s'",'next_s'), ('done','done')]}
        assert not np.asarray(rows['terminal_start']).any(), 'terminal-start replay row'
        assert len(data['s']) == 8192
        if not self.cce:
            self.buffer.add_batch(data)
            return
        slots = self.buffer.add_batch(data, jax_states=data['s'])
        for lo in range(0, len(slots), 1280):
            selected = slots[lo:lo + 1280]
            states = jnp.asarray(data['s'][lo:lo + 1280])
            # Preserve source scoring's training-key consumption, even at slip=0.
            self.key, subkey = jax.random.split(self.key)
            keys = jax.random.split(subkey, len(selected) * 7).reshape(len(selected), 7, 1, 2)
            del keys  # Deterministic branches need no actuator noise.
            returns = np.asarray(self.task.branch_returns(self.params, self.target, states))
            probs = np.asarray(jax.nn.softmax(self.q_fn(self.params, states) / .5, axis=-1))
            scores = weighted_scores(returns, probs, data['a'][lo:lo + 1280])
            assert np.isfinite(scores).all()
            self.buffer.update_consequence_scores(selected, scores, q_update_count=self.updates)
        assert np.all(self.buffer.scored_at[:len(self.buffer)] >= 0), 'incomplete score coverage'

    def train_block(self):
        self.key, key = jax.random.split(self.key)
        keys = jax.random.split(key, 8192).reshape(256, 32, 2)
        start = time.monotonic()
        self.carry, rows = self.task.collect(self.params, jnp.float32(self.epsilon), self.carry, keys)
        rows = jax.device_get(rows)
        collected = time.monotonic()
        self.insert(rows)
        scored = time.monotonic()
        ended = rows['ended'].astype(bool)
        lengths = rows['episode_length'][ended]
        self.episodes += int(ended.sum())
        if len(lengths):
            self.max_episode_length = max(self.max_episode_length, int(lengths.max()))
        stats = []
        for attempt in range(self.steps // 16, (self.steps + 8192) // 16):
            self.attempts += 1
            if self.buffer.can_sample(128):
                batch, indices, weights = self.buffer.sample(128)
                arrays = [jnp.asarray(np.array([t[k] for t in batch])) for k in ('s','a','r',"s'",'done')]
                self.params, self.opt_state, td, diagnostic = self.update_fn(
                    self.params, self.target, self.opt_state, *arrays, jnp.asarray(weights, dtype=jnp.float32))
                td, diagnostic = jax.device_get((td, diagnostic))
                assert np.isfinite(td).all() and np.isfinite(diagnostic).all(), 'nonfinite learning'
                self.buffer.update_priorities(indices, td)
                self.updates += 1
                stats.append(diagnostic)
            # Source phase: copy after zero-index update attempts 0,125,250,...
            if attempt % 125 == 0:
                self.target = jax.tree.map(jnp.copy, self.params)
        self.steps += 8192
        self.epsilon = 1. + (.05 - 1.) * min(1., self.episodes / 62500)
        values = np.mean(stats, axis=0)
        result = dict(zip(('loss','q_mean','q_abs_max_mean','target_mean','target_abs_max_mean',
                          'td_abs_mean','td_abs_max_mean'), map(float, values)))
        result.update(env_steps=self.steps, updates=self.updates, update_attempts=self.attempts,
            completed_episodes=self.episodes, epsilon=self.epsilon,
            explore_fraction=float(rows['explore'].mean()),
            completed_return_mean=float(rows['episode_return'][ended].mean()) if ended.any() else None,
            completed_length_mean=float(lengths.mean()) if len(lengths) else None,
            max_episode_length=self.max_episode_length, terminal_start_transitions=0,
            collect_seconds=collected-start, insert_score_seconds=scored-collected,
            update_seconds=time.monotonic()-scored)
        if self.cce:
            result.update(scored_frac=float(np.mean(self.buffer.scored_at[:len(self.buffer)] >= 0)))
        return result

    def rng_state(self):
        return (np.asarray(self.key).tobytes(), pickle.dumps(np.random.get_state()))

    def evaluate(self):
        before = self.rng_state()
        returns, lengths = jax.device_get(self.task.eval_paths(self.params, self.eval_keys))
        assert self.rng_state() == before, 'evaluation changed training RNG'
        return dict(win_rate=float((returns > 0).mean()), avg_return=float(returns.mean()),
                    avg_length=float(lengths.mean()), eval_episodes=1000, unique_tasks=1,
                    eval_bank_hash=self.eval_bank_hash)

    def diagnostics(self):
        before = self.rng_state()
        if self.cce:
            row = self.buffer.priority_diagnostics(now=self.updates)
        else:
            n = len(self.buffer)
            p = np.ones(n) / n if self.arm == 'uniform' else self.buffer.priorities[:n] / self.buffer.priorities[:n].sum()
            row = dict(n=n, ess_frac=float(1. / (n * np.sum(p ** 2))), p_max=float(p.max()))
        row.update(self.buffer.reward_diagnostics())
        assert self.rng_state() == before, 'diagnostics changed training RNG'
        return row

    def save(self, path):
        # Evaluation checkpoint, not a replay-resume promise. Fresh-run CLI only.
        payload = dict(config=self.config, params=jax.device_get(self.params), target=jax.device_get(self.target),
            opt_state=jax.device_get(self.opt_state), carry=jax.device_get(self.carry), key=np.asarray(self.key),
            numpy_rng=np.random.get_state(), steps=self.steps, episodes=self.episodes, updates=self.updates,
            attempts=self.attempts, epsilon=self.epsilon, max_episode_length=self.max_episode_length,
            source_hashes=source_hashes(), eval_bank_hash=self.eval_bank_hash, resumable=False)
        with path.open('wb') as f:
            pickle.dump(payload, f)

    def load(self, path):
        with path.open('rb') as f:
            payload = pickle.load(f)
        assert payload['config'] == self.config
        assert payload['source_hashes'] == source_hashes()
        assert payload['eval_bank_hash'] == self.eval_bank_hash
        for name in ('params','target','opt_state','carry','key','steps','episodes','updates',
                     'attempts','epsilon','max_episode_length'):
            setattr(self, name, jax.tree.map(jnp.asarray, payload[name]) if name in
                    ('params','target','opt_state','carry','key') else payload[name])
        return payload


def validate_gate(path):
    gate = json.loads(path.read_text())
    assert gate['passed'] is True
    reports = [json.loads(Path(p).read_text()) for p in gate['smoke_reports']]
    assert len(reports) == 10
    expected = {(env, arm) for env in ('doorkey','connect_four') for arm in ARMS}
    assert {(r['environment'], r['arm']) for r in reports} == expected
    for r in reports:
        assert r['passed'] and r['mode'] == 'smoke' and r['seed'] == 0
        assert r['env_steps'] == 65536 and r['updates'] > 0
        if r['environment'] == 'doorkey':
            assert r['source_hashes'] == source_hashes(), 'stale DoorKey smoke'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('smoke','train'), required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=(0,1), required=True)
    p.add_argument('--output', '--out', dest='out', type=Path, required=True)
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--gate-file', type=Path)
    p.add_argument('--allow-cpu', action='store_true', help='CPU correctness only; never a GPU smoke gate')
    args = p.parse_args()
    if args.mode == 'smoke':
        assert args.seed == 0
    elif args.gate_file is not None:
        validate_gate(args.gate_file)
    # Main's frozen submission layer owns the cross-environment ten-smoke gate.
    assert not (args.require_gpu and args.allow_cpu)
    assert args.allow_cpu or jax.default_backend() == 'gpu', 'GPU required'
    np.random.seed(args.seed)
    agent = Trainer(args.arm, args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    fingerprint = digest(hashes)
    provenance = dict(environment='doorkey', arm=args.arm, seed=args.seed, mode=args.mode,
        config=agent.config, full_params=agent.config, config_hash=digest(agent.config), source_hashes=hashes, source_fingerprint=fingerprint,
        source_commit=RECIPE['source_commit'], base_commit='0231116bd4e97cfc06d01c95ee7247763ff6bb0d',
        versions={name: importlib.metadata.version(name) for name in ('jax','jaxlib','flax','optax','numpy','scipy')},
        python=sys.version, device=str(jax.devices()), backend=jax.default_backend(),
        eval_bank_hash=agent.eval_bank_hash, cwd=str(Path.cwd()), command=sys.argv,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=HERE, text=True).strip())
    dump(args.out / 'provenance.json', provenance)
    dump(args.out / 'config.json', agent.config)
    print('PROVENANCE: ' + json.dumps(provenance), flush=True)
    initial = [np.asarray(x).copy() for x in jax.tree.leaves(agent.params)]
    blocks, interval = (8,4) if args.mode == 'smoke' else (2000,50)
    started = time.monotonic()
    evals, block_costs = [], []
    for block in range(1, blocks + 1):
        block_start = time.monotonic()
        row = agent.train_block()
        row.update(block=block, arm=args.arm, seed=args.seed, seconds=time.monotonic()-started)
        append(args.out / 'qdiag.jsonl', row)
        block_costs.append(time.monotonic()-block_start)
        if block % interval == 0:
            metrics = agent.evaluate()
            evaluation = dict(metrics, block=block, env_steps=agent.steps, arm=args.arm, seed=args.seed)
            evals.append(evaluation)
            append(args.out / 'eval.jsonl', evaluation)
            append(args.out / 'replay.jsonl', dict(agent.diagnostics(), block=block, env_steps=agent.steps))
            agent.save(args.out / 'last.pkl')
            if args.mode == 'train':
                agent.save(args.out / ('checkpoint_%09d.pkl' % agent.steps))
        dump(args.out / 'progress.json', row)
        print('PROGRESS: ' + json.dumps(row), flush=True)
    assert agent.updates == agent.attempts == blocks * 512
    assert agent.steps == blocks * 8192
    assert agent.max_episode_length <= 50 and agent.max_episode_length > 32
    changed = any(not np.array_equal(a, np.asarray(b)) for a,b in zip(initial,jax.tree.leaves(agent.params)))
    assert changed, 'no parameter changes'
    check = Trainer(args.arm, args.seed)
    check.load(args.out / 'last.pkl')
    assert check.evaluate() == agent.evaluate(), 'checkpoint evaluation changed'
    for a,b in zip(jax.tree.leaves(agent.params), jax.tree.leaves(check.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert source_hashes() == hashes, 'source changed during run'
    report = dict(environment='doorkey', arm=args.arm, seed=args.seed, mode=args.mode,
        backend=jax.default_backend(),
        passed=not args.allow_cpu and jax.default_backend() == 'gpu', correctness_passed=True,
        env_steps=agent.steps, actual_updates=agent.updates, updates=agent.updates, update_attempts=agent.attempts,
        config_hash=digest(agent.config), source_hashes=hashes, source_fingerprint=fingerprint,
        checkpoint_reload_passed=True, eval_rng_unchanged=True, finite_diagnostics=True,
        checkpoint_roundtrip=True, evaluation_rng_unchanged=True, diagnostic_rng_unchanged=True,
        score_coverage=1. if agent.cce else None, max_episode_length=agent.max_episode_length,
        eval_bank_hash=agent.eval_bank_hash, seconds=time.monotonic()-started,
        first_block_seconds=block_costs[0], steady_block_seconds=float(np.mean(block_costs[1:])),
        endpoint=float(np.mean([r['win_rate'] for r in evals[-4:]])) if args.mode == 'train' else None,
        run=str(args.out.resolve()))
    dump(args.out / 'complete.json', report)
    print('COMPLETE: ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
