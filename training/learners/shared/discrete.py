"""Shared finite-state DQN: one collector/update/scorer for all replay arms.

Environments own reset, observations and step. Shared modules own TD targets,
loss, consequence distances, priority mixing, storage and sampling.
"""
import hashlib
import json
import os
import pickle
from pathlib import Path
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from counterfactual_rl.learning.td import q_learning_targets, weighted_td_loss
from counterfactual_rl.scoring.weighted_wasserstein import weighted_wasserstein_scores
from counterfactual_rl.replay.rank_buffer import SelectionReplayBuffer


class Network(nn.Module):
    hidden_dim: int
    n_layers: int
    n_actions: int

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(self.n_layers):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.n_actions)(x)


def build_update(network, optimizer, gamma):
    @jax.jit
    def update(params, target, opt_state, obs, actions, rewards, next_obs, done, weights):
        targets = q_learning_targets(rewards, done, network.apply(params, next_obs),
                                     network.apply(target, next_obs), gamma, double_dqn=True)
        def loss(p):
            value, td, _ = weighted_td_loss(network.apply(p, obs), actions, targets, weights)
            return value, td
        (value, td), grad = jax.value_and_grad(loss, has_aux=True)(params)
        delta, opt_state = optimizer.update(grad, opt_state, params)
        return optax.apply_updates(params, delta), opt_state, td, value
    return update


def build_branches(env, horizon, gamma):
    """Exact simulator branches; one stochastic rollout/action, frozen policy.

    Q tables are exact network evaluations over all finite states. This avoids
    repeated MLP calls in the rollout, without changing greedy actions or tails.
    """
    def one(policy, target_values, state, action, key):
        def tick(carry, i):
            s, key, total, discount, ended = carry
            key, sk = jax.random.split(key)
            a = jnp.where(i == 0, action, policy[s])
            _, ns, r, done, _ = env.step(sk, s, a)
            return (ns, key, total + discount * jnp.where(ended, 0., r),
                    discount * gamma, ended | done), None
        (last, _, total, discount, ended), _ = jax.lax.scan(
            tick, (state, key, jnp.float32(0), jnp.float32(1), jnp.bool_(False)),
            jnp.arange(horizon))
        return total + discount * jnp.where(ended, 0., target_values[last])
    actions = jax.vmap(one, in_axes=(None, None, None, 0, 0))
    return jax.jit(jax.vmap(actions, in_axes=(None, None, 0, None, 0)))


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')


class DiscreteAgent:
    def __init__(self, config, env):
        self.config, self.env = config, env
        c = config
        self.network = Network(c['hidden_dim'], c['n_layers'], env.n_actions)
        # Separate streams: initialization, collection, scoring, evaluation.
        self.key = jax.random.PRNGKey(c['seed'])
        self.key, init = jax.random.split(self.key)
        self.params = self.network.init(init, env.reset(init)[0])
        self.target = jax.tree.map(jnp.copy, self.params)
        self.score_key = jax.random.PRNGKey(c['seed'] + 730001)
        self.optimizer = optax.chain(optax.clip_by_global_norm(10.), optax.adam(c['alpha']))
        self.opt_state = self.optimizer.init(self.params)
        per = c['PER_parameters']
        self.buffer = SelectionReplayBuffer(capacity=c['buffer_capacity'], eps=per['eps'],
            beta=per['beta'], max_priority=per['maximum_priority'], selection=c['arm'],
            rank_alpha=per['beta'], priority_mixing=c['priority_mixing'],
            mu=c['mu'], mu_c=c['mu_c'], mu_delta=c['mu_delta'], normalise_priorities='rank')
        self.update_fn = build_update(self.network, self.optimizer, c['gamma'])
        self.branch_fn = build_branches(env, c['cf_horizon'], c['cf_gamma'])
        self.all_obs = jax.vmap(env.get_obs)(jnp.arange(env.n_states, dtype=jnp.int32))
        self.q_table = jax.jit(lambda p: self.network.apply(p, self.all_obs))
        self.steps = self.updates = self.scored = 0
        self.last_loss = self.last_td = 0.
        self._build_collect()
        self._build_eval()
        self.states = jnp.full(c['n_envs'], env.start, dtype=jnp.int32)
        self.bank = np.asarray(jax.random.split(jax.random.PRNGKey(c['fixed_eval_seed']),
                                              c['eval_episodes'] * env.max_steps)).reshape(
                                                  c['eval_episodes'], env.max_steps, 2)

    def _build_collect(self):
        env, net = self.env, self.network
        def one(params, epsilon, state, keys):
            def tick(s, key):
                ak, ek, sk = jax.random.split(key, 3)
                a = jnp.where(jax.random.uniform(ek) < epsilon,
                              jax.random.randint(ak, (), 0, env.n_actions),
                              jnp.argmax(net.apply(params, env.get_obs(s))))
                _, ns, r, done, _ = env.step(sk, s, a)
                reset = env.reset(ak)[1]
                return jnp.where(done, reset, ns), (s, a, r, ns, done)
            return jax.lax.scan(tick, state, keys)
        self.collect_fn = jax.jit(jax.vmap(one, in_axes=(None, None, 0, 0)))

    def _build_eval(self):
        env, net = self.env, self.network
        def one(params, keys):
            def tick(carry, key):
                state, reward, length, ended = carry
                a = jnp.argmax(net.apply(params, env.get_obs(state)))
                _, ns, r, done, _ = env.step(key, state, a)
                return (ns, reward + jnp.where(ended, 0., r), length + ~ended, ended | done), None
            out, _ = jax.lax.scan(tick, (jnp.int32(env.start), jnp.float32(0),
                                            jnp.int32(0), jnp.bool_(False)), keys)
            return out[1], out[2]
        self.eval_fn = jax.jit(jax.vmap(one, in_axes=(None, 0)))

    def evaluation_identity(self):
        starts = np.full(len(self.bank), self.env.start, dtype=np.int32)
        return dict(eval_bank_hash=hashlib.sha256(starts.tobytes() + self.bank.tobytes()).hexdigest(),
                    eval_episodes=len(self.bank), bank_seed=self.config['fixed_eval_seed'],
                    eval_cutoff=self.env.max_steps)

    def evaluate(self):
        rewards, lengths = map(np.asarray, self.eval_fn(self.params, jnp.asarray(self.bank)))
        success = rewards > 0
        self.eval_raw = dict(success=success, length=lengths)
        return dict(success_rate=float(success.mean()), win_rate=float(success.mean()),
                    successful_length=float(lengths[success].mean()) if success.any() else None,
                    n_success=int(success.sum()), **self.evaluation_identity())

    def insert(self, outputs):
        s, a, r, ns, done = [np.asarray(x).reshape(-1) for x in outputs]
        written = self.buffer.add_batch(dict(s=s, a=a, r=r, **{"s'": ns}, done=done), jax_states=s)
        if self.config['arm'] == 'cce':
            q = self.q_table(self.params)
            target_values = self.q_table(self.target).max(axis=-1)
            self.score_key, key = jax.random.split(self.score_key)
            keys = jax.random.split(key, len(s) * self.env.n_actions).reshape(len(s), self.env.n_actions, 2)
            returns = np.asarray(self.branch_fn(q.argmax(axis=-1), target_values,
                                                jnp.asarray(s), jnp.arange(self.env.n_actions), keys))
            probs = np.asarray(jax.nn.softmax(q[s] / self.config['cf_action_prob_temperature'], axis=-1))
            scores = weighted_wasserstein_scores(returns, probs, a)
            if not np.isfinite(scores).all():
                raise FloatingPointError('Nonfinite CCE score')
            self.buffer.update_consequence_scores(written, scores, q_update_count=self.updates)
            self.scored += len(s)
            assert np.all(self.buffer.scored_at[:len(self.buffer)] >= 0)
        return written

    def update(self):
        batch, indices, weights = self.buffer.sample(self.config['batch_size'])
        s = jnp.asarray([t['s'] for t in batch])
        ns = jnp.asarray([t["s'"] for t in batch])
        self.params, self.opt_state, td, loss = self.update_fn(
            self.params, self.target, self.opt_state, self.env.get_obs(s),
            jnp.asarray([t['a'] for t in batch]), jnp.asarray([t['r'] for t in batch]),
            self.env.get_obs(ns), jnp.asarray([t['done'] for t in batch]), jnp.asarray(weights))
        td = np.asarray(td)
        if not np.isfinite(td).all():
            raise FloatingPointError('Nonfinite TD error')
        self.buffer.update_priorities(indices, td)
        self.updates += 1
        self.last_loss, self.last_td = float(loss), float(abs(td).mean())
        if self.updates % (self.config['target_update_freq'] // self.config['n_steps_per_update']) == 0:
            self.target = jax.tree.map(jnp.copy, self.params)

    def diagnostics(self):
        p = self.buffer._compute_priorities()
        n = len(p)
        return dict(ess=float(1 / np.sum(p*p)), ess_fraction=float(1 / np.sum(p*p) / n),
                    top_1pct_mass=float(np.sort(p)[-max(1, int(np.ceil(n*.01))):].sum()),
                    max_probability=float(p.max()), td_abs_mean=float(self.buffer.td_magnitudes[:n].mean()),
                    cce_mean=float(self.buffer.consequence_scores[:n].mean()),
                    scored_arrivals=self.scored, loss=self.last_loss)

    def learn(self, verbose=False):
        c = self.config
        np.random.seed(c['seed'])
        out = Path(os.environ['CCE_RUN_DIR'])
        np.savez_compressed(out/'evaluation_bank.npz', keys=self.bank,
                            starts=np.full(len(self.bank), self.env.start, dtype=np.int32))
        started = time.monotonic()
        def evaluate_and_save():
            metrics = self.evaluate()
            append(out/'eval.jsonl', dict(env_steps=self.steps, updates=self.updates, **metrics))
            np.savez_compressed(out/f'eval_{self.steps:09d}.npz', **self.eval_raw)
        evaluate_and_save()
        n = c['n_envs'] * c['collect_steps']
        for block in range(1, c['blocks'] + 1):
            epsilon = c['epsilon_start'] + min(1., self.steps / c['epsilon_decay_steps']) * (
                c['epsilon_end'] - c['epsilon_start'])
            self.key, key = jax.random.split(self.key)
            keys = jax.random.split(key, n).reshape(c['n_envs'], c['collect_steps'], 2)
            self.states, outputs = self.collect_fn(self.params, jnp.float32(epsilon), self.states, keys)
            self.insert(outputs)
            self.steps += n
            if self.steps > c['warmup_steps']:
                for _ in range(n // c['n_steps_per_update']):
                    self.update()
            append(out/'progress.jsonl', dict(block=block, env_steps=self.steps, updates=self.updates,
                epsilon=epsilon, seconds=time.monotonic()-started, **self.diagnostics()))
            if block % c['eval_every_blocks'] == 0 or block == c['blocks']:
                evaluate_and_save()
                valid = slice(0, len(self.buffer))
                np.savez_compressed(out/f'replay_{self.steps:09d}.npz',
                    cce=self.buffer.consequence_scores[valid], td=self.buffer.td_magnitudes[valid],
                    scored_at=self.buffer.scored_at[valid], probabilities=self.buffer._compute_priorities())
                with (out/'last.pkl').open('wb') as f:
                    pickle.dump(dict(params=jax.device_get(self.params), config=c, env_steps=self.steps), f)
        assert self.updates == (self.steps - c['warmup_steps']) // c['n_steps_per_update']
        assert c['arm'] != 'cce' or self.scored == self.steps
        return self
