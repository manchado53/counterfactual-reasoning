"""Connect Four model/settings with the JaxNav ranked CCE-to-TD sampler."""
import copy
import json
import os
import pickle
import time
from pathlib import Path
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from training.envs import connect_four_transfer as task
from counterfactual_rl.replay.buffers import PrioritizedReplayBuffer
from counterfactual_rl.replay.decay import DecayBuffer, blend_weight
from .legacy.policies import Connect4QNetwork
from .transfer import consequence, build_rollouts, targets
from .evaluation_v2 import Evaluator
from .protocol_v2 import validate


class Network(nn.Module):
    @nn.compact
    def __call__(self, obs):
        return Connect4QNetwork()(obs)[..., 0, :]


def build_update(network, optimizer, gamma):
    @jax.jit
    def update(params, target, opt_state, batch, weights):
        online = network.apply(params, batch['s_next'])
        y = jax.lax.stop_gradient(targets(batch['r'], batch['done'], online,
            network.apply(target, batch['s_next']), batch['next_mask'], gamma))
        def loss_fn(p):
            q = network.apply(p, batch['s'])
            chosen = jnp.take_along_axis(q, batch['a'][:, None], -1)[:, 0]
            td = y-chosen
            return jnp.mean(weights*td**2), (td, chosen, q)
        (loss, (td, chosen, q)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        diag = jnp.array([loss, jnp.mean(chosen), jnp.max(jnp.abs(q)), jnp.mean(y),
                          jnp.max(jnp.abs(y)), jnp.mean(jnp.abs(td)), jnp.max(jnp.abs(td))])
        return params, opt_state, td, diag
    return update


class Agent:
    def __init__(self, config):
        validate(config)
        self.config = copy.deepcopy(config)
        self.arm, self.seed = config['arm'], config['seed']
        self.cce = self.arm == 'cce_decay'
        self.network = Network()
        init, self.key, self.score_key = jax.random.split(jax.random.PRNGKey(self.seed), 3)
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 18]))
        self.params = self.network.init(init, jnp.zeros(84))
        self.target_params = jax.tree.map(jnp.copy, self.params)
        self.optimizer = optax.chain(optax.clip_by_global_norm(config['gradient_clip']), optax.adam(config['alpha']))
        self.opt_state = self.optimizer.init(self.params)
        kw = dict(capacity=config['buffer_capacity'], eps=config['replay_eps'], beta=config['replay_beta'], max_priority=1.)
        self.buffer = (DecayBuffer(**kw, priority_mixing='multiplicative', mu_c=1., mu_delta=1.,
                      normalise_priorities='rank') if self.cce else
                       PrioritizedReplayBuffer(**kw, uniform=self.arm == 'uniform'))
        self.collect_fn = task.build_collect(self.network)
        self.rollout_fn = build_rollouts(self.network, config['cf_horizon'])
        self.update_fn = build_update(self.network, self.optimizer, config['gamma'])
        self.key, init = jax.random.split(self.key)
        self.carry = task.initial_carry(jax.random.split(init, config['n_envs']))
        self.steps = self.updates = self.attempts = self.episodes = self.total_draws = 0
        self.draw_counts = np.zeros(config['buffer_capacity'], np.int64)
        self.epsilon = 1.
        self.last_diag = {}
        self.evaluator = Evaluator(self.network, config['eval_bank_hash'])

    def evaluation_identity(self):
        return dict(eval_bank_hash=self.config['eval_bank_hash'], eval_unique_cases=1000,
                    training_bank_hash=self.config['training_bank_hash'], eval_independent_trials=False)

    def rng_identity(self):
        return pickle.dumps((np.asarray(self.key), np.asarray(self.score_key),
                             self.replay_rng.bit_generator.state, np.random.get_state()))

    def evaluate(self):
        before = self.rng_identity()
        result = self.evaluator.evaluate(self.params)
        assert before == self.rng_identity(), 'Evaluation changed training RNG'
        return result

    def collect(self):
        cfg = self.config
        steps, nenv = cfg['collect_steps'], cfg['n_envs']
        self.key, key = jax.random.split(self.key)
        keys = jax.random.split(key, steps*nenv).reshape(steps, nenv, 2)
        self.carry, rows = self.collect_fn(self.params, jnp.float32(self.epsilon), self.carry, keys)
        states = jax.tree.map(lambda x: x.reshape((-1,)+x.shape[2:]), rows.pop('state'))
        flat = {k: np.asarray(v).reshape((-1,)+v.shape[2:]) for k, v in rows.items()}
        n = len(flat['a'])
        assert flat['mask'][np.arange(n), flat['a']].all()
        assert not np.asarray(states.terminated).any()
        assert np.all(np.asarray(states.current_player) == flat['learner'])
        slots = (self.buffer._write_pos + np.arange(n)) % self.buffer.capacity
        self.draw_counts[slots] = 0
        self.buffer.add_batch({k: flat[k] for k in ('s','a','r','s_next','done','mask','next_mask')})
        if self.cce:
            for lo in range(0, n, cfg['cf_score_chunk']):
                hi = min(lo+cfg['cf_score_chunk'], n)
                sub = jax.tree.map(lambda x: x[lo:hi], states)
                self.score_key, sk = jax.random.split(self.score_key)
                keys = jax.random.split(sk, (hi-lo)*7).reshape(hi-lo, 7, 2)
                returns, terminal = self.rollout_fn(self.params, self.target_params, sub,
                    jnp.asarray(flat['learner'][lo:hi]), jnp.arange(7), keys)
                assert np.asarray(terminal).all()
                q = self.network.apply(self.params, jnp.asarray(flat['s'][lo:hi]))
                scores = consequence(returns, q, jnp.asarray(flat['mask'][lo:hi]), flat['a'][lo:hi])
                assert np.isfinite(scores).all()
                self.buffer.update_consequence_scores(slots[lo:hi], scores, self.updates)
            assert (self.buffer.scored_at[:len(self.buffer)] >= 0).all()
        self.steps += n
        self.episodes += int(flat['done'].sum())
        self.epsilon = 1. - .95 * min(1., self.steps/(cfg['total_decisions']*.5))
        return flat

    def probabilities(self):
        n = len(self.buffer)
        if self.cce:
            return self.buffer._compute_priorities()
        p = np.ones(n) if self.arm == 'uniform' else self.buffer.priorities[:n].copy()
        return p/p.sum()

    def sample(self):
        if self.cce:
            assert (self.buffer.scored_at[:len(self.buffer)] >= 0).all(), 'Unscored row before draw'
        p = self.probabilities()
        indices = self.replay_rng.choice(len(p), self.config['batch_size'], p=p)
        np.add.at(self.draw_counts, indices, 1)
        self.total_draws += len(indices)
        batch = {k: jnp.asarray(np.stack([self.buffer.buffer[i][k] for i in indices]))
                 for k in ('s','a','r','s_next','done','next_mask')}
        return batch, indices, jnp.asarray(1./(len(p)*p[indices]))

    def learn_block(self):
        self.collect()
        stop = self.steps // self.config['update_every']
        for attempt in range(self.attempts, stop):
            if self.cce:
                self.buffer.blend = blend_weight(self.episodes, True,
                    self.config['decay_start'], self.config['decay_end'])
            if self.buffer.can_sample(self.config['batch_size']):
                batch, indices, weights = self.sample()
                self.params, self.opt_state, td, diag = self.update_fn(
                    self.params, self.target_params, self.opt_state, batch, weights)
                td, diag = np.asarray(td), np.asarray(diag)
                assert np.isfinite(td).all() and np.isfinite(diag).all()
                self.buffer.update_priorities(indices, td)
                self.updates += 1
                self.last_diag = dict(zip(('loss','q_mean','q_abs_max','target_mean','target_abs_max',
                                          'td_abs_mean','td_abs_max'), map(float, diag)))
            if (attempt+1) % (self.config['target_every']//self.config['update_every']) == 0:
                self.target_params = jax.tree.map(jnp.copy, self.params)
        self.attempts = stop

    def diagnostics(self):
        p = self.probabilities()
        n = len(p)
        out = dict(self.last_diag, steps=self.steps, episodes=self.episodes, updates=self.updates,
            attempts=self.attempts, epsilon=self.epsilon, buffer_size=n, total_replay_draws=self.total_draws,
            replay_draw_coverage=float(np.mean(self.draw_counts[:n] > 0)),
            replay_ess=float(1/np.sum(p*p)), replay_ess_fraction=float(1/(n*np.sum(p*p))),
            priority_top_1pct_mass=float(np.sort(p)[-max(1, int(np.ceil(n*.01))):].sum()),
            replay_max_probability=float(p.max()), fade=self.buffer.blend if self.cce else None,
            score_coverage=None, cce_score_mean=None, cce_score_max=None, cce_score_quantiles=None,
            score_age_mean=None, score_age_max=None)
        if self.cce:
            scores = self.buffer.consequence_scores[:n]
            age = self.updates-self.buffer.scored_at[:n]
            out.update(score_coverage=float(np.mean(self.buffer.scored_at[:n] >= 0)),
                cce_score_mean=float(scores.mean()), cce_score_max=float(scores.max()),
                cce_score_quantiles=np.quantile(scores, [0,.25,.5,.75,1]).tolist(),
                score_age_mean=float(age.mean()), score_age_max=int(age.max()))
        return out

    def save(self, path):
        # Evaluation checkpoint, not a resumable training snapshot.
        with Path(path).open('wb') as f:
            pickle.dump(jax.device_get(dict(params=self.params, config=self.config,
                steps=self.steps, resumable=False, **self.evaluation_identity())), f)

    def learn(self, verbose=False):
        out = Path(os.environ['CCE_RUN_DIR'])
        from .bank_v2 import BANK_PATH
        (out/'evaluation-bank.json').write_bytes(BANK_PATH.read_bytes())
        start = time.monotonic()
        initial = [np.asarray(x).copy() for x in jax.tree.leaves(self.params)]
        times, eval_times = [], []
        def emit(name, value):
            with (out/f'{name}.jsonl').open('a') as f:
                f.write(json.dumps(value, allow_nan=False)+'\n')
            print(name.upper()+': '+json.dumps(value, allow_nan=False), flush=True)
        def evaluate(block):
            clock = time.monotonic()
            result = self.evaluate()
            emit('eval', dict(result, block=block, env_steps=self.steps, arm=self.arm, seed=self.seed))
            self.save(out/'last.pkl')
            if not self.config['smoke']:
                self.save(out/f'checkpoint-{self.steps:09d}.pkl')
            eval_times.append(time.monotonic()-clock)
            return result
        evaluate(0)
        for block in range(1, self.config['blocks']+1):
            clock = time.monotonic()
            self.learn_block()
            times.append(time.monotonic()-clock)
            emit('progress', dict(self.diagnostics(), block=block, arm=self.arm, seed=self.seed,
                 block_seconds=times[-1], seconds=time.monotonic()-start))
            if block % self.config['eval_every_blocks'] == 0 or block == self.config['blocks']:
                final_eval = evaluate(block)
        assert any(not np.array_equal(a, np.asarray(b)) for a,b in zip(initial, jax.tree.leaves(self.params)))
        assert self.updates == self.attempts == self.steps//self.config['update_every']
        with (out/'last.pkl').open('rb') as f:
            saved = pickle.load(f)
        assert saved['eval_bank_hash'] == self.config['eval_bank_hash']
        before = self.rng_identity()
        assert self.evaluator.evaluate(saved['params']) == final_eval
        assert before == self.rng_identity()
        if self.config['smoke'] and self.cce:
            # Separate fixture on the live buffer; restore production blend afterward.
            original = self.buffer.blend
            self.buffer.blend = 0.
            p0 = self.probabilities().copy()
            self.buffer.blend = 1.
            p1 = self.probabilities().copy()
            np.testing.assert_allclose(p1, self.buffer.ranked_td(), rtol=0, atol=0)
            self.buffer.blend = .5
            np.testing.assert_allclose(self.probabilities(), .5*p0+.5*p1)
            self.buffer.blend = original
        complete = dict(passed=True, smoke=self.config['smoke'], arm=self.arm, seed=self.seed,
            steps=self.steps, updates=self.updates, backend=jax.default_backend(),
            source_commit=os.environ.get('C4_COMMIT'),
            score_coverage=1. if self.cce else None, checkpoint_reload_passed=True,
            eval_rng_unchanged=True, finite_diagnostics=True, parameters_changed=True,
            eval_bank_hash=self.config['eval_bank_hash'], seconds=time.monotonic()-start,
            first_block_seconds=times[0], steady_block_seconds=float(np.median(times[1:])),
            eval_seconds=float(np.median(eval_times[1:])), result_scope='smoke only' if self.config['smoke'] else 'full training')
        (out/'complete.json').write_text(json.dumps(complete, indent=2)+'\n')
        print('COMPLETE: '+json.dumps(complete), flush=True)
        return self
