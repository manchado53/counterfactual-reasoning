"""Step-budget loop moved from the sealed October 1 transfer launcher.

Collection, insert/scoring, update, target sync and epsilon order are retained.
Run-path selection and raw record output replace the old SLURM-only launch shell.
This does not support exact resume or run the held-out final-test bank.
"""
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from .dqn import _MetricsLogger
from .transfer import TransferCCE, TransferDQN


def _append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


class StepBudget:
    def evaluation_identity(self, n_episodes=None):
        n = self.config['eval_episodes'] if n_episodes is None else n_episodes
        bank = np.asarray(jax.random.split(jax.random.PRNGKey(self.config['fixed_eval_seed']), n * 201))
        return {'eval_bank_hash': hashlib.sha256(bank.tobytes()).hexdigest(),
                'bank_seed': self.config['fixed_eval_seed'], 'eval_episodes': n,
                'eval_cutoff': 200, 'bank_generator': 'jax.random.split(seed,n*201)'}

    def learn(self, n_episodes=None, verbose=True):
        if n_episodes is not None:
            raise ValueError('Fixed transfer takes blocks, not an episode override')
        cfg = self.config
        np.random.seed(cfg['seed'])  # historical host replay stream, not isolated
        self.metrics_logger = _MetricsLogger(cfg, cfg['n_episodes'], None, cfg['eval_episodes'])
        timer = self.metrics_logger.timer
        timer.start('total')
        run = Path(self.metrics_logger.dir)
        try:
            self._build_collect_fn()
            self._key, key = jax.random.split(self._key)
            states = jax.vmap(lambda k: self.env.reset(k)[1])(jax.random.split(key, cfg['n_envs']))
            carry = self._initial_collect_carry(states)
            n = cfg['n_envs'] * cfg['collect_steps']
            terminal = np.array([s for s, tile in enumerate(''.join(self.env.desc)) if tile in 'HG'])
            started = time.monotonic()
            loss_sum, loss_n = 0., 0
            cce = cfg['algorithm'] == 'consequence-dqn'
            for block in range(1, cfg['blocks'] + 1):
                self._key, key = jax.random.split(self._key)
                keys = jax.random.split(key, n).reshape(cfg['n_envs'], cfg['collect_steps'], 2)
                carry, out = self._collect_fn(self.params, jnp.float32(self.epsilon), carry, keys)
                jax.block_until_ready(out[4])
                assert not np.isin(np.asarray(out[0]), terminal).any(), 'Stepped a terminal state'
                with timer('insert_and_score', episode=self._current_episode):
                    self._add_chunk_to_buffer(out)
                ended = np.asarray(out[7], dtype=bool)
                self.episode_returns.extend(np.asarray(out[5])[ended].tolist())
                self.episode_lengths.extend(np.asarray(out[6])[ended].tolist())
                episode = self._current_episode = len(self.episode_returns)
                start_update = self.total_steps // self.n_steps_per_update
                self.total_steps += n
                end_update = self.total_steps // self.n_steps_per_update
                freq = max(1, self.target_update_freq // self.n_steps_per_update)
                with timer('q_updates', episode=episode):
                    for update in range(start_update, end_update):
                        self._update()
                        loss_sum += self._last_loss
                        loss_n += 1
                        if update % freq == 0:
                            self._update_target_network()
                decay = min(1., episode / max(1, self.epsilon_decay_episodes))
                self.epsilon = self.epsilon_start + (self.epsilon_end-self.epsilon_start) * decay
                coverage = None
                if cce:
                    coverage = float(np.mean(self.buffer.scored_at[:len(self.buffer)] >= 0))
                    assert coverage == 1. and self._scored_arrivals == self.total_steps
                progress = dict(block=block, blocks=cfg['blocks'], env_steps=self.total_steps,
                                episode=episode, updates=self.q_update_count, epsilon=self.epsilon,
                                seconds=time.monotonic()-started, scored_frac=coverage,
                                max_episode_length=max(self.episode_lengths, default=0))
                _append(run / 'coverage.jsonl', progress)
                if block % cfg['eval_every_blocks'] == 0 or block == cfg['blocks']:
                    with timer('eval', episode=episode):
                        metrics = self.evaluate(cfg['eval_episodes'])
                    _append(run / 'eval.jsonl', {**progress, **metrics, **self.evaluation_identity()})
                    self.metrics_logger.log_eval(episode, self.q_update_count, self.epsilon, metrics)
                    q = np.asarray(jax.vmap(self.network.apply, in_axes=(None, 0))(
                        self.params, jnp.arange(self.n_states)))
                    assert np.isfinite(q).all()
                    _append(run / 'qdiag.jsonl', dict(env_steps=self.total_steps,
                            updates=self.q_update_count, loss_mean=loss_sum/max(1,loss_n),
                            td_abs_last=self._last_td_abs, q_mean=float(q.mean()),
                            q_abs_max=float(abs(q).max())))
                    loss_sum, loss_n = 0., 0
                    if cce:
                        _append(run / 'ess.jsonl', dict(env_steps=self.total_steps,
                                **self.buffer.priority_diagnostics(now=self.q_update_count)))
                    self.save(str(run / 'last.pkl'))
                    if verbose:
                        print(json.dumps(progress), flush=True)
            assert self.q_update_count == self.total_steps // cfg['n_steps_per_update']
            self.collector_state = carry
            return self
        finally:
            timer.stop('total')
            self.metrics_logger.close()


class FixedDQN(StepBudget, TransferDQN):
    pass


class FixedCCE(StepBudget, TransferCCE):
    pass
