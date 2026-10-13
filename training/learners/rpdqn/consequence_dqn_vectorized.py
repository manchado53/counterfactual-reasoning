"""
Vectorized consequence-weighted DQN for the neural CVRP.

Mirror of ``agents/jax_nav/consequence_dqn_vectorized.py``. The collect loop is
the plain vectorized one plus one extra scan output: the PRE-step ``CVRPState``
for every transition, which is what CCE replays when it scores.

Resolution order is by MRO rather than jax_nav's explicit delegation:

    CVRPNeuralConsequenceDQNVectorized
      -> CVRPNeuralConsequenceDQN     _update (scoring), buffer, cf_* knobs
      -> CVRPNeuralDQNVectorized      learn, evaluate, _build_eval_fn
      -> CVRPNeuralDQN                network, _update_step, save/load

so the CCE ``_update`` and the vectorized ``learn`` compose without either being
copied.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from .consequence_dqn import CVRPNeuralConsequenceDQN
from .dqn_vectorized import CVRPNeuralDQNVectorized, _tree_where, _explore_action


class CVRPNeuralConsequenceDQNVectorized(CVRPNeuralConsequenceDQN, CVRPNeuralDQNVectorized):
    """CCE-weighted DQN with vectorized collection over n_envs instances."""

    def _build_collect_fn(self):
        env = self.env
        network = self.network
        mode = self.exploration

        def _collect_single_env(params, temp, init_state, step_keys):
            def _step(carry, step_key):
                state, obs, already_done, cum_return, ep_length = carry
                k_reset, k_act, k_env = jax.random.split(step_key, 3)

                new_obs, new_state = env.reset(k_reset)
                state = _tree_where(already_done, new_state, state)
                obs = jnp.where(already_done, new_obs, obs)
                cum_return = jnp.where(already_done, jnp.float32(0.0), cum_return)
                ep_length = jnp.where(already_done, jnp.int32(0), ep_length)

                q = network.apply(params, obs)
                action = _explore_action(k_act, q, temp, mode)

                next_obs, next_state, reward, done, _ = env.step(k_env, state, action)
                new_cum = cum_return + reward
                new_len = ep_length + jnp.int32(1)

                ep_return_out = jnp.where(done, new_cum, jnp.float32(0.0))
                ep_length_out = jnp.where(done, new_len, jnp.int32(0))

                return (next_state, next_obs, done, new_cum, new_len), (
                    obs, action, reward, next_obs, done,
                    ep_return_out, ep_length_out, done,
                    # PRE-step state: the counterfactual branches from here, so it must
                    # be the state BEFORE the action, carrying that episode's own
                    # coordinates and demands.
                    state,
                )

            init_obs = env.get_obs(init_state)
            init_carry = (
                init_state, init_obs, jnp.bool_(False), jnp.float32(0.0), jnp.int32(0),
            )
            (final_state, _, _, _, _), outputs = jax.lax.scan(_step, init_carry, step_keys)
            return final_state, outputs

        self._collect_fn = jax.jit(
            jax.vmap(_collect_single_env, in_axes=(None, None, 0, 0))
        )

    def _add_chunk_to_buffer(self, outputs):
        obs_np      = np.array(outputs[0])   # (n_envs, collect_steps, n_nodes, F)
        actions_np  = np.array(outputs[1])
        rewards_np  = np.array(outputs[2])
        next_obs_np = np.array(outputs[3])
        dones_np    = np.array(outputs[4])
        states      = outputs[8]             # CVRPState pytree, (n_envs, collect_steps, ...)

        n = obs_np.shape[0] * obs_np.shape[1]
        # Flatten (n_envs, collect_steps) into one leading axis on every leaf, so the
        # buffer can slice one state per transition with jax.tree.map.
        #
        # np.asarray is NOT cosmetic. ConsequenceReplayBuffer.add_batch slices with a
        # per-transition `jax.tree.map(lambda leaf: leaf[i], ...)` Python loop; handing
        # it live device arrays makes that 2048 x 7 JAX dispatches per chunk. Measured
        # on Gate 0 (job 274697): buffer.add was 227s of a 398s run -- 58% of wall
        # clock, versus 0.7s for the PER arm, which already passed numpy. Pulling the
        # states host-side once costs one transfer and makes each slice a numpy view.
        # The scoring path re-stacks them with jnp.stack, so nothing downstream cares
        # (jax_nav stores numpy leaves for the same reason).
        jax_states = jax.tree.map(
            lambda leaf: np.asarray(leaf).reshape(n, *leaf.shape[2:]), states
        )

        # A slot that gets overwritten no longer holds the score it was stamped with.
        # The buffer is a ring, so the n slots from the current write position are the
        # ones about to be replaced. Without this, scored_frac would only ever climb.
        cap = self.config['buffer_capacity']
        idx = (np.arange(self._buffer_write_pos, self._buffer_write_pos + n) % cap)
        self._scored_at[idx] = -1
        self._buffer_write_pos = int((self._buffer_write_pos + n) % cap)

        self.buffer.add_batch({
            's':    obs_np.reshape(n, *self.obs_shape),
            'a':    actions_np.reshape(n),
            'r':    rewards_np.reshape(n).astype(np.float32),
            "s'":   next_obs_np.reshape(n, *self.obs_shape),
            'done': dones_np.reshape(n),
        }, jax_states=jax_states)

    # ------------------------------------------------------------------

    def _log_priority_diagnostics(self, episode: int, metrics: dict):
        """One JSON line per eval: realized concentration of the sampling distribution.

        Wrapped so a diagnostics failure can never kill a training run.
        """
        try:
            diag = self.buffer.priority_diagnostics()
            if not diag:
                return
            diag.update(self.coverage_diagnostics())
            diag.update({
                'episode': episode,
                'mean_tour': metrics.get('mean_tour'),
                'quality': metrics.get('win_rate'),
                'mu_c': self.config.get('mu_c'),
                'mu_delta': self.config.get('mu_delta'),
                'mu': self.config.get('mu'),
                'priority_mixing': self.config.get('priority_mixing'),
                'consequence_metric': self.consequence_metric,
                'cf_horizon': self.cf_horizon,
                'cf_n_rollouts': self.cf_n_rollouts,
                'q_update_count': self.q_update_count,
            })
            path = os.path.join(self.metrics_logger.dir, 'ess.jsonl')
            with open(path, 'a') as f:
                f.write(json.dumps(diag, default=float) + '\n')
        except Exception as e:  # pragma: no cover - diagnostics must never be fatal
            print(f"Warning: priority diagnostics failed: {e}", flush=True)

    def evaluate(self, n_episodes: int = 512, init_states=None) -> dict:
        metrics = CVRPNeuralDQNVectorized.evaluate(self, n_episodes, init_states=init_states)
        if getattr(self, 'metrics_logger', None) is not None:
            self._log_priority_diagnostics(self._current_episode, metrics)
        return metrics
