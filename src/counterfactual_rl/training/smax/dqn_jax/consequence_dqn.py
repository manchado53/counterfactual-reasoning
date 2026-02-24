"""
Consequence-weighted DQN (Algorithm 2) for SMAX.

Glue code that wires the existing counterfactual analysis pipeline
into the DQN training loop. Uses batched triple-vmap rollouts for
efficient consequence scoring.
"""

import os
import numpy as np
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from tqdm import tqdm

from .dqn import DQN
from .consequence_buffers import ConsequenceReplayBuffer
from counterfactual_rl.utils.smax_utils import get_valid_actions_mask
from counterfactual_rl.utils.smax_jax_utils import jax_actions_array_to_dict, jax_sum_rewards
from counterfactual_rl.utils.action_selection import (
    beam_search_top_k_joint_actions,
    convert_mask_to_indices,
)
from counterfactual_rl.analysis.metrics import compute_consequence_metric
from ..shared.metrics import MetricsLogger
from ..shared.utils import (
    evaluate,
    get_global_state,
    get_action_masks,
    get_global_reward,
    is_done,
)


class ConsequenceDQN(DQN):
    """
    Consequence-weighted DQN with batched counterfactual scoring (Algorithm 2).

    Inherits all DQN internals and overrides:
    - Buffer: ConsequenceReplayBuffer (Equations 2-4 priorities)
    - _update(): adds consequence scoring before Q-update
    - learn(): stores JAX state/obs in buffer for counterfactual rollouts
    """

    def __init__(self, env, env_info: Dict, config: Optional[Dict] = None):
        super().__init__(env, env_info, config)

        # Replace buffer with consequence-weighted version
        per_params = self.config.get('PER_parameters', {})
        self.buffer = ConsequenceReplayBuffer(
            capacity=self.config.get('M', 100000),
            eps=per_params.get('eps', 0.01),
            beta=per_params.get('beta', 0.25),
            max_priority=per_params.get('maximum_priority', 1.0),
            mu=self.config.get('mu', 0.5),
        )

        # Consequence scoring params
        self.score_interval = self.config.get('score_interval', 1)
        self.n_score_sample = self.config.get('n_score_sample', 256)
        self.consequence_metric = self.config.get('consequence_metric', 'jensen_shannon')
        self.consequence_aggregation = self.config.get('consequence_aggregation', 'weighted_mean')

        # Counterfactual rollout params
        self.cf_horizon = self.config.get('cf_horizon', 20)
        self.cf_n_rollouts = self.config.get('cf_n_rollouts', 48)
        self.cf_top_k = self.config.get('cf_top_k', 20)
        self.cf_gamma = self.config.get('cf_gamma', 0.99)

        self.q_update_count = 0

        # Batched rollout function (built once, params passed dynamically)
        self._compiled_batched_fn = None

    def _build_batched_rollout_fn(self):
        """
        Build triple-vmapped JIT function for batched consequence scoring.

        Parallelism levels:
            vmap over B transitions (different states)
              vmap over K actions (different first actions)
                vmap over N rollouts (different RNG keys)
                  lax.scan over H horizon steps (sequential)

        Q-network params are a dynamic argument so the compiled function
        is reused across scoring passes without recompilation.
        """
        env = self.env
        agent_names = self.env_info['agent_names']
        horizon = self.cf_horizon
        gamma = self.cf_gamma
        network = self.network
        obs_type = self.env_info['obs_type']

        def policy_from_params(params, key, obs, avail_actions):
            """Q-network greedy policy with params as explicit argument."""
            if obs_type == 'world_state':
                global_state = obs["world_state"]
            else:
                global_state = jnp.concatenate([obs[agent] for agent in agent_names])
            q_values = network.apply(params, global_state)
            action_dict = {}
            for i, agent_name in enumerate(agent_names):
                mask = avail_actions[agent_name]
                masked_q = jnp.where(mask, q_values[i], -jnp.inf)
                action_dict[agent_name] = jnp.argmax(masked_q)
            return action_dict

        def single_rollout(params, state, first_action_array, rng_key):
            """One rollout: take first_action, then follow policy for horizon-1 steps."""
            action_dict = jax_actions_array_to_dict(first_action_array, agent_names)
            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = env.step(step_key, state, action_dict)
            first_reward = jax_sum_rewards(rewards, agent_names)
            done = dones["__all__"]

            init_carry = (state, obs, rng_key, first_reward, jnp.float32(gamma), done)

            def scan_step(carry, _):
                state_c, obs_c, key_c, cum_return, discount, done_flag = carry
                avail_actions = env.get_avail_actions(state_c)
                key_c, policy_key, step_key = jax.random.split(key_c, 3)
                action_dict = policy_from_params(params, policy_key, obs_c, avail_actions)
                new_obs, new_state, rewards, dones, infos = env.step(
                    step_key, state_c, action_dict
                )
                step_reward = jax_sum_rewards(rewards, agent_names)
                masked_reward = jnp.where(done_flag, 0.0, step_reward)
                new_cum_return = cum_return + discount * masked_reward
                new_discount = jnp.where(done_flag, discount, discount * gamma)
                new_done = jnp.logical_or(done_flag, dones["__all__"])
                return (new_state, new_obs, key_c, new_cum_return, new_discount, new_done), None

            final_carry, _ = jax.lax.scan(
                scan_step, init_carry, xs=None, length=horizon - 1
            )
            return final_carry[3]  # cumulative return

        # Triple vmap: rollouts -> actions -> transitions
        batched_rollouts = jax.vmap(single_rollout, in_axes=(None, None, None, 0))
        batched_actions = jax.vmap(batched_rollouts, in_axes=(None, None, 0, 0))
        batched_transitions = jax.vmap(batched_actions, in_axes=(None, 0, 0, 0))

        self._compiled_batched_fn = jax.jit(batched_transitions)

    def _score_buffer_transitions(self):
        """
        Algorithm 2, lines 11-12: Batched consequence scoring.

        1. Uniform sample from buffer
        2. Beam search per transition (sequential Python, fast)
        3. Stack into arrays
        4. One JIT call via triple-vmap (all rollouts parallel on GPU)
        5. Compute metrics per transition (sequential CPU, scipy)
        6. Update buffer consequence scores
        """
        import time
        n_score = min(self.n_score_sample, len(self.buffer))
        if n_score == 0:
            return

        t_start = time.time()

        # Sample uniformly from buffer (line 11)
        transitions, indices = self.buffer.sample_uniform(n_score)

        t_sample = time.time()

        # --- Python-side: beam search per transition ---
        all_actions = []        # List of List[Tuple[int,...]], len K each
        all_action_probs = []   # List of Dict[Tuple, float]
        all_actual_actions = []
        valid_indices = []      # Indices into the sampled batch that have valid jax state
        valid_states = []

        for i, (transition, idx) in enumerate(zip(transitions, indices)):
            jax_state = self.buffer.get_jax_state(idx)
            if jax_state is None:
                continue

            actual_action = tuple(int(a) for a in transition['a'])

            # Get valid actions from stored state
            valid_actions_mask = get_valid_actions_mask(self.env, jax_state)
            valid_actions = convert_mask_to_indices(valid_actions_mask)

            # Beam search top-K
            actions_to_eval, joint_probs = beam_search_top_k_joint_actions(
                valid_actions=valid_actions,
                k=self.cf_top_k,
                return_probs=True,
            )

            # Ensure actual action is included
            if actual_action not in actions_to_eval:
                actions_to_eval = [actual_action] + actions_to_eval[:self.cf_top_k - 1]
                if actual_action not in joint_probs:
                    min_prob = min(joint_probs.values()) if joint_probs else 0.01
                    joint_probs[actual_action] = min_prob * 0.5

            # Pad to exactly top_k if fewer actions available
            while len(actions_to_eval) < self.cf_top_k:
                actions_to_eval.append(actual_action)

            valid_indices.append(i)
            valid_states.append(jax_state)
            all_actions.append(actions_to_eval)
            all_action_probs.append(joint_probs)
            all_actual_actions.append(actual_action)

        if not valid_states:
            return

        t_beam = time.time()

        B = len(valid_states)
        K = self.cf_top_k
        N = self.cf_n_rollouts

        # --- Stack into batched arrays ---
        # States: stack pytrees along new batch dimension
        batched_states = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *valid_states)

        # Actions: (B, K, n_agents)
        actions_array = jnp.array(
            [[list(a) for a in trans_actions] for trans_actions in all_actions],
            dtype=jnp.int32,
        )

        # Keys: (B, K, N, 2)
        self._key, subkey = jax.random.split(self._key)
        batch_keys = jax.random.split(subkey, B)  # (B, 2)
        all_keys = []
        for b_key in batch_keys:
            action_keys = jax.random.split(b_key, K)  # (K, 2)
            rollout_keys = jax.vmap(
                lambda k: jax.random.split(k, N)
            )(action_keys)  # (K, N, 2)
            all_keys.append(rollout_keys)
        keys_array = jnp.stack(all_keys, axis=0)  # (B, K, N, 2)

        # Build compiled function lazily on first scoring pass
        if self._compiled_batched_fn is None:
            print("Compiling batched rollout function (one-time cost)...")
            self._build_batched_rollout_fn()

        t_stack = time.time()

        # --- One JIT call: all rollouts in parallel ---
        returns_array = self._compiled_batched_fn(
            self.params, batched_states, actions_array, keys_array
        )
        returns_array = jax.block_until_ready(returns_array)  # (B, K, N)
        returns_np = np.array(returns_array)

        t_rollouts = time.time()

        # --- CPU-side: compute metrics per transition ---
        scores = np.zeros(B)
        for i in range(B):
            actual_action = all_actual_actions[i]
            actions_list = all_actions[i]
            action_probs = all_action_probs[i]

            # Build return distributions dict
            return_distributions = {}
            for j, action_tuple in enumerate(actions_list):
                if action_tuple not in return_distributions:
                    return_distributions[action_tuple] = returns_np[i, j]

            scores[i] = compute_consequence_metric(
                actual_action,
                return_distributions,
                metric=self.consequence_metric,
                action_probs=action_probs,
                aggregation=self.consequence_aggregation,
            )

        # Sanitize: replace NaN/inf with 0.0 (safe default = no consequence signal)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        t_metrics = time.time()

        # Update buffer (line 12)
        scored_buffer_indices = indices[np.array(valid_indices)]
        self.buffer.update_consequence_scores(scored_buffer_indices, scores)

        t_end = time.time()
        print(f"[Scoring] sample={t_sample-t_start:.3f}s  beam={t_beam-t_sample:.3f}s  "
              f"stack={t_stack-t_beam:.3f}s  rollouts={t_rollouts-t_stack:.3f}s  "
              f"metrics={t_metrics-t_rollouts:.3f}s  update={t_end-t_metrics:.3f}s  "
              f"total={t_end-t_start:.3f}s  (B={B})", flush=True)

    def _update(self):
        """Perform update with consequence scoring (Algorithm 2, lines 10-18)."""
        if not self.buffer.can_sample(self.batch_size):
            return

        self.q_update_count += 1

        # Scoring pass (lines 11-12)
        if (self.q_update_count % self.score_interval == 0
                and len(self.buffer) >= self.n_score_sample):
            self._score_buffer_transitions()

        # Sample via combined priorities (line 13)
        transitions, indices, is_weights = self.buffer.sample(self.batch_size)

        # Stack transitions into JAX arrays
        states = jnp.array(np.array([d['s'] for d in transitions]))
        next_states = jnp.array(np.array([d["s'"] for d in transitions]))
        actions = jnp.array(np.array([d['a'] for d in transitions]), dtype=jnp.int32)
        rewards = jnp.array(np.array([d['r'] for d in transitions]), dtype=jnp.float32)
        dones = jnp.array(np.array([d['done'] for d in transitions]), dtype=jnp.float32)
        next_masks = jnp.array(
            np.array([d['next_masks'] for d in transitions]), dtype=jnp.bool_
        )
        weights = jnp.array(is_weights, dtype=jnp.float32)

        # Update Q weights (lines 15, 17)
        self.params, self.opt_state, loss, td_errors = self._update_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, next_states, dones, next_masks, weights,
        )

        # Update TD magnitudes: m^delta_j <- |delta_j| (line 16)
        self.buffer.update_priorities(indices, np.array(td_errors))

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'ConsequenceDQN':
        """
        Train Consequence-weighted DQN (Algorithm 2).

        Same as parent DQN but stores JAX state/obs in buffer for
        future counterfactual rollouts.
        """
        n_episodes = n_episodes or self.config['n_episodes']
        save_every = self.config.get('save_every', 500)
        eval_interval = self.config.get('eval_interval', None)
        eval_episodes = self.config.get('eval_episodes', 20)

        # Set up metrics log and run directory
        self.metrics_logger = MetricsLogger(
            backend='JAX (Consequence)', config=self.config, env_info=self.env_info,
            n_episodes=n_episodes, eval_interval=eval_interval, eval_episodes=eval_episodes,
        )

        # Model save paths within run directory
        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_win_rate = -1.0

        if verbose:
            print(f"Training Consequence-weighted DQN on SMAX {self.env_info['scenario']}")
            print(f"  Obs type: {self.env_info['obs_type']}")
            print(f"  Obs dim: {self.obs_dim}")
            print(f"  Num agents: {self.num_agents}")
            print(f"  Actions per agent: {self.actions_per_agent}")
            print(f"  Epsilon: {self.epsilon_start} -> {self.epsilon_end} "
                  f"over {self.epsilon_decay_episodes} episodes")
            print(f"  mu: {self.config.get('mu', 0.5)}, metric: {self.consequence_metric}")
            print(f"  Score interval: {self.score_interval}, "
                  f"Score sample: {self.n_score_sample}")
            print(f"  CF rollouts: {self.cf_n_rollouts}, CF horizon: {self.cf_horizon}, "
                  f"CF top-K: {self.cf_top_k}")
            print(f"  Metrics dir: {self.metrics_logger.dir}")

        agent_names = self.env_info['agent_names']

        pbar = tqdm(range(n_episodes), disable=not verbose)
        for episode in pbar:
            # Reset environment
            self._key, reset_key = jax.random.split(self._key)
            obs, state = self.env.reset(reset_key)

            global_state = get_global_state(obs, agent_names, self.env_info['obs_type'])
            action_masks = get_action_masks(self.env, state)

            done = False
            episode_return = 0.0
            episode_length = 0

            while not done:
                # Save JAX state/obs BEFORE stepping (for counterfactual rollouts)
                saved_state = state
                saved_obs = obs

                joint_action = self.select_action(global_state, action_masks)

                self._key, step_key = jax.random.split(self._key)
                action_dict = {
                    agent: joint_action[i] for i, agent in enumerate(agent_names)
                }

                obs, state, rewards, dones_dict, infos = self.env.step(
                    step_key, state, action_dict
                )

                next_global_state = get_global_state(
                    obs, agent_names, self.env_info['obs_type']
                )
                next_action_masks = get_action_masks(self.env, state)

                global_reward = get_global_reward(rewards, agent_names)
                done = is_done(dones_dict)

                transition = {
                    's': np.array(global_state),
                    'a': np.array(joint_action),
                    'r': global_reward,
                    "s'": np.array(next_global_state),
                    'done': done,
                    'masks': np.array(action_masks),
                    'next_masks': np.array(next_action_masks),
                }
                # Store with JAX state/obs for future counterfactual rollouts
                self.buffer.add(transition, jax_state=saved_state, jax_obs=saved_obs)

                self.total_steps += 1
                if self.total_steps % self.n_steps_for_Q_update == 0:
                    self._update()

                if self.total_steps % self.target_update_freq == 0:
                    self._update_target_network()

                global_state = next_global_state
                action_masks = next_action_masks
                episode_return += global_reward
                episode_length += 1

            self.episode_returns.append(episode_return)
            self.episode_lengths.append(episode_length)

            # Epsilon decay (linear)
            decay_progress = min(1.0, episode / self.epsilon_decay_episodes)
            self.epsilon = (self.epsilon_start
                            + (self.epsilon_end - self.epsilon_start) * decay_progress)

            if len(self.episode_returns) >= 100:
                avg_return = np.mean(self.episode_returns[-100:])
                pbar.set_description(
                    f"Avg Return (100 ep): {avg_return:.2f}, eps: {self.epsilon:.3f}"
                )

            if (episode + 1) % save_every == 0:
                self.save(last_path)
                if verbose:
                    print(f"\nSaved checkpoint at episode {episode + 1}")

            if eval_interval and (episode + 1) % eval_interval == 0:
                metrics = evaluate(self, n_episodes=eval_episodes, parallel=True)
                self.metrics_logger.log_eval(episode + 1, self.epsilon, metrics)
                if metrics['win_rate'] > best_win_rate:
                    best_win_rate = metrics['win_rate']
                    self.save(best_path)
                    if verbose:
                        print(f"\nNew best model (win rate: {best_win_rate:.1%})")

        self.save(last_path)
        self.metrics_logger.plot_training_curves(self.episode_returns, self.episode_lengths)
        self.metrics_logger.close()
        if verbose:
            print(f"Training complete. Run saved to {self.metrics_logger.dir}")

        return self
