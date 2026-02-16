"""
JAX/Flax DQN Agent with Prioritized Experience Replay for SMAX.

Equivalent to the PyTorch DQN but with JIT-compiled action selection
and update steps. Produces a JIT-compatible policy for counterfactual analysis.
"""

import os
import pickle
from datetime import datetime
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import optax

from .policies import CentralizedQNetwork
from ..shared.buffers import PrioritizedReplayBuffer
from ..shared.config import DEFAULT_CONFIG
from ..shared.metrics import MetricsLogger
from ..shared.utils import (
    create_smax_env,
    evaluate,
    get_action_masks,
    get_global_state,
    get_global_reward,
    is_done,
)


class DQN:
    """
    JAX/Flax DQN Agent with Prioritized Experience Replay for multi-agent SMAX.

    Same public interface as the PyTorch version but with JIT-compiled internals.

    Usage:
        env, key, env_info = create_smax_env(scenario='3m')
        agent = DQN(env, env_info)
        agent.learn()
        policy_fn = agent.make_policy_fn()  # JIT-compatible for counterfactuals
    """

    def __init__(self, env, env_info: Dict, config: Optional[Dict] = None):
        """
        Initialize the JAX DQN agent.

        Args:
            env: SMAX environment (HeuristicEnemySMAX)
            env_info: Dict with obs_dim, num_agents, actions_per_agent, etc.
            config: Optional config overrides
        """
        # Merge config with defaults
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Store environment
        self.env = env
        self.env_info = env_info
        self._key = jax.random.PRNGKey(0)

        # Environment parameters
        self.obs_dim = env_info['obs_dim']
        self.num_agents = env_info['num_agents']
        self.actions_per_agent = env_info['actions_per_agent']

        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.05)
        self.epsilon_decay_episodes = self.config.get('epsilon_decay_episodes', 5000)
        self.epsilon = self.epsilon_start
        self.alpha = self.config.get('alpha', 0.0005)
        self.batch_size = self.config.get('B', 32)
        self.target_update_freq = self.config.get('C', 500)
        self.n_steps_for_Q_update = self.config.get('n_steps_for_Q_update', 4)

        # Network
        hidden_dim = self.config.get('hidden_dim', 256)
        self.network = CentralizedQNetwork(
            num_agents=self.num_agents,
            actions_per_agent=self.actions_per_agent,
            hidden_dim=hidden_dim,
        )

        # Initialize parameters
        self._key, init_key = jax.random.split(self._key)
        dummy_state = jnp.zeros(self.obs_dim)
        self.params = self.network.init(init_key, dummy_state)
        self.target_params = jax.tree.map(jnp.copy, self.params)

        # Optimizer with gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(100.0),
            optax.adam(self.alpha),
        )
        self.opt_state = self.optimizer.init(self.params)

        # Replay buffer (numpy-based, framework-agnostic)
        per_params = self.config.get('PER_parameters', {})
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config.get('M', 100000),
            eps=per_params.get('eps', 0.01),
            beta=per_params.get('beta', 0.25),
            max_priority=per_params.get('maximum_priority', 1.0)
        )

        # Tracking
        self.total_steps = 0
        self.episode_returns = []
        self.episode_lengths = []

        # Build JIT-compiled functions
        self._build_jit_fns()

    def _build_jit_fns(self):
        """Build JIT-compiled action selection and update functions."""
        network = self.network
        gamma = self.gamma

        @jax.jit
        def greedy_action(params, state, masks):
            """JIT-compiled greedy action selection."""
            q = network.apply(params, state)  # (num_agents, actions_per_agent)
            masked_q = jnp.where(masks, q, -jnp.inf)
            return jnp.argmax(masked_q, axis=-1)  # (num_agents,)

        @jax.jit
        def update_step(params, target_params, opt_state, states, actions,
                        rewards, next_states, dones, next_masks, weights):
            """JIT-compiled batched update step."""

            def loss_fn(p):
                # Current Q-values
                q_values = jax.vmap(network.apply, in_axes=(None, 0))(p, states)
                # Gather Q-values for taken actions
                q_taken = jnp.take_along_axis(
                    q_values, actions[:, :, None], axis=-1
                ).squeeze(-1)  # (B, num_agents)
                q_taken = q_taken.sum(axis=-1)  # (B,)

                # Target Q-values
                next_q = jax.vmap(network.apply, in_axes=(None, 0))(target_params, next_states)
                next_q = jnp.where(next_masks, next_q, -jnp.inf)
                max_next_q = next_q.max(axis=-1).sum(axis=-1)  # (B,)
                targets = rewards + gamma * max_next_q * (1.0 - dones)

                # Weighted MSE loss
                td_errors = targets - q_taken
                loss = jnp.mean(weights * td_errors ** 2)
                return loss, td_errors

            (loss, td_errors), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss, td_errors

        self._greedy_action = greedy_action
        self._update_step = update_step

    def select_action(self, global_state: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        """
        Select joint action using epsilon-greedy with action masking.

        Args:
            global_state: Global observation, shape (obs_dim,)
            action_masks: Valid actions per agent, shape (num_agents, actions_per_agent)

        Returns:
            Joint action array, shape (num_agents,)
        """
        if np.random.uniform() < self.epsilon:
            # Random valid actions
            actions = []
            for agent_idx in range(self.num_agents):
                valid_actions = np.where(action_masks[agent_idx])[0]
                if len(valid_actions) > 0:
                    actions.append(np.random.choice(valid_actions))
                else:
                    actions.append(0)
            return np.array(actions)
        else:
            state_jnp = jnp.array(global_state)
            masks_jnp = jnp.array(action_masks, dtype=jnp.bool_)
            actions = self._greedy_action(self.params, state_jnp, masks_jnp)
            return np.array(actions)

    def _update(self):
        """Perform one batched update step if enough samples in buffer."""
        if not self.buffer.can_sample(self.batch_size):
            return

        transitions, indices, is_weights = self.buffer.sample(self.batch_size)

        # Stack transitions into JAX arrays
        states = jnp.array(np.array([d['s'] for d in transitions]))
        next_states = jnp.array(np.array([d["s'"] for d in transitions]))
        actions = jnp.array(np.array([d['a'] for d in transitions]), dtype=jnp.int32)
        rewards = jnp.array(np.array([d['r'] for d in transitions]), dtype=jnp.float32)
        dones = jnp.array(np.array([d['done'] for d in transitions]), dtype=jnp.float32)
        next_masks = jnp.array(np.array([d['next_masks'] for d in transitions]), dtype=jnp.bool_)
        weights = jnp.array(is_weights, dtype=jnp.float32)

        self.params, self.opt_state, loss, td_errors = self._update_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, next_states, dones, next_masks, weights
        )

        # Update priorities in buffer
        self.buffer.update_priorities(indices, np.array(td_errors))

    def _update_target_network(self):
        """Hard update of target network."""
        self.target_params = jax.tree.map(jnp.copy, self.params)

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'DQN':
        """
        Train the DQN agent on SMAX.

        Args:
            n_episodes: Number of episodes to train (default from config)
            verbose: Whether to print progress

        Returns:
            self (for chaining)
        """
        n_episodes = n_episodes or self.config['n_episodes']
        save_every = self.config.get('save_every', 500)
        eval_interval = self.config.get('eval_interval', None)
        eval_episodes = self.config.get('eval_episodes', 20)

        # Set up metrics log and run directory
        self.metrics_logger = MetricsLogger(
            backend='JAX', config=self.config, env_info=self.env_info,
            n_episodes=n_episodes, eval_interval=eval_interval, eval_episodes=eval_episodes,
        )

        # Model save paths within run directory
        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_win_rate = -1.0

        if verbose:
            print(f"Training JAX DQN on SMAX {self.env_info['scenario']}")
            print(f"  Obs type: {self.env_info['obs_type']}")
            print(f"  Obs dim: {self.obs_dim}")
            print(f"  Num agents: {self.num_agents}")
            print(f"  Actions per agent: {self.actions_per_agent}")
            print(f"  Epsilon: {self.epsilon_start} -> {self.epsilon_end} over {self.epsilon_decay_episodes} episodes")
            print(f"  Backend: JAX")
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
                joint_action = self.select_action(global_state, action_masks)

                self._key, step_key = jax.random.split(self._key)
                action_dict = {agent: joint_action[i] for i, agent in enumerate(agent_names)}

                obs, state, rewards, dones, infos = self.env.step(step_key, state, action_dict)

                next_global_state = get_global_state(obs, agent_names, self.env_info['obs_type'])
                next_action_masks = get_action_masks(self.env, state)

                global_reward = get_global_reward(rewards, agent_names)
                done = is_done(dones)

                transition = {
                    's': np.array(global_state),
                    'a': np.array(joint_action),
                    'r': global_reward,
                    "s'": np.array(next_global_state),
                    'done': done,
                    'masks': np.array(action_masks),
                    'next_masks': np.array(next_action_masks),
                }
                self.buffer.add(transition)

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
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay_progress

            if len(self.episode_returns) >= 100:
                avg_return = np.mean(self.episode_returns[-100:])
                pbar.set_description(f"Avg Return (100 ep): {avg_return:.2f}, eps: {self.epsilon:.3f}")

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

    def make_policy_fn(self):
        """
        Return a JIT-compatible policy function for counterfactual analysis.

        Returns:
            policy_fn(key, obs, avail_actions) -> action_dict

        The returned function is compatible with the vectorized counterfactual
        analyzer and can be used with jax.vmap and jax.jit.
        """
        network = self.network
        params = self.params
        agent_names = self.env_info['agent_names']
        obs_type = self.env_info['obs_type']

        def policy_fn(key, obs, avail_actions):
            # Convert obs dict to global state
            if obs_type == 'world_state':
                global_state = obs["world_state"]
            else:
                global_state = jnp.concatenate([obs[agent] for agent in agent_names])

            # Get Q-values and select greedy actions
            q_values = network.apply(params, global_state)

            # Build action dict
            action_dict = {}
            for i, agent_name in enumerate(agent_names):
                mask = avail_actions[agent_name]
                masked_q = jnp.where(mask, q_values[i], -jnp.inf)
                action_dict[agent_name] = jnp.argmax(masked_q)

            return action_dict

        return policy_fn

    def save(self, path: str):
        """Save agent state to pickle file."""
        checkpoint = {
            'params': jax.tree.map(np.array, self.params),
            'target_params': jax.tree.map(np.array, self.target_params),
            'opt_state': jax.tree.map(
                lambda x: np.array(x) if hasattr(x, 'shape') else x,
                self.opt_state
            ),
            'config': self.config,
            'env_info': self.env_info,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'total_steps': self.total_steps,
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load(self, path: str):
        """Load agent state from pickle file."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = jax.tree.map(jnp.array, checkpoint['params'])
        self.target_params = jax.tree.map(jnp.array, checkpoint['target_params'])
        self.opt_state = jax.tree.map(
            lambda x: jnp.array(x) if hasattr(x, 'shape') else x,
            checkpoint['opt_state']
        )
        self.episode_returns = checkpoint.get('episode_returns', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.total_steps = checkpoint.get('total_steps', 0)

        # Rebuild JIT functions with loaded params
        self._build_jit_fns()

    @classmethod
    def from_checkpoint(cls, path: str, env=None) -> 'DQN':
        """
        Create agent from saved checkpoint.

        Args:
            path: Path to checkpoint file
            env: Optional environment (if None, creates new one from saved config)

        Returns:
            Loaded DQN agent
        """
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        env_info = checkpoint['env_info']
        config = checkpoint['config']

        if env is None:
            env, _, _ = create_smax_env(
                scenario=env_info.get('scenario', config.get('scenario', '3m')),
                obs_type=env_info.get('obs_type', 'world_state'),
            )

        agent = cls(env, env_info, config)
        agent.load(path)
        return agent
