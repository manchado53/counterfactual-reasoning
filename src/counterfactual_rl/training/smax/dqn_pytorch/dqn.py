"""
DQN Agent with Prioritized Experience Replay for SMAX.

SB3-style implementation with .learn() method for training.
Adapted from Professor Kedziora's battleship DQN implementation.
"""

import os
import copy
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Optional

import jax

from .policies import CentralizedQNetwork
from ..shared.buffers import PrioritizedReplayBuffer
from ..shared.config import DEFAULT_CONFIG
from ..shared.utils import (
    create_smax_env,
    get_action_masks,
    get_global_state,
    get_global_reward,
    is_done,
)


class DQN:
    """
    DQN Agent with Prioritized Experience Replay for multi-agent SMAX.

    Features:
        - Centralized controller (single network for all agents)
        - Prioritized Experience Replay (PER) with TD-error priorities
        - Target network with hard updates
        - Importance sampling weights for unbiased gradients
        - Action masking for invalid actions
        - SB3-style .learn() method for training

    Usage:
        env, key, env_info = create_smax_env(scenario='3m')
        agent = DQN(env, env_info)
        agent.learn(n_episodes=2000)
        agent.save('model.pt')
    """

    def __init__(self, env, env_info: Dict, config: Optional[Dict] = None):
        """
        Initialize the DQN agent.

        Args:
            env: SMAX environment (wrapped with CTRolloutManager)
            env_info: Dict with obs_dim, num_agents, actions_per_agent, scenario
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

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Networks
        hidden_dim = self.config.get('hidden_dim', 256)
        self.Q = CentralizedQNetwork(
            self.obs_dim, self.num_agents, self.actions_per_agent, hidden_dim
        ).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)

        # Optimizer
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.alpha)

        # Replay buffer
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
            return self._greedy_action(global_state, action_masks)

    def _greedy_action(self, global_state: np.ndarray, action_masks: np.ndarray) -> np.ndarray:
        """Select greedy action based on Q-values with masking."""
        state_tensor = torch.tensor(global_state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_values = self.Q(state_tensor)

        mask_tensor = torch.tensor(action_masks, dtype=torch.bool, device=self.device)
        masked_q = q_values.clone()
        masked_q[~mask_tensor] = float('-inf')

        actions = torch.argmax(masked_q, dim=-1).cpu().numpy()
        return actions

    def get_q_values(self, global_state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a state (useful for counterfactual analysis).

        Args:
            global_state: Global observation, shape (obs_dim,)

        Returns:
            Q-values, shape (num_agents, actions_per_agent)
        """
        state_tensor = torch.tensor(global_state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.Q(state_tensor)
        return q_values.cpu().numpy()

    def _update(self):
        """Perform one batched update step if enough samples in buffer."""
        if not self.buffer.can_sample(self.batch_size):
            return

        transitions, indices, is_weights = self.buffer.sample(self.batch_size)

        # Stack transitions into batched tensors
        states = torch.tensor(
            np.array([d['s'] for d in transitions]), dtype=torch.float32, device=self.device
        )  # (B, obs_dim)
        next_states = torch.tensor(
            np.array([d["s'"] for d in transitions]), dtype=torch.float32, device=self.device
        )  # (B, obs_dim)
        actions = torch.tensor(
            np.array([d['a'] for d in transitions]), dtype=torch.long, device=self.device
        )  # (B, num_agents)
        rewards = torch.tensor(
            np.array([d['r'] for d in transitions]), dtype=torch.float32, device=self.device
        )  # (B,)
        dones = torch.tensor(
            np.array([d['done'] for d in transitions]), dtype=torch.float32, device=self.device
        )  # (B,)
        next_masks = torch.tensor(
            np.array([d['next_masks'] for d in transitions]), dtype=torch.bool, device=self.device
        )  # (B, num_agents, actions_per_agent)

        # Target Q-values — single batched forward pass
        with torch.no_grad():
            next_q = self.Q_target(next_states)  # (B, num_agents, actions_per_agent)
        next_q[~next_masks] = float('-inf')
        max_next_q = next_q.max(dim=-1)[0].sum(dim=-1)  # (B,)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Current Q-values — single batched forward pass
        q_values = self.Q(states)  # (B, num_agents, actions_per_agent)
        q_taken = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (B, num_agents)
        q_taken = q_taken.sum(dim=-1)  # (B,)

        # Update priorities
        td_errors = (targets - q_taken.detach()).cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        # Compute weighted loss
        weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
        loss = (weights * (q_taken - targets) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.Q.parameters(), 100)
        self.optimizer.step()

    def _update_target_network(self):
        """Hard update of target network."""
        self.Q_target.load_state_dict(self.Q.state_dict())

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
        save_path = self.config.get('save_path', 'models/smax_dqn.pt')

        if verbose:
            print(f"Training DQN on SMAX {self.env_info['scenario']}")
            print(f"  Obs type: {self.env_info['obs_type']}")
            print(f"  Obs dim: {self.obs_dim}")
            print(f"  Num agents: {self.num_agents}")
            print(f"  Actions per agent: {self.actions_per_agent}")
            print(f"  Epsilon: {self.epsilon_start} -> {self.epsilon_end} over {self.epsilon_decay_episodes} episodes")
            print(f"  Device: {self.device}")

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

                # Convert to action dict for SMAX
                action_dict = {agent: joint_action[i] for i, agent in enumerate(agent_names)}

                obs, state, rewards, dones, infos = self.env.step(step_key, state, action_dict)

                next_global_state = get_global_state(obs, agent_names, self.env_info['obs_type'])
                next_action_masks = get_action_masks(self.env, state)

                global_reward = get_global_reward(rewards, agent_names)
                done = is_done(dones)

                transition = {
                    's': global_state,
                    'a': joint_action,
                    'r': global_reward,
                    "s'": next_global_state,
                    'done': done,
                    'masks': action_masks,
                    'next_masks': next_action_masks,
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
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                self.save(save_path)
                if verbose:
                    print(f"\nSaved checkpoint at episode {episode + 1}")

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        self.save(save_path)
        if verbose:
            print(f"Training complete. Model saved to {save_path}")

        return self

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'Q_state_dict': self.Q.state_dict(),
            'Q_target_state_dict': self.Q_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'env_info': self.env_info,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'total_steps': self.total_steps,
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.Q.load_state_dict(checkpoint['Q_state_dict'])
        self.Q_target.load_state_dict(checkpoint['Q_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_returns = checkpoint.get('episode_returns', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.total_steps = checkpoint.get('total_steps', 0)

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
        checkpoint = torch.load(path, map_location='cpu')
        env_info = checkpoint['env_info']
        config = checkpoint['config']

        if env is None:
            env, _, _ = create_smax_env(
                scenario=env_info.get('scenario', config.get('scenario', '3m'))
            )

        agent = cls(env, env_info, config)
        agent.load(path)
        return agent
