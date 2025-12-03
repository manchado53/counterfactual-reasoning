"""
PPO agent training for FrozenLake and other environments
"""

from typing import Optional
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class PPOTrainer:
    """
    Wrapper for training PPO agents with configurable hyperparameters.

    This class provides a clean interface for training and evaluating
    PPO agents on various Gymnasium environments.
    """

    def __init__(
        self,
        env_id: str = "FrozenLake-v1",
        env_kwargs: Optional[dict] = None,
        policy_kwargs: Optional[dict] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        seed: int = 42,
        verbose: int = 1
    ):
        """
        Initialize PPO trainer.

        Args:
            env_id: Gymnasium environment ID
            env_kwargs: Keyword arguments for environment creation
            policy_kwargs: Keyword arguments for policy network
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to run per update
            batch_size: Minibatch size
            n_epochs: Number of epochs for each update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            seed: Random seed
            verbose: Verbosity level
        """
        self.env_id = env_id
        self.env_kwargs = env_kwargs or {}
        self.seed = seed
        self.verbose = verbose

        # Default policy architecture: small MLP
        if policy_kwargs is None:
            policy_kwargs = dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64])  # Fixed: dict not list
            )

        self.ppo_kwargs = {
            "policy": "MlpPolicy",
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "verbose": verbose,
            "seed": seed,
            "policy_kwargs": policy_kwargs,
        }

        self.model = None

    def train(
        self,
        total_timesteps: int = 100000,
        save_path: Optional[str] = None
    ) -> PPO:
        """
        Train a PPO agent.

        Args:
            total_timesteps: Number of training timesteps
            save_path: Optional path to save the trained model

        Returns:
            Trained PPO model
        """
        print(f"\n{'='*60}")
        print(f"Training PPO agent on {self.env_id}")
        print(f"{'='*60}\n")

        # Create environment
        env = gym.make(self.env_id, **self.env_kwargs)
        env = DummyVecEnv([lambda: env])

        # Create PPO agent
        self.model = PPO(env=env, **self.ppo_kwargs)

        # Train the agent
        self.model.learn(total_timesteps=total_timesteps)

        # Evaluate performance
        eval_env = gym.make(self.env_id, **self.env_kwargs)
        mean_reward, std_reward = evaluate_policy(
            self.model, eval_env, n_eval_episodes=100
        )
        print(f"\nTraining complete! Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Save model if path provided
        if save_path:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")

        return self.model

    def load(self, model_path: str) -> PPO:
        """
        Load a pre-trained model.

        Args:
            model_path: Path to saved model

        Returns:
            Loaded PPO model
        """
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)
        return self.model

    def evaluate(
        self,
        n_eval_episodes: int = 100,
        deterministic: bool = True
    ) -> tuple:
        """
        Evaluate the trained model.

        Args:
            n_eval_episodes: Number of episodes for evaluation
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (mean_reward, std_reward)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        eval_env = gym.make(self.env_id, **self.env_kwargs)
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic
        )

        print(f"Evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward
