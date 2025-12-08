# """
# Train a PPO agent on SMAC 3m map using global state.

# This script trains a centralized PPO agent that controls all 3 marines
# as a single "super-agent" using the global battlefield state.
# """

# import os
# import sys

# # Add src to path BEFORE importing counterfactual_rl
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# from smac.env import StarCraft2Env
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.vec_env import DummyVecEnv
# import gymnasium as gym
# import numpy as np

# from counterfactual_rl.environments.smac import CentralizedSmacWrapper

# # Set SC2 Path
# os.environ['SC2PATH'] = r'C:\Program Files (x86)\StarCraft II'

# class GymWrapper(gym.Env):
#     """
#     Wraps CentralizedSmacWrapper to be compatible with Stable Baselines3.
#     """
#     def __init__(self, map_name="3m"):
#         super().__init__()
#         # Create SMAC environment
#         self.smac_env = StarCraft2Env(map_name=map_name)
#         self.wrapped_env = CentralizedSmacWrapper(self.smac_env, use_state=True)
        
#         # Define action and observation spaces for Gym
#         self.action_space = gym.spaces.Discrete(self.wrapped_env.joint_action_space_size)
        
#         # Get state size from environment
#         obs, _ = self.wrapped_env.reset()
#         obs_size = len(obs)
#         self.observation_space = gym.spaces.Box(
#             low=-np.inf, 
#             high=np.inf, 
#             shape=(obs_size,), 
#             dtype=np.float32
#         )
        
#     def reset(self, seed=None, options=None):
#         obs, info = self.wrapped_env.reset(seed=seed, options=options)
#         return obs, info
    
#     def step(self, action):
#         return self.wrapped_env.step(action)
    
#     def close(self):
#         self.smac_env.close()


# def make_env():
#     """Factory function to create environment."""
#     return GymWrapper(map_name="3m")


# def train_ppo(
#     total_timesteps=200_000,
#     save_freq=50_000,
#     model_dir="models/ppo_smac_3m",
#     log_dir="logs/ppo_smac_3m"
# ):
#     """
#     Train PPO agent on SMAC 3m.
    
#     Args:
#         total_timesteps: Total training steps
#         save_freq: Save checkpoint every N steps
#         model_dir: Directory to save models
#         log_dir: Directory for tensorboard logs
#     """
#     # Create directories
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)
    
#     # Create vectorized environment
#     env = DummyVecEnv([make_env])
    
#     # Create PPO agent
#     model = PPO(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         tensorboard_log=None,  # Disabled to avoid tensorboard dependency
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,  # Encourage exploration
#     )
    
#     # Create checkpoint callback
#     checkpoint_callback = CheckpointCallback(
#         save_freq=save_freq,
#         save_path=model_dir,
#         name_prefix="ppo_smac_3m"
#     )
    
#     print(f"Starting training for {total_timesteps} timesteps...")
#     print(f"Models will be saved to: {model_dir}")
#     print(f"Logs will be saved to: {log_dir}")
#     print(f"Monitor training with: tensorboard --logdir {log_dir}")
    
#     # Train the agent
#     model.learn(
#         total_timesteps=total_timesteps,
#         callback=checkpoint_callback
#     )
    
#     # Save final model
#     final_model_path = os.path.join(model_dir, "ppo_smac_3m_final")
#     model.save(final_model_path)
#     print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
#     # Close environment
#     env.close()
    
#     return model


# if __name__ == "__main__":
#     # Train the agent
#     model = train_ppo(
#         total_timesteps=200_000,  # ~2-4 hours on CPU
#         save_freq=50_000,
#         model_dir="models/ppo_smac_3m",
#         log_dir="logs/ppo_smac_3m"
#     )
    
#     print("\n" + "="*50)
#     print("Training finished!")
#     print("Next steps:")
#     print("1. Load the model: model = PPO.load('models/ppo_smac_3m/ppo_smac_3m_final')")
#     print("2. Run counterfactual analysis with this trained agent")
#     print("="*50)
