"""Historical strict checkpoint loader; training remains in shared source."""
import pickle
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from counterfactual_rl.agents.frozen_lake.dqn import FrozenLakeDQN as Base

class FrozenLakeDQN(Base):
    @staticmethod
    def resolve_slip(config: dict) -> float:
        """
        Effective slip probability a config implies, mirroring FrozenLakeEnv.

        An explicit slip_prob wins; otherwise is_slippery=True means 2/3 (the
        original [1/3, 1/3, 1/3] dynamics) and is_slippery=False means 0.
        Older checkpoints predate slip_prob and carry only is_slippery.
        """
        slip = config.get('slip_prob', None)
        if slip is not None:
            return float(slip)
        return (2.0 / 3.0) if config.get('is_slippery', True) else 0.0


    @classmethod
    def from_checkpoint(cls, path: str, overrides: Optional[dict] = None):
        """
        Build an agent whose env matches the checkpoint's own saved config.

        Use this instead of `FrozenLakeDQN()` + `.load()` whenever the checkpoint
        may not come from the default MDP — `load()` restores weights but never
        rebuilds the env, so a bare constructor silently scores the default
        slippery map.
        """
        with open(path, 'rb') as f:
            saved = pickle.load(f).get('config') or {}
        config = dict(saved)
        if overrides:
            config.update(overrides)
        agent = cls(config)
        agent.load(path)
        return agent


    def _assert_env_matches(self, saved: dict, path: str):
        """Fail loudly when the live env is not the one the checkpoint trained in."""
        want_map = saved.get('map_name', self.config['map_name'])
        if want_map != self.config['map_name']:
            raise ValueError(
                f"Checkpoint/env map mismatch loading {path}: checkpoint trained on "
                f"map_name={want_map!r} but this agent's env is "
                f"{self.config['map_name']!r}. Use FrozenLakeDQN.from_checkpoint(path), "
                f"or pass strict_env=False if the mismatch is deliberate."
            )
        want_slip = self.resolve_slip(saved)
        if abs(want_slip - self.env.slip_prob) > 1e-9:
            raise ValueError(
                f"Checkpoint/env slip mismatch loading {path}: checkpoint trained at "
                f"slip_prob={want_slip:.6f} but this agent's env is at "
                f"slip_prob={self.env.slip_prob:.6f}. Scoring would use the wrong MDP. "
                f"Use FrozenLakeDQN.from_checkpoint(path), or pass strict_env=False "
                f"if the mismatch is deliberate."
            )


    def load(self, path: str, strict_env: bool = True):
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        saved_config = ckpt.get('config')
        if strict_env and saved_config:
            self._assert_env_matches(saved_config, path)
        self.params = jax.tree.map(jnp.array, ckpt['params'])
        self.target_params = jax.tree.map(jnp.array, ckpt['target_params'])
        self.opt_state = jax.tree.map(
            lambda x: jnp.array(x) if hasattr(x, 'shape') else x, ckpt['opt_state']
        )
        self.episode_returns = ckpt.get('episode_returns', [])
        self.episode_lengths = ckpt.get('episode_lengths', [])
        self.total_steps = ckpt.get('total_steps', 0)
        self.epsilon = ckpt.get('epsilon', self.epsilon_start)
        self._build_jit_fns()
