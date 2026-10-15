"""Compatibility name for the shared array-backed replay buffer."""

from counterfactual_rl.replay.array_buffer import ArrayReplayBuffer


ChessArrayReplayBuffer = ArrayReplayBuffer

__all__ = ["ChessArrayReplayBuffer"]
