"""Shared learning math used by every current environment."""

from .td import q_learning_targets, weighted_td_loss

__all__ = ["q_learning_targets", "weighted_td_loss"]
