"""DoorKey task adapter. One action is one replay and discount step.

The 50-action limit is a collection truncation, never an MDP terminal.
Observation is the one-hot encoding of the complete enumerated state.
"""
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from experiments.envs.doorkey import DoorKeyEnv


class Carry(NamedTuple):
    state: object
    obs: object
    pending_reset: object
    episode_return: object
    length: object


class QNetwork(nn.Module):
    n_states: int

    @nn.compact
    def __call__(self, state):
        x = jax.nn.one_hot(state, self.n_states)
        for _ in range(3):
            x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(7)(x)


def initial_carry(n_envs):
    zeros = jnp.zeros(n_envs, dtype=jnp.int32)
    return Carry(zeros, zeros, jnp.zeros(n_envs, dtype=bool),
                 jnp.zeros(n_envs, dtype=jnp.float32), zeros)


def ddqn_targets(online_next, target_next, rewards, terminated, gamma=.99):
    actions = jnp.argmax(online_next, axis=-1)
    value = jnp.take_along_axis(target_next, actions[..., None], axis=-1)[..., 0]
    return rewards + gamma * jnp.where(terminated, 0., value)


def weighted_scores(returns, probabilities, taken):
    """Source N=1 Wasserstein, float64 subtraction and ordered accumulation."""
    r = np.asarray(returns, dtype=np.float64)
    p = np.asarray(probabilities)
    taken = np.asarray(taken, dtype=int)
    reference = r[np.arange(len(r)), taken]
    num, den = np.zeros(len(r)), np.zeros(len(r))
    for action in range(7):
        alt = taken != action
        weight = p[:, action].astype(np.float64)
        num = np.where(alt, num + weight * np.abs(reference - r[:, action]), num)
        den = np.where(alt, den + weight, den)
    result = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    for i in np.flatnonzero(den <= 0):
        result[i] = np.mean([abs(reference[i] - r[i, a]) for a in range(7) if a != taken[i]])
    return result


class DoorKeyAdapter:
    def __init__(self):
        self.env = DoorKeyEnv(layout_name='6x6', slip_prob=0., random_start='fixed')
        self.net = QNetwork(self.env.n_states)
        self.next_state = self.env.next_states[:, :, 0]
        self.reward = self.env.rewards[:, :, 0]
        self.terminal = self.env.dones[:, :, 0]
        self.terminal_state = jnp.asarray([s[:2] == self.env.goal_cell for s in self.env._order])
        self.collect = jax.jit(jax.vmap(self._collect, in_axes=(None, None, 0, 0)))
        self.collect_actions = jax.jit(jax.vmap(self._collect_actions, in_axes=(0, 0)))
        self.branch_returns = jax.jit(self._branch_returns)
        self.eval_paths = jax.jit(jax.vmap(self._eval_path, in_axes=(None, 0)))

    def _step(self, carry, action):
        state = jnp.where(carry.pending_reset, 0, carry.state)
        total = jnp.where(carry.pending_reset, 0., carry.episode_return)
        length = jnp.where(carry.pending_reset, 0, carry.length)
        ns = self.next_state[state, action]
        reward = self.reward[state, action]
        terminated = self.terminal[state, action]
        total, length = total + reward, length + 1
        truncated = (length >= 50) & ~terminated
        ended = terminated | truncated
        row = dict(s=state, a=action, r=reward, next_s=ns, done=terminated,
                   truncated=truncated, ended=ended,
                   episode_return=jnp.where(ended, total, 0.),
                   episode_length=jnp.where(ended, length, 0),
                   terminal_start=self.terminal_state[state])
        return Carry(ns, ns, ended, total, length), row

    def _collect(self, params, epsilon, carry, keys):
        def step(c, key):
            # Hold=1. Fresh epsilon draw and action at each task action.
            _, ka, kx, _ = jax.random.split(key, 4)
            s = jnp.where(c.pending_reset, 0, c.state)
            greedy = jnp.argmax(self.net.apply(params, s))
            explore = jax.random.uniform(kx) < epsilon
            action = jnp.where(explore, jax.random.randint(ka, (), 0, 7), greedy)
            c, row = self._step(c, action)
            row['explore'] = explore
            return c, row
        return jax.lax.scan(step, carry, keys)

    def _collect_actions(self, carry, actions):
        return jax.lax.scan(self._step, carry, actions)

    def _branch_returns(self, params, target_params, states):
        # All branches still take all 60 actions. Computing the frozen Q table once
        # is exact for this finite deterministic full-state task; no score is skipped.
        ids = jnp.arange(self.env.n_states)
        q = self.net.apply(params, ids)
        tq = self.net.apply(target_params, ids)
        greedy = jnp.argmax(q, axis=-1)
        states = jnp.broadcast_to(states[:, None], (len(states), 7))
        actions = jnp.broadcast_to(jnp.arange(7), states.shape)
        ns = self.next_state[states, actions]
        rewards = self.reward[states, actions]
        done = self.terminal[states, actions]
        def step(carry, _):
            s, total, discount, ended = carry
            a = greedy[s]
            new_s = self.next_state[s, a]
            r = self.reward[s, a]
            new_end = ended | self.terminal[s, a]
            return (new_s, total + jnp.where(ended, 0., discount * r),
                    jnp.where(ended, discount, discount * .99), new_end), None
        (ns, total, discount, done), _ = jax.lax.scan(
            step, (ns, rewards, jnp.full(states.shape, .99), done), None, length=59)
        return total + jnp.where(done, 0., discount * jnp.max(tq[ns], axis=-1))

    def _eval_path(self, params, key):
        # key belongs to the independent bank. Deterministic task ignores it.
        def step(carry, _):
            state, ended, total, length = carry
            action = jnp.argmax(self.net.apply(params, state))
            ns = self.next_state[state, action]
            reward = self.reward[state, action]
            return (ns, ended | self.terminal[state, action],
                    total + jnp.where(ended, 0., reward), length + (~ended)), None
        (state, ended, total, length), _ = jax.lax.scan(
            step, (jnp.int32(0), jnp.bool_(False), jnp.float32(0), jnp.int32(0)), None, length=50)
        return total, length
