"""Pure PGX learner-turn adapter for the approved fixed-opponent transfer.

Role 0 moves first; role 1 moves second. Learner identity is kept separately
from PGX's current player. No reset occurs in step() or a CCE rollout.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pgx
from flax import struct

ENV = pgx.make("connect_four")
ORDER = jnp.array([3, 2, 4, 1, 5, 0, 6], dtype=jnp.int32)


def _windows():
    return jnp.array([
        [(r + k * dr) * 7 + c + k * dc for k in range(4)]
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(6) for c in range(7)
        if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7
    ], dtype=jnp.int32)


WINDOWS = _windows()


def legal_mask(state):
    # PGX exposes all-true masks at terminal states. The adapter does not.
    return state.legal_action_mask & ~(state.terminated | state.truncated)


def opponent(state):
    """Win, block, center; every tie uses ORDER, with no random key."""
    obs = state.observation
    rows = jnp.maximum(0, 5 - obs.any(-1).sum(0))

    def wins(col, channel):
        pieces = obs[:, :, channel].at[rows[col], col].set(True)
        return pieces.reshape(-1)[WINDOWS].all(-1).any()

    cols = jnp.arange(7)
    win = jax.vmap(lambda c: wins(c, 0))(cols)
    block = jax.vmap(lambda c: wins(c, 1))(cols)
    rank = jnp.where(win, 2, jnp.where(block, 1, 0))
    scores = jnp.where(legal_mask(state), rank, -1)
    return ORDER[jnp.argmax(scores[ORDER])]


def observe(state, learner):
    return ENV.observe(state, learner).astype(jnp.float32).reshape(84)


def reset(key, role):
    state = ENV.init(key)
    learner = jnp.where(role == 0, state.current_player, 1 - state.current_player)
    state = jax.lax.cond(role == 1, lambda s: ENV.step(s, opponent(s)), lambda s: s, state)
    return state, learner


class Step(NamedTuple):
    state: object
    reward: object
    done: object
    board_moves: object


def step(state, learner, action):
    """One learner move plus reply. Ended states are absorbing, with reward 0."""
    def live(s):
        first = ENV.step(s, action)
        ended = first.terminated | first.truncated
        last = jax.lax.cond(ended, lambda x: x, lambda x: ENV.step(x, opponent(x)), first)
        reward = first.rewards[learner] + jnp.where(ended, 0., last.rewards[learner])
        return Step(last, reward, last.terminated | last.truncated,
                    jnp.where(ended, 1, 2).astype(jnp.int32))
    return jax.lax.cond(state.terminated | state.truncated,
                        lambda s: Step(s, jnp.float32(0), jnp.bool_(True), jnp.int32(0)), live, state)


@struct.dataclass
class Carry:
    state: object
    learner: object
    role: object
    pending: object
    returns: object
    length: object
    board_length: object
    episode_id: object
    next_id: object


def initial_carry(keys):
    n = keys.shape[0]
    ids = jnp.arange(n, dtype=jnp.int32)
    role = ids % 2
    state, learner = jax.vmap(reset)(keys, role)
    return Carry(state, learner, role, jnp.zeros(n, bool), jnp.zeros(n),
                 jnp.zeros(n, jnp.int32), role, ids, jnp.int32(n))


def build_collect(network):
    """Keys are time-major (T,N,2); carry and key stream survive chunk splits."""
    def collect(params, epsilon, carry, keys):
        def tick(c, ks):
            split = jax.vmap(lambda k: jax.random.split(k, 3))(ks)
            ids = c.next_id + jnp.cumsum(c.pending.astype(jnp.int32)) - 1
            roles = ids % 2
            fresh, players = jax.vmap(reset)(split[:, 0], roles)
            def choose(new, old):
                shape = (len(c.pending),) + (1,) * (old.ndim - 1)
                return jnp.where(c.pending.reshape(shape), new, old)
            state = jax.tree.map(choose, fresh, c.state)
            learner = jnp.where(c.pending, players, c.learner)
            role = jnp.where(c.pending, roles, c.role)
            ep_ids = jnp.where(c.pending, ids, c.episode_id)
            returns = jnp.where(c.pending, 0., c.returns)
            lengths = jnp.where(c.pending, 0, c.length)
            board_lengths = jnp.where(c.pending, role, c.board_length)
            obs = jax.vmap(observe)(state, learner)
            mask = jax.vmap(legal_mask)(state)
            q = network.apply(params, obs)
            greedy = jnp.argmax(jnp.where(mask, q, -jnp.inf), -1)
            random = jax.vmap(lambda k, m: jax.random.categorical(k, jnp.where(m, 0., -jnp.inf)))(split[:, 1], mask)
            explore = jax.vmap(jax.random.uniform)(split[:, 2]) < epsilon
            actions = jnp.where(explore, random, greedy).astype(jnp.int32)
            result = jax.vmap(step)(state, learner, actions)
            returns += result.reward
            lengths += 1
            board_lengths += result.board_moves
            next_c = Carry(result.state, learner, role, result.done, returns, lengths,
                           board_lengths, ep_ids, c.next_id + c.pending.sum())
            row = dict(s=obs, a=actions, r=result.reward, s_next=jax.vmap(observe)(result.state, learner),
                       done=result.done, mask=mask, next_mask=jax.vmap(legal_mask)(result.state),
                       state=state, learner=learner, role=role, episode_id=ep_ids,
                       episode_return=returns, episode_length=lengths, board_length=board_lengths,
                       board_moves=result.board_moves + jnp.where(c.pending, role, 0),
                       started=c.pending)
            return next_c, row
        return jax.lax.scan(tick, carry, keys)
    return jax.jit(collect)
