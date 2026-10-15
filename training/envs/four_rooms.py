"""Sparse Four Rooms: SPS (1999), Fig. 2 geometry; study-specific reward/cap.

State encodes (elapsed time, cell), so the finite-horizon task is fully observed.
No replay, value update, or consequence scoring belongs here.
"""
import jax
import jax.numpy as jnp
import numpy as np

LAYOUT = (
    '#############',
    '#S....#.....#',
    '#.....#.....#',
    '#...........#',
    '#.....#.....#',
    '#.....#.....#',
    '##.####.....#',
    '#.....###.###',
    '#.....#.....#',
    '#.....#..G..#',
    '#...........#',
    '#.....#.....#',
    '#############',
)
DOORS = ((3, 6), (6, 2), (7, 9), (10, 6))
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class FourRoomsEnv:
    max_steps = 200
    n_actions = 4

    def __init__(self):
        self.cells = tuple((r, c) for r, row in enumerate(LAYOUT)
                           for c, tile in enumerate(row) if tile != '#')
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.n_positions = len(self.cells)
        self.n_states = (self.max_steps + 1) * self.n_positions
        self.start = self.index[(1, 1)]
        self.goal = self.index[(9, 9)]
        moves = np.empty((self.n_positions, 4), dtype=np.int32)
        for i, (r, c) in enumerate(self.cells):
            for a, (dr, dc) in enumerate(DIRECTIONS):
                moves[i, a] = self.index.get((r + dr, c + dc), i)
        self.moves = jnp.asarray(moves)
        self.outcome_probabilities = jnp.asarray([2/3, 1/9, 1/9, 1/9])

    def reset(self, key):
        del key
        s = jnp.int32(self.start)
        return self.get_obs(s), s

    def get_obs(self, state):
        pos, elapsed = state % self.n_positions, state // self.n_positions
        return jnp.concatenate((jax.nn.one_hot(pos, self.n_positions),
                                jnp.asarray(elapsed / self.max_steps)[..., None]), axis=-1)

    def legal_actions(self, state):
        del state
        return jnp.ones(4, dtype=jnp.bool_)

    def transition(self, state, executed_direction):
        """Exact deterministic branch, after the stochastic direction draw."""
        pos, elapsed = state % self.n_positions, state // self.n_positions
        ended = (pos == self.goal) | (elapsed >= self.max_steps)
        next_pos = jnp.where(ended, pos, self.moves[pos, executed_direction])
        time = jnp.where(ended, elapsed, elapsed + 1)
        ns = time * self.n_positions + next_pos
        reward = ((next_pos == self.goal) & ~ended).astype(jnp.float32)
        terminated = next_pos == self.goal
        truncated = (time >= self.max_steps) & ~terminated
        return self.get_obs(ns), ns, reward, terminated | truncated, {
            'terminated': terminated, 'truncated': truncated,
            # This study defines a finite-horizon task, not an external cutoff.
            'bootstrap': ~(terminated | truncated),
        }

    def step(self, key, state, action):
        # Slot zero is intended; the other three slots cover all other actions.
        slot = jax.random.choice(key, 4, p=self.outcome_probabilities)
        direction = (action + slot) % 4
        return self.transition(state, direction)
