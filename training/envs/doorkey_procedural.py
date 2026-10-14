"""JAX DoorKey dynamics matching MiniGrid's procedural symbolic task."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DONE = range(7)
DOOR_OPEN, DOOR_CLOSED, DOOR_LOCKED = 0, 1, 2

UNSEEN = jnp.array((0, 0, 0), dtype=jnp.uint8)
EMPTY = jnp.array((1, 0, 0), dtype=jnp.uint8)
WALL = jnp.array((2, 5, 0), dtype=jnp.uint8)
KEY = jnp.array((5, 4, 0), dtype=jnp.uint8)
GOAL = jnp.array((8, 1, 0), dtype=jnp.uint8)
DIR_TO_VEC = jnp.array(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=jnp.int32)


class DoorKeyState(NamedTuple):
    x: jax.Array
    y: jax.Array
    direction: jax.Array
    carrying: jax.Array
    key_x: jax.Array
    key_y: jax.Array
    door_state: jax.Array
    split_x: jax.Array
    door_y: jax.Array
    step_count: jax.Array


class ProceduralDoorKey:
    """Pure JAX copy of the state-based MiniGrid DoorKey task."""

    view_size = 7
    n_actions = 7

    def __init__(self, size: int = 12, max_steps: int | None = None):
        if size < 5:
            raise ValueError("DoorKey size must be at least 5")
        self.size = int(size)
        self.max_steps = int(max_steps if max_steps is not None else 10 * size**2)

    def _tile(self, state: DoorKeyState, x, y):
        outside = (x < 0) | (x >= self.size) | (y < 0) | (y >= self.size)
        boundary = (x == 0) | (x == self.size - 1) | (y == 0) | (y == self.size - 1)
        at_door = (x == state.split_x) & (y == state.door_y)
        split_wall = (x == state.split_x) & ~at_door
        at_key = (x == state.key_x) & (y == state.key_y) & ~jnp.asarray(state.carrying)
        at_goal = (x == self.size - 2) & (y == self.size - 2)
        door = jnp.array((4, 4, state.door_state), dtype=jnp.uint8)
        tile = EMPTY
        tile = jnp.where(at_goal, GOAL, tile)
        tile = jnp.where(at_key, KEY, tile)
        tile = jnp.where(at_door, door, tile)
        tile = jnp.where(outside | boundary | split_wall, WALL, tile)
        return tile

    def observe(self, state: DoorKeyState):
        forward = DIR_TO_VEC[state.direction]
        right = jnp.array((-forward[1], forward[0]), dtype=jnp.int32)
        position = jnp.array((state.x, state.y), dtype=jnp.int32)
        top_left = position + forward * (self.view_size - 1) - right * (self.view_size // 2)
        coords = jnp.stack(jnp.meshgrid(
            jnp.arange(self.view_size, dtype=jnp.int32),
            jnp.arange(self.view_size, dtype=jnp.int32),
            indexing="ij",
        ), axis=-1)
        world = top_left - coords[..., 1, None] * forward + coords[..., 0, None] * right
        grid = jax.vmap(jax.vmap(lambda xy: self._tile(state, xy[0], xy[1])))(world)

        mask = jnp.zeros((self.view_size, self.view_size), dtype=jnp.bool_)
        mask = mask.at[self.view_size // 2, self.view_size - 1].set(True)
        for view_y in reversed(range(self.view_size)):
            for view_x in range(self.view_size - 1):
                cell = grid[view_x, view_y]
                see_behind = (cell[0] != 2) & ~((cell[0] == 4) & (cell[2] != DOOR_OPEN))
                visible = mask[view_x, view_y] & see_behind
                mask = mask.at[view_x + 1, view_y].set(mask[view_x + 1, view_y] | visible)
                if view_y > 0:
                    mask = mask.at[view_x + 1, view_y - 1].set(mask[view_x + 1, view_y - 1] | visible)
                    mask = mask.at[view_x, view_y - 1].set(mask[view_x, view_y - 1] | visible)
            for view_x in reversed(range(1, self.view_size)):
                cell = grid[view_x, view_y]
                see_behind = (cell[0] != 2) & ~((cell[0] == 4) & (cell[2] != DOOR_OPEN))
                visible = mask[view_x, view_y] & see_behind
                mask = mask.at[view_x - 1, view_y].set(mask[view_x - 1, view_y] | visible)
                if view_y > 0:
                    mask = mask.at[view_x - 1, view_y - 1].set(mask[view_x - 1, view_y - 1] | visible)
                    mask = mask.at[view_x, view_y - 1].set(mask[view_x, view_y - 1] | visible)

        image = jnp.where(mask[..., None], grid, UNSEEN)
        carried = jnp.where(state.carrying, KEY, EMPTY)
        image = image.at[self.view_size // 2, self.view_size - 1].set(carried)
        return {"image": image, "direction": state.direction}

    def observe_full(self, state: DoorKeyState):
        """Match MiniGrid's FullyObsWrapper symbolic grid encoding."""
        coordinates = jnp.stack(
            jnp.meshgrid(
                jnp.arange(self.size, dtype=jnp.int32),
                jnp.arange(self.size, dtype=jnp.int32),
                indexing="ij",
            ),
            axis=-1,
        )
        image = jax.vmap(jax.vmap(lambda xy: self._tile(state, xy[0], xy[1])))(
            coordinates
        )
        agent = jnp.array((10, 0, state.direction), dtype=jnp.uint8)
        image = image.at[state.x, state.y].set(agent)
        return {"image": image, "direction": state.direction}

    def reset(self, key):
        split_key, agent_key, direction_key, door_key, key_key = jax.random.split(key, 5)
        split_x = jax.random.randint(split_key, (), 2, self.size - 2)
        width = split_x - 1
        height = self.size - 2
        count = width * height
        agent_index = jax.random.randint(agent_key, (), 0, count)
        x = 1 + agent_index % width
        y = 1 + agent_index // width
        direction = jax.random.randint(direction_key, (), 0, 4)
        door_y = jax.random.randint(door_key, (), 1, self.size - 2)
        key_slot = jax.random.randint(key_key, (), 0, count - 1)
        key_index = key_slot + (key_slot >= agent_index)
        key_x = 1 + key_index % width
        key_y = 1 + key_index // width
        state = DoorKeyState(
            x=x,
            y=y,
            direction=direction,
            carrying=jnp.bool_(False),
            key_x=key_x,
            key_y=key_y,
            door_state=jnp.int32(DOOR_LOCKED),
            split_x=split_x,
            door_y=door_y,
            step_count=jnp.int32(0),
        )
        return self.observe(state), state

    def step(self, key, state: DoorKeyState, action):
        del key  # Official DoorKey transitions are deterministic after reset.
        action = jnp.asarray(action, dtype=jnp.int32)
        direction = jnp.asarray(state.direction, dtype=jnp.int32)
        carrying_now = jnp.asarray(state.carrying, dtype=jnp.bool_)
        forward = DIR_TO_VEC[direction]
        front_x, front_y = state.x + forward[0], state.y + forward[1]
        front = self._tile(state, front_x, front_y)
        front_type = front[0]

        turn_left = action == LEFT
        turn_right = action == RIGHT
        direction = jnp.where(turn_left, (direction - 1) % 4, direction)
        direction = jnp.where(turn_right, (direction + 1) % 4, direction)

        is_door = front_type == 4
        can_move = (front_type == 1) | (front_type == 8) | (
            is_door & (state.door_state == DOOR_OPEN)
        )
        moves = (action == FORWARD) & can_move
        x = jnp.where(moves, front_x, state.x)
        y = jnp.where(moves, front_y, state.y)

        picks_up = (action == PICKUP) & (front_type == 5) & ~carrying_now
        carrying = carrying_now | picks_up
        key_x = jnp.where(picks_up, -1, state.key_x)
        key_y = jnp.where(picks_up, -1, state.key_y)

        drops = (action == DROP) & carrying_now & (front_type == 1)
        carrying = jnp.where(drops, False, carrying)
        key_x = jnp.where(drops, front_x, key_x)
        key_y = jnp.where(drops, front_y, key_y)

        toggles = (action == TOGGLE) & is_door
        unlocks = toggles & (state.door_state == DOOR_LOCKED) & carrying_now
        closes = toggles & (state.door_state == DOOR_OPEN)
        opens = toggles & (state.door_state == DOOR_CLOSED)
        door_state = jnp.where(unlocks | opens, DOOR_OPEN, state.door_state)
        door_state = jnp.where(closes, DOOR_CLOSED, door_state)

        step_count = jnp.asarray(state.step_count, dtype=jnp.int32) + 1
        reached_goal = (action == FORWARD) & (front_type == 8)
        reward = jnp.where(
            reached_goal,
            1.0 - 0.9 * step_count.astype(jnp.float32) / self.max_steps,
            0.0,
        )
        terminated = reached_goal
        truncated = step_count >= self.max_steps
        next_state = DoorKeyState(
            x=x,
            y=y,
            direction=direction,
            carrying=carrying,
            key_x=key_x,
            key_y=key_y,
            door_state=door_state,
            split_x=state.split_x,
            door_y=state.door_y,
            step_count=step_count,
        )
        return self.observe(next_state), next_state, reward, terminated, truncated, {}
