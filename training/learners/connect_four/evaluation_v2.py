"""PGX evaluation from fixed, distinct openings with independent random keys."""
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
from training.envs import connect_four_transfer as task
from .bank_v2 import load


def foe(state, policy, key):
    obs = state.observation
    rows = jnp.maximum(0, 5 - obs.any(-1).sum(0))
    def wins(col, channel):
        pieces = obs[:, :, channel].at[rows[col], col].set(True)
        return pieces.reshape(-1)[task.WINDOWS].all(-1).any()
    win = jax.vmap(lambda c: wins(c, 0))(jnp.arange(7))
    block = jax.vmap(lambda c: wins(c, 1))(jnp.arange(7))
    ranks = jnp.where((policy >= 2) & win, 2, jnp.where((policy >= 3) & block, 1, 0))
    order = jnp.where(policy == 4, jnp.arange(7), task.ORDER)
    greedy = order[jnp.argmax(jnp.where(task.legal_mask(state), ranks, -1)[order])]
    random = jax.random.categorical(key, jnp.where(task.legal_mask(state), 0., -jnp.inf))
    return jnp.where(policy == 0, random, greedy).astype(jnp.int32)


class Evaluator:
    def __init__(self, network, expected_hash):
        self.bank, self.hash = load(expected_hash)
        self.expected = expected_hash
        cases = self.bank['cases']
        self.roles = np.array([c['role'] for c in cases], np.int32)
        self.policies = np.array([c['policy'] for c in cases], np.int32)
        self.seeds = np.array([c['seed'] for c in cases], np.uint32)
        openings = np.full((1000, 12), -1, np.int32)
        for i, c in enumerate(cases):
            openings[i, :len(c['opening'])] = c['opening']
        def start(seed, role, moves):
            state = task.ENV.init(jax.random.PRNGKey(seed))
            learner = jnp.where(role == 0, state.current_player, 1-state.current_player)
            def tick(s, a):
                return jax.lax.cond(a >= 0, lambda x: task.ENV.step(x, a), lambda x: x, s), None
            state, _ = jax.lax.scan(tick, state, moves)
            return state, learner
        self.states, self.players = jax.jit(jax.vmap(start))(
            jnp.asarray(self.seeds), jnp.asarray(self.roles), jnp.asarray(openings))
        assert not np.asarray(self.states.terminated).any()
        np.testing.assert_array_equal(self.states.current_player, self.players)
        # Verify PGX sees the same boards as the portable bank, from learner view.
        observations = np.asarray(jax.vmap(task.observe)(self.states, self.players)).reshape(1000, 6, 7, 2)
        for i, c in enumerate(cases):
            b = np.asarray(c['board'])
            np.testing.assert_array_equal(observations[i, :, :, 0], b == 1+c['role'])
            np.testing.assert_array_equal(observations[i, :, :, 1], b == 2-c['role'])
        def one(params, state, learner, policy, seed):
            def tick(carry, ply):
                s, reward = carry
                def live(s):
                    q = network.apply(params, task.observe(s, learner))
                    greedy = jnp.argmax(jnp.where(task.legal_mask(s), q, -jnp.inf))
                    opponent = foe(s, policy, jax.random.fold_in(jax.random.PRNGKey(seed), ply))
                    action = jnp.where(s.current_player == learner, greedy, opponent).astype(jnp.int32)
                    legal = task.legal_mask(s)[action]
                    nxt = task.ENV.step(s, action)
                    return nxt, nxt.rewards[learner], action, legal
                nxt, r, a, legal = jax.lax.cond(s.terminated | s.truncated,
                    lambda x: (x, jnp.float32(0), jnp.int32(-1), jnp.bool_(True)), live, s)
                return (nxt, reward+r), (a, legal)
            (end, reward), (actions, legal) = jax.lax.scan(tick, (state, jnp.float32(0)), jnp.arange(42))
            return reward, end.terminated | end.truncated, actions, legal
        self.run = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0)))

    def evaluate(self, params):
        load(self.expected)  # Check persisted bytes every time, not just a cached label.
        parts = []
        for lo in range(0, 1000, 100):
            states = jax.tree.map(lambda x: x[lo:lo+100], self.states)
            parts.append(tuple(map(np.asarray, self.run(params, states,
                self.players[lo:lo+100], self.policies[lo:lo+100], self.seeds[lo:lo+100]))))
        rewards, done, actions, legal = [np.concatenate([p[i] for p in parts]) for i in range(4)]
        assert done.all() and legal.all()
        def stats(pick):
            r = rewards[pick]
            return dict(cases=len(r), win_rate=float(np.mean(r == 1)),
                        draw_rate=float(np.mean(r == 0)), loss_rate=float(np.mean(r == -1)))
        paths = set()
        for c, row in zip(self.bank['cases'], actions):
            paths.add(tuple(c['opening']) + tuple(int(a) for a in row if a >= 0))
        return dict(**stats(np.ones(1000, bool)), eval_bank_hash=self.hash,
            unique_start_positions=1000, unique_game_paths=len(paths), independent_training_seeds=1,
            by_role={str(r): stats(self.roles == r) for r in (0, 1)},
            by_opponent={name: stats(self.policies == p) for p, name in enumerate(self.bank['policies'])},
            by_role_opponent={f'{r}:{p}': stats((self.roles == r) & (self.policies == p))
                              for r in (0, 1) for p in range(5)},
            outcome_hash=hashlib.sha256(rewards.tobytes()+actions.tobytes()).hexdigest())
