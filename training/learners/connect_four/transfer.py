"""Approved fixed-JaxNav recipe on learner-turn Connect Four.

No legacy trainer imports. Shared replay implementations are pinned by main.
"""
import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from training.envs import connect_four_transfer as task
from counterfactual_rl.replay.buffers import PrioritizedReplayBuffer
from ._source_shared.consequence_buffers import ConsequenceReplayBuffer

ARMS = ("uniform", "per", "cce_mul", "cce_add", "cce_only")
RECIPE = dict(gamma=.99, alpha=.0003, hidden_dim=128, n_layers=3, batch_size=128,
              buffer_capacity=100000, n_steps_per_update=16, target_update_freq=2000,
              n_envs=256, collect_steps=32, epsilon_decay_episodes=62500,
              epsilon_start=1., epsilon_end=.05, cf_horizon=60, cf_n_rollouts=1,
              cf_gamma=.99, cf_rollout_temperature=0., cf_action_prob_temperature=.5,
              cf_score_chunk=1280, double_dqn=True, cf_bootstrap=True,
              explore_hold_steps=1, score_on_insert=True, refresh=False,
              consequence_metric="wasserstein", consequence_aggregation="weighted_mean",
              normalise_priorities="rank", gradient_clip=10., fixed_eval_seed=20260923,
              eval_episodes=1000, blocks=2000, eval_every_blocks=50,
              opponent="win-block-center-v1", opponent_ties=[3, 2, 4, 1, 5, 0, 6],
              PER_parameters=dict(eps=.01, beta=.25, maximum_priority=1.))


class Network(nn.Module):
    @nn.compact
    def __call__(self, obs):
        x = obs.astype(jnp.float32)
        for _ in range(3):
            x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(7)(x)


def targets(reward, done, online, target, mask, gamma=.99):
    """Terminal branches never multiply zero by a masked infinity."""
    action = jnp.argmax(jnp.where(mask, online, -jnp.inf), axis=-1)
    value = jnp.take_along_axis(target, action[..., None], -1)[..., 0]
    return reward + gamma * jnp.where(done, 0., value)


def consequence(returns, q, mask, actions):
    """Pinned N=1 score arithmetic, with illegal actions excluded.

    Softmax is float32 over ALL legal moves, including the taken move. The
    host casts returns/probabilities to float64 before subtraction and sums
    alternatives in ascending column order. Underflow uses the source's
    unweighted-mean fallback, not a new softmax over alternatives.
    """
    probs = np.asarray(jax.nn.softmax(jnp.where(mask, q.astype(jnp.float32) / .5, -jnp.inf), -1))
    values = np.asarray(returns).astype(np.float64)
    actions = np.asarray(actions)
    masks = np.asarray(mask)
    n = len(actions)
    taken = values[np.arange(n), actions]
    numerator, denominator = np.zeros(n), np.zeros(n)
    for action in range(7):
        alt = masks[:, action] & (actions != action)
        p = probs[:, action].astype(np.float64)
        numerator = np.where(alt, numerator + p * np.abs(taken - values[:, action]), numerator)
        denominator = np.where(alt, denominator + p, denominator)
    scores = np.zeros(n)
    ok = denominator > 0
    scores[ok] = numerator[ok] / denominator[ok]
    for i in np.flatnonzero(~ok):
        alternatives = [a for a in range(7) if masks[i, a] and a != actions[i]]
        if alternatives:
            scores[i] = float(np.mean([abs(taken[i] - values[i, a]) for a in alternatives]))
    return scores


def build_rollouts(network, horizon=60):
    def rollout(params, target_params, state, learner, action, key):
        del key  # Source scoring key still consumed by caller; dynamics are fixed.
        valid = task.legal_mask(state)[action]
        def legal_branch(_):
            first = task.step(state, learner, action)
            def tick(c, _):
                s, total, discount, done = c
                q = network.apply(params, task.observe(s, learner))
                a = jnp.argmax(jnp.where(task.legal_mask(s), q, -jnp.inf))
                out = task.step(s, learner, a)
                return (out.state, total + discount * out.reward,
                        discount * .99, done | out.done), None
            (last, total, discount, done), _ = jax.lax.scan(
                tick, (first.state, first.reward, jnp.float32(.99), first.done),
                None, length=horizon - 1)
            tail_q = network.apply(target_params, task.observe(last, learner))
            tail = jnp.max(jnp.where(task.legal_mask(last), tail_q, -jnp.inf))
            return total + discount * jnp.where(done, 0., tail), done
        return jax.lax.cond(valid, legal_branch,
                            lambda _: (jnp.float32(0.), jnp.bool_(True)), operand=None)
    over_actions = jax.vmap(rollout, in_axes=(None, None, None, None, 0, 0))
    return jax.jit(jax.vmap(over_actions, in_axes=(None, None, 0, 0, None, 0)))


def build_update(network, optimizer):
    @jax.jit
    def update(params, target_params, opt_state, batch, weights):
        q_next = network.apply(params, batch['s_next'])
        target_next = network.apply(target_params, batch['s_next'])
        y = jax.lax.stop_gradient(targets(batch['r'], batch['done'], q_next,
                                          target_next, batch['next_mask']))
        def loss_fn(p):
            q = network.apply(p, batch['s'])
            chosen = jnp.take_along_axis(q, batch['a'][:, None], -1)[:, 0]
            td = y - chosen
            return jnp.mean(weights * td ** 2), (td, chosen, q)
        (loss, (td, chosen, q)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        diagnostics = jnp.array([loss, jnp.mean(chosen), jnp.max(jnp.abs(q)),
                                 jnp.mean(y), jnp.max(jnp.abs(y)),
                                 jnp.mean(jnp.abs(td)), jnp.max(jnp.abs(td))])
        return params, opt_state, td, diagnostics
    return update


class Trainer:
    def __init__(self, arm, seed, *, n_envs=256, capacity=100000):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.seed, self.n_envs = arm, seed, n_envs
        self.cce = arm.startswith('cce')
        self.network = Network()
        self.key = jax.random.PRNGKey(seed)
        self.key, init = jax.random.split(self.key)
        self.params = self.network.init(init, jnp.zeros(84))
        self.target_params = jax.tree.map(jnp.copy, self.params)
        self.optimizer = optax.chain(optax.clip_by_global_norm(10.), optax.adam(.0003))
        self.opt_state = self.optimizer.init(self.params)
        kwargs = dict(capacity=capacity, eps=.01, beta=.25, max_priority=1.)
        if self.cce:
            self.buffer = ConsequenceReplayBuffer(**kwargs,
                mu=1. if arm == 'cce_only' else .5,
                priority_mixing='multiplicative' if arm == 'cce_mul' else 'additive',
                mu_c=1., mu_delta=1., normalise_priorities='rank')
        else:
            self.buffer = PrioritizedReplayBuffer(**kwargs, uniform=arm == 'uniform')
        self.collect_fn = task.build_collect(self.network)
        self.rollout_fn = build_rollouts(self.network)
        self.update_fn = build_update(self.network, self.optimizer)
        self.key, init = jax.random.split(self.key)
        self.carry = task.initial_carry(jax.random.split(init, n_envs))
        self.steps = self.updates = self.attempts = self.episodes = 0
        self.board_moves = int(np.asarray(self.carry.board_length).sum())
        self.started_roles = np.bincount(np.asarray(self.carry.role), minlength=2)
        self.completed_roles = np.zeros(2, dtype=np.int64)
        self.epsilon = 1.
        self.last_diag = None
        self.last_episode_lengths = []
        self.eval_fn = self._build_eval()

    def _build_eval(self):
        def one(params, role, key):
            state, player = task.reset(key, role)
            def tick(c, _):
                s, reward, length, moves, done = c
                q = self.network.apply(params, task.observe(s, player))
                a = jnp.argmax(jnp.where(task.legal_mask(s), q, -jnp.inf))
                out = task.step(s, player, a)
                return (out.state, reward + out.reward, length + (~done),
                        moves + out.board_moves, done | out.done), None
            end, _ = jax.lax.scan(tick, (state, jnp.float32(0), jnp.int32(0),
                                        role, jnp.bool_(False)), None, length=21)
            return end[1:]
        return jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))

    def evaluate(self):
        # Two paths, each repeated 500 times. No train key or NumPy RNG use.
        roles = jnp.arange(1000, dtype=jnp.int32) % 2
        keys = jax.random.split(jax.random.PRNGKey(20260923), 1000)
        rewards, lengths, moves, done = map(np.asarray, self.eval_fn(self.params, roles, keys))
        assert done.all()
        out = dict(eval_episodes=1000, unique_role_cases=2,
                   win_rate=float(np.mean(rewards == 1)), draw_rate=float(np.mean(rewards == 0)),
                   loss_rate=float(np.mean(rewards == -1)), avg_length=float(lengths.mean()),
                   avg_board_moves=float(moves.mean()), independent_trials=False)
        for role in (0, 1):
            pick = np.asarray(roles) == role
            assert len(set(zip(rewards[pick], lengths[pick], moves[pick]))) == 1
            out[f'role_{role}'] = dict(win_rate=float(np.mean(rewards[pick] == 1)),
                draw_rate=float(np.mean(rewards[pick] == 0)), loss_rate=float(np.mean(rewards[pick] == -1)),
                length=int(lengths[pick][0]), board_moves=int(moves[pick][0]), repeats=500)
        return out

    def collect(self, steps=32):
        self.key, key = jax.random.split(self.key)
        keys = jax.random.split(key, steps * self.n_envs).reshape(steps, self.n_envs, 2)
        self.carry, rows = self.collect_fn(self.params, jnp.float32(self.epsilon), self.carry, keys)
        jax.block_until_ready(rows['r'])
        # Keep immutable pre-action PGX states for all new CCE branches.
        states = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), rows.pop('state'))
        flat = {k: np.asarray(v).reshape((-1,) + v.shape[2:]) for k, v in rows.items()}
        assert np.all(flat['mask'].any(-1))
        assert not np.asarray(states.terminated).any()
        assert np.all(np.asarray(states.current_player) == flat['learner'])
        ended = flat['done']
        self.episodes += int(ended.sum())
        self.completed_roles += np.bincount(flat['role'][ended], minlength=2)
        self.started_roles += np.bincount(flat['role'][flat['started']], minlength=2)
        self.board_moves += int(flat['board_moves'].sum())
        self.last_episode_lengths = flat['episode_length'][ended].tolist()
        fields = ('s', 'a', 'r', 's_next', 'done', 'mask', 'next_mask')
        data = {k: flat[k] for k in fields}
        slots = self.buffer.add_batch(data)
        if self.cce:
            # No refresh. Score every arrival, before the first replay draw.
            for lo in range(0, len(flat['a']), 1280):
                hi = min(lo + 1280, len(flat['a']))
                sub = jax.tree.map(lambda x: x[lo:hi], states)
                self.key, score_key = jax.random.split(self.key)
                keys = jax.random.split(score_key, (hi - lo) * 7).reshape(hi - lo, 7, 2)
                returns, terminal = self.rollout_fn(self.params, self.target_params, sub,
                        jnp.asarray(flat['learner'][lo:hi]), jnp.arange(7), keys)
                assert np.asarray(terminal).all(), 'H60 Connect Four must have no live tail'
                q = self.network.apply(self.params, jnp.asarray(flat['s'][lo:hi]))
                scores = np.asarray(consequence(returns, q, jnp.asarray(flat['mask'][lo:hi]),
                                                jnp.asarray(flat['a'][lo:hi])))
                assert np.isfinite(scores).all()
                self.buffer.update_consequence_scores(slots[lo:hi], scores, self.updates)
            assert np.all(self.buffer.scored_at[:len(self.buffer)] >= 0)
        self.steps += len(flat['a'])
        self.epsilon = 1. + (.05 - 1.) * min(1., self.episodes / 62500.)
        return flat

    def learn_block(self, steps=32):
        rows = self.collect(steps)
        stop = self.steps // 16
        for index in range(self.attempts, stop):
            if self.buffer.can_sample(128):
                transitions, indices, weights = self.buffer.sample(128)
                batch = {k: jnp.asarray(np.stack([t[k] for t in transitions]))
                         for k in ('s', 'a', 'r', 's_next', 'done', 'next_mask')}
                self.params, self.opt_state, td, diag = self.update_fn(
                    self.params, self.target_params, self.opt_state, batch, jnp.asarray(weights))
                td, diag = np.asarray(td), np.asarray(diag)
                assert np.isfinite(diag).all() and np.isfinite(td).all()
                self.buffer.update_priorities(indices, td)
                self.last_diag = dict(zip(('loss', 'q_mean', 'q_abs_max', 'target_mean',
                                          'target_abs_max', 'td_abs_mean', 'td_abs_max'), map(float, diag)))
                self.updates += 1
            # Source phase: AFTER zero-index attempts 0,125,250,...
            if index % 125 == 0:
                self.target_params = jax.tree.map(jnp.copy, self.params)
        self.attempts = stop
        return rows

    def diagnostics(self):
        if self.cce:
            probs = self.buffer._compute_priorities()
            stamps = self.buffer.scored_at[:len(self.buffer)]
            coverage = float(np.mean(stamps >= 0))
            score_age = self.updates - stamps
        else:
            probs = self.buffer.priorities[:len(self.buffer)].copy()
            if self.arm == 'uniform':
                probs[:] = 1.
            probs /= probs.sum()
            coverage, score_age = None, np.array([0])
        return dict(**(self.last_diag or {}), steps=self.steps, board_moves=self.board_moves,
            episodes=self.episodes, update_attempts=self.attempts, actual_updates=self.updates,
            epsilon=self.epsilon, buffer_size=len(self.buffer), score_coverage=coverage,
            score_age_mean=float(score_age.mean()) if self.cce else None,
            score_age_max=int(score_age.max()) if self.cce else None,
            replay_ess_fraction=float(1. / (len(probs) * np.sum(probs ** 2))),
            replay_max_probability=float(probs.max()), started_roles=self.started_roles.tolist(),
            completed_roles=self.completed_roles.tolist(),
            last_mean_episode_length=float(np.mean(self.last_episode_lengths)) if self.last_episode_lengths else None)

    def save(self, path):
        # Source-style evaluation checkpoint. No replay dump or resume claim.
        names = ('arm', 'seed', 'n_envs', 'params', 'target_params', 'opt_state',
                 'steps', 'updates', 'attempts', 'episodes', 'epsilon')
        record = {name: getattr(self, name) for name in names}
        record['resumable'] = False
        with open(path, 'wb') as handle:
            pickle.dump(jax.device_get(record), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as handle:
            record = pickle.load(handle)
        obj = cls(record['arm'], record['seed'], n_envs=record['n_envs'])
        obj.__dict__.update(record)
        return obj
