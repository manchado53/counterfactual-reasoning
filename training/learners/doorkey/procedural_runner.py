"""Carry-safe DQN runner for the procedural MiniGrid DoorKey benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from counterfactual_rl.replay.buffers import PrioritizedReplayBuffer
from counterfactual_rl.replay.decay import DecayBuffer, blend_weight
from training.envs.doorkey_procedural import DoorKeyState, ProceduralDoorKey
from .adapter import ddqn_targets, weighted_scores


class Carry(NamedTuple):
    state: DoorKeyState
    obs: jax.Array
    pending_reset: jax.Array
    episode_return: jax.Array
    length: jax.Array


class QNetwork(nn.Module):
    hidden_dim: int = 128
    n_layers: int = 3

    @nn.compact
    def __call__(self, observation):
        x = observation.astype(jnp.float32)
        for _ in range(self.n_layers):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(7)(x)


def encode_observation(observation):
    """Flatten MiniGrid's symbolic image without exposing latent map state."""
    image = observation["image"].astype(jnp.float32)
    scales = jnp.array((10.0, 5.0, 2.0), dtype=jnp.float32)
    image = (image / scales).reshape(image.shape[:-3] + (7 * 7 * 3,))
    direction = jax.nn.one_hot(observation["direction"], 4, dtype=jnp.float32)
    return jnp.concatenate((image, direction), axis=-1)


def _empty_state(shape=()):
    zeros = jnp.zeros(shape, dtype=jnp.int32)
    return DoorKeyState(
        zeros, zeros, zeros, jnp.zeros(shape, dtype=jnp.bool_), zeros, zeros,
        zeros, zeros, zeros, zeros,
    )


class ProceduralAgent:
    """Three matched replay arms over fresh procedural 12x12 DoorKey maps."""

    def __init__(self, config):
        self.config = config
        self.arm = config["arm"]
        self.env = ProceduralDoorKey(config["size"], config["max_episode_steps"])
        self.eval_env = ProceduralDoorKey(config["size"], config["eval_max_steps"])
        self.net = QNetwork(config["hidden_dim"], config["n_layers"])

        root = jax.random.PRNGKey(config["seed"])
        self.collect_key, init_key = jax.random.split(root)
        self.params = self.net.init(init_key, jnp.zeros((151,), dtype=jnp.float32))
        self.target = jax.tree.map(jnp.copy, self.params)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config["gradient_clip"]),
            optax.adam(config["alpha"]),
        )
        self.opt_state = self.optimizer.init(self.params)
        replay = config["PER_parameters"]
        self.cce = self.arm == "cce_decay"
        if self.cce:
            self.buffer = DecayBuffer(
                capacity=config["buffer_capacity"],
                eps=replay["eps"],
                beta=replay["beta"],
                max_priority=replay["maximum_priority"],
                mu=config["mu"],
                priority_mixing=config["priority_mixing"],
                mu_c=config["mu_c"],
                mu_delta=config["mu_delta"],
                normalise_priorities=config["normalise_priorities"],
            )
        else:
            self.buffer = PrioritizedReplayBuffer(
                capacity=config["buffer_capacity"],
                eps=replay["eps"],
                beta=replay["beta"],
                max_priority=replay["maximum_priority"],
                uniform=self.arm == "uniform",
            )

        n_envs = config["n_envs"]
        self.carry = Carry(
            _empty_state((n_envs,)),
            jnp.zeros((n_envs, 151), dtype=jnp.float32),
            jnp.ones((n_envs,), dtype=jnp.bool_),
            jnp.zeros((n_envs,), dtype=jnp.float32),
            jnp.zeros((n_envs,), dtype=jnp.int32),
        )
        self.steps = self.episodes = self.updates = self.attempts = 0
        self.epsilon = float(config["epsilon_start"])
        self.eval_keys = jax.random.split(
            jax.random.PRNGKey(config["fixed_eval_seed"]), config["eval_episodes"]
        )
        identity = (
            f"procedural-doorkey-{config['size']}-symbolic7-"
            f"cap{config['eval_max_steps']}-greedy-lowest-index"
        ).encode()
        self.eval_bank_hash = hashlib.sha256(
            np.asarray(self.eval_keys).tobytes() + identity
        ).hexdigest()

        self._collect_fn = jax.jit(jax.vmap(self._collect_one, in_axes=(None, None, 0, 0)))
        self._update_fn = jax.jit(self._update)
        self._branch_fn = jax.jit(jax.vmap(self._branch_state, in_axes=(None, None, 0)))
        self._eval_fn = jax.jit(jax.vmap(self._eval_one, in_axes=(None, 0)))

    def _collect_one(self, params, epsilon, carry, keys):
        def one_step(current, key):
            reset_key, action_key, explore_key = jax.random.split(key, 3)
            reset_obs, reset_state = self.env.reset(reset_key)
            state = jax.tree.map(
                lambda fresh, old: jnp.where(current.pending_reset, fresh, old),
                reset_state,
                current.state,
            )
            obs = jnp.where(
                current.pending_reset,
                encode_observation(reset_obs),
                current.obs,
            )
            total = jnp.where(current.pending_reset, 0.0, current.episode_return)
            length = jnp.where(current.pending_reset, 0, current.length)
            greedy = jnp.argmax(self.net.apply(params, obs))
            explore = jax.random.uniform(explore_key) < epsilon
            action = jnp.where(explore, jax.random.randint(action_key, (), 0, 7), greedy)
            next_observation, next_state, reward, terminated, truncated, _ = self.env.step(
                None, state, action
            )
            next_obs = encode_observation(next_observation)
            total = total + reward
            length = length + 1
            ended = terminated | truncated
            row = dict(
                s=obs,
                a=action,
                r=reward,
                next_s=next_obs,
                done=terminated,
                truncated=truncated,
                ended=ended,
                episode_return=jnp.where(ended, total, 0.0),
                episode_length=jnp.where(ended, length, 0),
                explore=explore,
                latent_state=state,
            )
            return Carry(next_state, next_obs, ended, total, length), row

        return jax.lax.scan(one_step, carry, keys)

    def _update(self, params, target, opt_state, s, a, r, ns, done, weights):
        next_online = self.net.apply(params, ns)
        next_target = self.net.apply(target, ns)
        targets = jax.lax.stop_gradient(
            ddqn_targets(next_online, next_target, r, done, gamma=self.config["gamma"])
        )

        def loss_fn(values):
            q = self.net.apply(values, s)
            qa = jnp.take_along_axis(q, a[:, None], axis=1)[:, 0]
            td = targets - qa
            loss = jnp.mean(weights * td**2)
            stats = jnp.stack(
                (loss, jnp.mean(qa), jnp.max(jnp.abs(q)), jnp.mean(targets),
                 jnp.max(jnp.abs(targets)), jnp.mean(jnp.abs(td)), jnp.max(jnp.abs(td)))
            )
            return loss, (td, stats)

        (_, (td, stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, td, stats

    def _branch_state(self, params, target, state):
        actions = jnp.arange(7, dtype=jnp.int32)

        def branch(first_action):
            observation, next_state, reward, terminated, truncated, _ = self.env.step(
                None, state, first_action
            )
            feature = encode_observation(observation)

            def rollout(carry, _):
                current_state, current_feature, total, discount, term, trunc = carry
                active = ~(term | trunc)
                action = jnp.argmax(self.net.apply(params, current_feature))
                obs, proposed_state, r, new_term, new_trunc, _ = self.env.step(
                    None, current_state, action
                )
                proposed_feature = encode_observation(obs)
                selected_state = jax.tree.map(
                    lambda new, old: jnp.where(active, new, old), proposed_state, current_state
                )
                selected_feature = jnp.where(active, proposed_feature, current_feature)
                return (
                    selected_state,
                    selected_feature,
                    total + jnp.where(active, discount * r, 0.0),
                    jnp.where(active, discount * self.config["cf_gamma"], discount),
                    term | (active & new_term),
                    trunc | (active & new_trunc),
                ), None

            initial = (
                next_state,
                feature,
                reward,
                jnp.float32(self.config["cf_gamma"]),
                terminated,
                truncated,
            )
            final, _ = jax.lax.scan(
                rollout, initial, None, length=max(0, self.config["cf_horizon"] - 1)
            )
            _, feature, total, discount, term, _ = final
            bootstrap = jnp.max(self.net.apply(target, feature))
            return total + jnp.where(term | (not self.config["cf_bootstrap"]), 0.0, discount * bootstrap)

        return jax.vmap(branch)(actions)

    def _eval_one(self, params, key):
        observation, state = self.eval_env.reset(key)
        feature = encode_observation(observation)

        def one_step(carry, _):
            current_state, current_feature, ended, total, length = carry
            action = jnp.argmax(self.net.apply(params, current_feature))
            obs, proposed_state, reward, terminated, truncated, _ = self.eval_env.step(
                None, current_state, action
            )
            active = ~ended
            proposed_feature = encode_observation(obs)
            return (
                jax.tree.map(lambda new, old: jnp.where(active, new, old), proposed_state, current_state),
                jnp.where(active, proposed_feature, current_feature),
                ended | terminated | truncated,
                total + jnp.where(active, reward, 0.0),
                length + active,
            ), None

        final, _ = jax.lax.scan(
            one_step,
            (state, feature, jnp.bool_(False), jnp.float32(0.0), jnp.int32(0)),
            None,
            length=self.config["eval_max_steps"],
        )
        return final[3], final[4]

    @staticmethod
    def _flatten_latent(states):
        return jax.tree.map(lambda leaf: np.asarray(leaf).reshape((-1,) + leaf.shape[2:]), states)

    def _insert(self, rows):
        transitions = {
            "s": np.asarray(rows["s"]).reshape((-1, 151)),
            "a": np.asarray(rows["a"]).reshape(-1),
            "r": np.asarray(rows["r"]).reshape(-1),
            "s'": np.asarray(rows["next_s"]).reshape((-1, 151)),
            "done": np.asarray(rows["done"]).reshape(-1),
        }
        if not self.cce:
            self.buffer.add_batch(transitions)
            return
        states = self._flatten_latent(rows["latent_state"])
        slots = self.buffer.add_batch(transitions, jax_states=states)
        chunk = self.config["cf_score_chunk"]
        for lo in range(0, len(slots), chunk):
            hi = min(lo + chunk, len(slots))
            selected_states = jax.tree.map(lambda leaf: jnp.asarray(leaf[lo:hi]), states)
            returns = np.asarray(self._branch_fn(self.params, self.target, selected_states))
            probabilities = np.asarray(jax.nn.softmax(
                self.net.apply(self.params, jnp.asarray(transitions["s"][lo:hi])) /
                self.config["cf_action_prob_temperature"], axis=-1
            ))
            scores = weighted_scores(returns, probabilities, transitions["a"][lo:hi])
            if not np.isfinite(scores).all():
                raise FloatingPointError("Non-finite CCE score")
            self.buffer.update_consequence_scores(
                slots[lo:hi], scores, q_update_count=self.updates
            )

    def train_block(self):
        count = self.config["n_envs"] * self.config["collect_steps"]
        self.collect_key, block_key = jax.random.split(self.collect_key)
        keys = jax.random.split(block_key, count).reshape(
            self.config["n_envs"], self.config["collect_steps"], 2
        )
        started = time.monotonic()
        self.carry, rows = self._collect_fn(
            self.params, jnp.float32(self.epsilon), self.carry, keys
        )
        rows = jax.device_get(rows)
        collected = time.monotonic()
        self._insert(rows)
        scored = time.monotonic()

        ended = np.asarray(rows["ended"], dtype=bool)
        self.episodes += int(ended.sum())
        stats = []
        stride = self.config["n_steps_per_update"]
        target_attempts = self.config["target_update_freq"] // stride
        for attempt in range(self.steps // stride, (self.steps + count) // stride):
            self.attempts += 1
            if self.buffer.can_sample(self.config["batch_size"]):
                batch, indices, weights = self.buffer.sample(self.config["batch_size"])
                arrays = [jnp.asarray(np.stack([transition[key] for transition in batch]))
                          for key in ("s", "a", "r", "s'", "done")]
                self.params, self.opt_state, td, diagnostic = self._update_fn(
                    self.params, self.target, self.opt_state, *arrays,
                    jnp.asarray(weights, dtype=jnp.float32),
                )
                td, diagnostic = jax.device_get((td, diagnostic))
                if not np.isfinite(td).all() or not np.isfinite(diagnostic).all():
                    raise FloatingPointError("Non-finite learner update")
                self.buffer.update_priorities(indices, td)
                self.updates += 1
                stats.append(diagnostic)
            if attempt % target_attempts == 0:
                self.target = jax.tree.map(jnp.copy, self.params)

        self.steps += count
        self.epsilon = self.config["epsilon_start"] + (
            self.config["epsilon_end"] - self.config["epsilon_start"]
        ) * min(1.0, self.steps / self.config["epsilon_decay_steps"])
        if self.cce:
            self.buffer.blend = blend_weight(
                self.steps,
                enabled=self.config["decay_enabled"],
                start=self.config["decay_start"],
                end=self.config["decay_end"],
            )
        means = np.mean(stats, axis=0) if stats else np.full(7, np.nan)
        row = dict(zip(
            ("loss", "q_mean", "q_abs_max_mean", "target_mean", "target_abs_max_mean",
             "td_abs_mean", "td_abs_max_mean"),
            map(float, means),
        ))
        row.update(
            env_steps=self.steps,
            updates=self.updates,
            update_attempts=self.attempts,
            completed_episodes=self.episodes,
            epsilon=self.epsilon,
            explore_fraction=float(np.asarray(rows["explore"]).mean()),
            collect_seconds=collected - started,
            insert_score_seconds=scored - collected,
            update_seconds=time.monotonic() - scored,
        )
        if self.cce:
            row.update(
                scored_frac=float(np.mean(self.buffer.scored_at[:len(self.buffer)] >= 0)),
                decay_blend=self.buffer.blend,
            )
        return row

    def training_rng_state(self):
        return np.asarray(self.collect_key).tobytes(), pickle.dumps(np.random.get_state())

    def evaluation_identity(self):
        return dict(
            eval_bank_hash=self.eval_bank_hash,
            eval_unique_cases=self.config["eval_episodes"],
            eval_independent_trials=True,
        )

    def evaluate(self):
        before = self.training_rng_state()
        returns, lengths = jax.device_get(self._eval_fn(self.params, self.eval_keys))
        if self.training_rng_state() != before:
            raise RuntimeError("Evaluation changed training RNG")
        return dict(
            win_rate=float(np.mean(returns > 0)),
            avg_return=float(np.mean(returns)),
            avg_length=float(np.mean(lengths)),
            eval_episodes=self.config["eval_episodes"],
            unique_tasks=self.config["eval_episodes"],
            eval_bank_hash=self.eval_bank_hash,
        )

    def diagnostics(self):
        if self.cce:
            result = self.buffer.priority_diagnostics(now=self.updates)
        else:
            n = len(self.buffer)
            if self.arm == "uniform":
                probabilities = np.ones(n) / n
            else:
                probabilities = self.buffer.priorities[:n].copy()
                probabilities /= probabilities.sum()
            result = dict(
                n=n,
                ess_frac=float(1.0 / (n * np.sum(probabilities**2))),
                p_max=float(probabilities.max()),
            )
        result.update(self.buffer.reward_diagnostics())
        return result

    def save(self, path):
        payload = dict(
            config=self.config,
            params=jax.device_get(self.params),
            target=jax.device_get(self.target),
            opt_state=jax.device_get(self.opt_state),
            carry=jax.device_get(self.carry),
            collect_key=np.asarray(self.collect_key),
            numpy_rng=np.random.get_state(),
            steps=self.steps,
            episodes=self.episodes,
            updates=self.updates,
            attempts=self.attempts,
            epsilon=self.epsilon,
            eval_bank_hash=self.eval_bank_hash,
            resumable=False,
        )
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    def learn(self, verbose=False):
        del verbose
        np.random.seed(self.config["seed"])
        output = Path(os.environ["CCE_RUN_DIR"])
        next_eval = self.config["eval_interval_steps"]
        block = 0
        while self.steps < self.config["total_env_steps"]:
            block += 1
            row = dict(self.train_block(), block=block)
            with (output / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            if self.steps >= next_eval or self.steps >= self.config["total_env_steps"]:
                evaluation = dict(self.evaluate(), block=block, env_steps=self.steps)
                with (output / "eval.jsonl").open("a") as handle:
                    handle.write(json.dumps(evaluation, allow_nan=False) + "\n")
                with (output / "replay.jsonl").open("a") as handle:
                    handle.write(json.dumps(
                        dict(self.diagnostics(), block=block, env_steps=self.steps),
                        allow_nan=False,
                    ) + "\n")
                self.save(output / "last.pkl")
                while next_eval <= self.steps:
                    next_eval += self.config["eval_interval_steps"]
        return self
