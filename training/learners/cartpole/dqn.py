"""A small JAX Double DQN used to isolate replay sampling on CartPole."""

import hashlib
import json
import os
from pathlib import Path

from flax import linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from counterfactual_rl.learning.td import q_learning_targets
from .replay import CartPoleReplayBuffer


class QNetwork(nn.Module):
    hidden_size: int
    action_count: int

    @nn.compact
    def __call__(self, observations):
        values = nn.relu(nn.Dense(self.hidden_size)(observations))
        values = nn.relu(nn.Dense(self.hidden_size)(values))
        return nn.Dense(self.action_count)(values)


def weighted_huber_td_loss(q_values, actions, targets, weights):
    """Return the paper's importance-weighted smooth-L1 loss and signed TD."""
    chosen = jnp.take_along_axis(q_values, actions[:, None], axis=-1)[:, 0]
    td_errors = targets - chosen
    absolute = jnp.abs(td_errors)
    huber = jnp.where(absolute <= 1.0, 0.5 * td_errors**2, absolute - 0.5)
    return jnp.mean(weights * huber), td_errors


class CartPoleDDQN:
    def __init__(self, config):
        self.config = config
        self.output = Path(os.environ["CCE_RUN_DIR"])
        self.rng = np.random.default_rng(config["seed"])
        self.env = gym.make("CartPole-v1")
        observation, _ = self.env.reset(seed=config["seed"])
        self.observation = np.asarray(observation, dtype=np.float32)
        self.network = QNetwork(config["hidden_size"], self.env.action_space.n)
        key = jax.random.PRNGKey(config["seed"])
        self.params = self.network.init(key, jnp.zeros((1, self.observation.size), dtype=jnp.float32))
        self.target_params = jax.tree.map(lambda value: value.copy(), self.params)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(config["learning_rate"]),
        )
        self.optimizer_state = self.optimizer.init(self.params)
        self.buffer = CartPoleReplayBuffer(
            config["buffer_capacity"], self.observation.size, config["replay"], self.rng,
            paper_alpha=config["paper_per_alpha"], paper_epsilon=config["paper_per_epsilon"],
            repo_exponent=config["repo_per_exponent"], repo_epsilon=config["repo_per_epsilon"],
        )
        self.eval_seed_spec = {
            "base": int(config["seed"] * 1_000_000 + 700_000),
            "formula": "base + checkpoint_index * eval_episodes + episode_index",
            "episodes": config["eval_episodes"],
            "epsilon": config["eval_epsilon"],
        }
        self._train_step = jax.jit(self._make_train_step())

    def _make_train_step(self):
        network = self.network
        optimizer = self.optimizer
        gamma = self.config["gamma"]

        def train_step(params, target_params, optimizer_state, observations, actions,
                       rewards, next_observations, dones, weights):
            online_next_q = network.apply(params, next_observations)
            target_next_q = network.apply(target_params, next_observations)
            targets = q_learning_targets(
                rewards, dones, online_next_q, target_next_q, gamma, double_dqn=True,
            )

            def loss_fn(current_params):
                q_values = network.apply(current_params, observations)
                return weighted_huber_td_loss(q_values, actions, targets, weights)

            (loss, td_errors), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            return params, optimizer_state, loss, td_errors

        return train_step

    def _epsilon(self, step):
        decay_steps = max(1, int(self.config["epsilon_fraction"] * self.config["total_steps"]))
        fraction = min(step / decay_steps, 1.0)
        return self.config["epsilon_start"] + fraction * (
            self.config["epsilon_end"] - self.config["epsilon_start"]
        )

    def _beta(self, step):
        fraction = min(step / self.config["total_steps"], 1.0)
        return self.config["paper_per_beta_start"] + fraction * (
            self.config["paper_per_beta_end"] - self.config["paper_per_beta_start"]
        )

    def _action(self, observation, epsilon, rng):
        if rng.random() < epsilon:
            return int(rng.integers(self.env.action_space.n))
        q_values = self.network.apply(self.params, jnp.asarray(observation)[None, :])
        return int(np.asarray(jnp.argmax(q_values[0])))

    def _append_jsonl(self, name, row):
        with (self.output / name).open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _update(self, step):
        sample = self.buffer.sample(self.config["batch_size"], beta=self._beta(step))
        self.params, self.optimizer_state, loss, td_errors = self._train_step(
            self.params, self.target_params, self.optimizer_state,
            jnp.asarray(sample.observations), jnp.asarray(sample.actions),
            jnp.asarray(sample.rewards), jnp.asarray(sample.next_observations),
            jnp.asarray(sample.dones), jnp.asarray(sample.weights),
        )
        td_errors = np.asarray(td_errors)
        if self.config["replay"] != "uniform":
            self.buffer.update_priorities(sample.indices, td_errors)
        return float(loss), float(np.mean(np.abs(td_errors)))

    def evaluate(self, step=0):
        checkpoint_index = step // self.config["eval_frequency"]
        returns = []
        eval_env = gym.make("CartPole-v1")
        try:
            for episode in range(self.config["eval_episodes"]):
                seed = self.eval_seed_spec["base"] + checkpoint_index * self.config["eval_episodes"] + episode
                observation, _ = eval_env.reset(seed=seed)
                eval_rng = np.random.default_rng(seed + 500_000_000)
                total = 0.0
                done = False
                while not done:
                    action = self._action(np.asarray(observation, dtype=np.float32),
                                          self.config["eval_epsilon"], eval_rng)
                    observation, reward, terminated, truncated, _ = eval_env.step(action)
                    total += reward
                    done = terminated or truncated
                returns.append(total)
        finally:
            eval_env.close()
        mean_return = float(np.mean(returns))
        return {"step": step, "mean_return": mean_return, "returns": returns,
                "solved": mean_return >= self.config["solve_threshold"]}

    def evaluation_identity(self):
        encoded = json.dumps(self.eval_seed_spec, sort_keys=True).encode()
        return {"eval_bank_hash": hashlib.sha256(encoded).hexdigest(),
                "eval_seed_protocol": self.eval_seed_spec}

    def learn(self, verbose=False):
        solved_step = None
        update_count = 0
        last_loss = None
        last_td = None
        try:
            for step in range(1, self.config["total_steps"] + 1):
                action = self._action(self.observation, self._epsilon(step - 1), self.rng)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_observation = np.asarray(next_observation, dtype=np.float32)
                self.buffer.add(self.observation, action, reward, next_observation, done)
                self.observation = next_observation
                if done:
                    self.observation = np.asarray(self.env.reset()[0], dtype=np.float32)

                if step >= self.config["learning_starts"] and step % self.config["train_frequency"] == 0:
                    losses = []
                    td_values = []
                    for _ in range(self.config["gradient_steps"]):
                        last_loss, last_td = self._update(step)
                        losses.append(last_loss)
                        td_values.append(last_td)
                        update_count += 1
                    self._append_jsonl("training_metrics.jsonl", {
                        "step": step, "updates": update_count, "mean_loss": float(np.mean(losses)),
                        "mean_abs_td": float(np.mean(td_values)), "beta": self._beta(step),
                    })

                if step % self.config["target_update_frequency"] == 0:
                    self.target_params = jax.tree.map(lambda value: value.copy(), self.params)

                if step % self.config["eval_frequency"] == 0:
                    result = self.evaluate(step)
                    self._append_jsonl("evaluations.jsonl", result)
                    if result["solved"]:
                        solved_step = step
                        break
        finally:
            self.env.close()
        evaluations = [json.loads(line) for line in (self.output / "evaluations.jsonl").read_text().splitlines()]
        summary = {
            "replay": self.config["replay"], "seed": self.config["seed"],
            "steps_run": evaluations[-1]["step"], "steps_to_threshold": solved_step,
            "best_mean_return": max(row["mean_return"] for row in evaluations),
            "final_mean_return": evaluations[-1]["mean_return"], "updates": update_count,
            "last_loss": last_loss, "last_mean_abs_td": last_td,
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2))
