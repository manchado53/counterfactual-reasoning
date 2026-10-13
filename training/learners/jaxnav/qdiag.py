"""Q-value / loss diagnostics for the JaxNav trainers (``qdiag.jsonl``).

WHY THIS EXISTS
---------------
About half of every JaxNav run climbs to a 0.5-0.7 win rate and then falls back
under 0.2, in every arm (uniform 50%, PER 39%, CCE-mul 53%, CCE-add 50% of the
seeds that peaked >= 0.4). It is not the exploration schedule: epsilon reaches its
0.05 floor at episode 62.5k and the collapses land at 140k-210k. Two explanations
fit the eval logs equally well, and nothing we record can separate them:

  1. STARVATION. Reward is 1.0 at the goal and 0.0 for both crashing and timing
     out, and the buffer is a 100k sliding window (~900 episodes). Once the policy
     wobbles, fewer successes enter the buffer, the signal weakens, the policy
     degrades further, and the buffer ends up holding no reward at all.
  2. Q BLOW-UP. Ordinary overestimation through the bootstrap target.

``buffer_reward_frac`` is the direct test of (1). For (2) there is a sharp bound:
rewards are at most 1.0 and gamma is 0.99, so the true Q of any state-action pair
is at most 1.0. Any logged Q above 1.0 is MEASURED overestimation, not a judgement
call.

WHY THE LOSS IS DUPLICATED HERE
-------------------------------
``JaxNavDQN._build_jit_fns`` already computes every one of these quantities inside
its jitted ``update_step``, and returning them from there would be nearly free --
``_update`` already syncs the device each step to write replay priorities. It is
not done that way on purpose: adding outputs to a jitted function changes the HLO
module, and XLA is free to fuse a changed module differently, which can move the
last bits of a float. Runs made with logging on must be bit-identical to runs made
with it off, so ``update_step`` is left exactly as it was and the loss body is
repeated here instead.

The cost of that choice is drift: if the loss in ``dqn.py`` changes and this copy
does not, the diagnostics quietly describe a different objective.
``tests/test_qdiag_identity.py::test_qdiag_td_errors_match_update_step`` pins the
two together and fails loudly if they separate.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Order of the scalars packed by the jitted probe. Kept as one tuple so the
# accumulator, the field names and the probe cannot drift apart.
FIELDS = (
    'loss',
    'q_taken_mean',
    'q_taken_max',
    'q_max',
    'q_greedy_mean',
    'q_spread_mean',
    'td_abs_mean',
    'target_mean',
    'grad_norm',
    'batch_reward_frac',
)


def build_probe(network, gamma: float, double_dqn: bool):
    """Jitted forward+backward pass returning the FIELDS scalars for one batch.

    Mirrors ``JaxNavDQN._build_jit_fns.update_step``'s ``loss_fn`` exactly; see the
    module docstring for why it is a copy rather than a shared helper.
    """

    @jax.jit
    def probe(params, target_params, states, actions, rewards, next_states, dones, weights):
        def loss_fn(p):
            q = jax.vmap(network.apply, in_axes=(None, 0))(p, states)              # (B, A)
            q_taken = q[jnp.arange(q.shape[0]), actions]                           # (B,)
            next_q_target = jax.vmap(network.apply, in_axes=(None, 0))(target_params, next_states)
            if double_dqn:
                next_q_online = jax.vmap(network.apply, in_axes=(None, 0))(p, next_states)
                next_actions = jnp.argmax(jax.lax.stop_gradient(next_q_online), axis=-1)
                max_next_q = next_q_target[jnp.arange(next_q_target.shape[0]), next_actions]
            else:
                max_next_q = next_q_target.max(axis=-1)
            targets = rewards + gamma * max_next_q * (1.0 - dones)
            td_errors = targets - q_taken
            loss = jnp.mean(weights * td_errors ** 2)
            return loss, (q, q_taken, targets, td_errors)

        (loss, (q, q_taken, targets, td_errors)), grads = \
            jax.value_and_grad(loss_fn, has_aux=True)(params)

        q_greedy = jnp.max(q, axis=-1)
        q_worst = jnp.min(q, axis=-1)
        return jnp.stack([
            loss,
            jnp.mean(q_taken),
            jnp.max(q_taken),
            jnp.max(q),
            jnp.mean(q_greedy),
            jnp.mean(q_greedy - q_worst),
            jnp.mean(jnp.abs(td_errors)),
            jnp.mean(targets),
            # PRE-clip: the optimizer chain applies clip_by_global_norm(10.0), so
            # this is the raw norm, which is what shows an explosion.
            optax.global_norm(grads),
            jnp.mean((rewards > 0).astype(jnp.float32)),
        ]), td_errors

    return probe


class QDiagRecorder:
    """Accumulates per-gradient-step Q statistics between evals.

    Measuring every step would add a whole extra forward+backward pass to each
    update, so only every ``every``-th step is probed (default 10, ~+3% of
    gradient-step time; gradient steps are already under 1% of a CCE chunk).
    """

    def __init__(self, network, gamma: float, double_dqn: bool, every: int = 10):
        self.every = max(1, int(every))
        self._probe = build_probe(network, gamma, double_dqn)
        self._sums = np.zeros(len(FIELDS), dtype=np.float64)
        self._n_measured = 0
        self._n_seen = 0

    def maybe_record(self, params, target_params,
                     states, actions, rewards, next_states, dones, weights) -> None:
        """Call once per REAL gradient step, after the update.

        Must not be called for updates that returned early (an empty buffer), or
        the per-eval means get diluted by steps that never happened.
        """
        self._n_seen += 1
        if (self._n_seen - 1) % self.every:
            return
        stats, _ = self._probe(params, target_params,
                               states, actions, rewards, next_states, dones, weights)
        self._sums += np.asarray(stats, dtype=np.float64)
        self._n_measured += 1

    def drain(self) -> Optional[dict]:
        """Mean of everything since the last drain, then reset.

        Returns None when no step was measured, so the caller writes no line --
        the same "empty means silence" convention ``_log_priority_diagnostics``
        uses for ess.jsonl.
        """
        if self._n_measured == 0:
            self._n_seen = 0
            return None
        out = {name: float(v / self._n_measured) for name, v in zip(FIELDS, self._sums)}
        out['n_updates'] = int(self._n_seen)
        out['n_measured'] = int(self._n_measured)
        self._sums[:] = 0.0
        self._n_measured = 0
        self._n_seen = 0
        return out
