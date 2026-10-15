"""
Consequence-weighted DQN (Algorithm 2) for the neural CVRP.

Mirror of ``agents/jax_nav/consequence_dqn.py``, with three CVRP-specific changes.

1. THE ROLLOUT IS THE SHARED ONE. This agent calls
   ``agents/shared/cf_rollout.build_rollout_fn`` instead of carrying a sixth
   private copy (repo issue #12). ``tests/test_cf_rollout_parity.py`` pins it to
   jax_nav's copy bit-for-bit.

2. ONLY LEGAL ACTIONS ARE COUNTERFACTUALS. The rollout is vmapped over a fixed
   action array so it can be jitted, but an illegal first action (a customer
   already served, or one that does not fit on the truck) produces a corrupt
   state and a meaningless return, and Wasserstein would happily let that
   garbage dominate the score. Illegal actions are dropped when the per-action
   return distributions are assembled. The counterfactual question is "among the
   choices actually available, how much did this one matter?" -- an unavailable
   choice is not an alternative.

3. WASSERSTEIN, NOT TOTAL VARIATION. Their CVRP is deterministic with a
   continuous distance reward, which is exactly where TV degenerates:
   ``metrics.py`` returns a hardcoded 1.0 for any two distinct constants, so
   every state ties at the ceiling and the correlation with the oracle is
   undefined. On a deterministic env Wasserstein reduces to |R_a* - R_a|, which
   is the oracle's own definition estimated by rollout.

Variants (identical knobs to every other env):
  algorithm='consequence-dqn', mu=1.0                            -> CCE-only
  algorithm='consequence-dqn', priority_mixing='additive'        -> CCE + TD (Eq. 4)
  algorithm='consequence-dqn', priority_mixing='multiplicative'  -> CCE + TD (Eq. 5)
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .dqn import CVRPNeuralDQN, MASK_VALUE
from counterfactual_rl.replay.consequence_buffers import ConsequenceReplayBuffer
from ._historical.cf_rollout import build_rollout_fn
from counterfactual_rl.scoring.metrics import compute_consequence_metric


class CVRPNeuralConsequenceDQN(CVRPNeuralDQN):
    """Consequence-weighted DQN for the neural CVRP (Algorithm 2)."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        per = self.config.get('PER_parameters', {})
        self.buffer = ConsequenceReplayBuffer(
            capacity=self.config['buffer_capacity'],
            eps=per.get('eps', 0.01),
            beta=per.get('beta', 0.4),
            max_priority=per.get('maximum_priority', 1.0),
            mu=self.config.get('mu', 0.5),
            priority_mixing=self.config.get('priority_mixing', 'multiplicative'),
            mu_c=self.config.get('mu_c', 1.0),
            mu_delta=self.config.get('mu_delta', 1.0),
            cce_balance=self.config.get('cce_balance', None),
            target_ess_frac=self.config.get('target_ess_frac', None),
            ess_recalib_every=self.config.get('ess_recalib_every', 50),
        )

        self.score_interval = self.config.get('score_interval', 4)
        self.n_score_sample = self.config.get('n_score_sample', 512)
        self.consequence_metric = self.config.get('consequence_metric', 'wasserstein')
        self.consequence_aggregation = self.config.get('consequence_aggregation', 'weighted_mean')
        self.cf_horizon = self.config.get('cf_horizon', 10)
        self.cf_n_rollouts = self.config.get('cf_n_rollouts', 1)
        self.cf_gamma = self.config.get('cf_gamma', 1.0)
        self.cf_rollout_temperature = self.config.get('cf_rollout_temperature', 0.0)
        self.cf_bootstrap = self.config.get('cf_bootstrap', True)

        # Softmax temperature used ONLY to weight alternatives in 'weighted_mean'.
        # It must be its own knob: the rollout continuation runs at temperature 0
        # (greedy, which is what makes cf_n_rollouts=1 exact), and reusing that here
        # would make action_probs a one-hot on the taken action. Every alternative
        # would then get weight ~0, the weights would sum to 0, and
        # compute_consequence_metric would silently fall back to an unweighted mean
        # -- the same defect as repo issue #3.
        self.cf_score_temperature = self.config.get('cf_score_temperature', 1.0)

        self._all_actions = jnp.arange(self.n_actions, dtype=jnp.int32)
        self.q_update_count = 0
        self._compiled_rollout_fn = None

        # COVERAGE INSTRUMENTATION. Nothing in ConsequenceReplayBuffer records which
        # entries hold a MEASURED score -- on insertion an entry inherits the running
        # mean (Algorithm 1 line 7), so consequence_scores is never zero and cannot be
        # used to tell scored from unscored. Without this the coverage number is
        # derived, not observed, and the derived figure has already been shown to be
        # about 2x the instantaneous truth on JaxNav.
        #   scored_at[i] = the q_update_count at which slot i was last scored, else -1.
        # Owned here rather than in the shared buffer, which another session is editing.
        self._scored_at = np.full(self.config['buffer_capacity'], -1, dtype=np.int64)
        self._buffer_write_pos = 0

    # ------------------------------------------------------------------
    # Counterfactual rollout (shared implementation)
    # ------------------------------------------------------------------

    def _build_rollout_fn(self):
        """(B states) x (A actions) x (N rollouts) -> returns (B, A, N)."""
        self._compiled_rollout_fn = build_rollout_fn(
            self.env.step,
            self.network.apply,
            horizon=self.cf_horizon,
            gamma=self.cf_gamma,
            temperature=self.cf_rollout_temperature,
            bootstrap=self.cf_bootstrap,
        )

    # ------------------------------------------------------------------

    def _score_buffer_transitions(self):
        """Score a uniform sample of stored transitions by counterfactual return spread."""
        n_score = min(self.n_score_sample, len(self.buffer))
        if n_score == 0:
            return

        transitions, indices = self.buffer.sample_uniform(n_score)

        valid_states, valid_obs, valid_actions_taken, valid_indices = [], [], [], []
        for i, (t, idx) in enumerate(zip(transitions, indices)):
            s = self.buffer.get_jax_state(idx)
            if s is None:
                continue
            valid_states.append(s)              # a CVRPState pytree
            valid_obs.append(t['s'])            # (n_nodes, F) obs at that state
            valid_actions_taken.append(int(t['a']))
            valid_indices.append(i)

        if not valid_states:
            return

        B = len(valid_states)
        N = self.cf_n_rollouts
        A = self.n_actions

        if self._compiled_rollout_fn is None:
            print("Compiling consequence rollout function (one-time cost)...", flush=True)
            self._build_rollout_fn()

        states_batched = jax.tree.map(lambda *leaves: jnp.stack(leaves), *valid_states)

        self._key, subkey = jax.random.split(self._key)
        keys_array = jax.random.split(subkey, B * A * N).reshape(B, A, N, 2)

        returns = self._compiled_rollout_fn(
            self.params, self.target_params, states_batched, self._all_actions, keys_array
        )
        returns = jax.block_until_ready(returns)
        returns_np = np.array(returns)                                   # (B, A, N)

        obs_batch = jnp.asarray(np.stack(valid_obs), dtype=jnp.float32)  # (B, n_nodes, F)
        legal = np.array(self.env.action_mask_from_obs(obs_batch))       # (B, A) bool

        # Alternative weights for 'weighted_mean', over legal actions only.
        q_batch = jax.vmap(self.network.apply, in_axes=(None, 0))(self.params, obs_batch)
        temp = max(self.cf_score_temperature, 1e-3)
        probs_batch = np.array(jax.nn.softmax(q_batch / temp, axis=-1))  # (B, A)

        scores = np.zeros(B)
        for i in range(B):
            taken_action = valid_actions_taken[i]
            legal_i = np.flatnonzero(legal[i])
            # The action actually taken must be present as the reference, even in the
            # pathological case where the stored mask no longer marks it legal.
            if taken_action not in legal_i:
                legal_i = np.append(legal_i, taken_action)
            if len(legal_i) < 2:
                # Forced move: there was no choice, so the choice changed nothing.
                scores[i] = 0.0
                continue
            return_distributions = {(int(a),): returns_np[i, a] for a in legal_i}
            action_probs = {(int(a),): float(probs_batch[i, a]) for a in legal_i}
            scores[i] = compute_consequence_metric(
                action=(taken_action,),
                return_distributions=return_distributions,
                metric=self.consequence_metric,
                action_probs=action_probs,
                aggregation=self.consequence_aggregation,
            )
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        scored_indices = indices[np.array(valid_indices)]
        self.buffer.update_consequence_scores(scored_indices, scores)
        self._scored_at[scored_indices] = self.q_update_count

    def coverage_diagnostics(self) -> dict:
        """What fraction of the LIVE buffer actually holds a measured score?

        Slots evicted since they were scored are excluded by re-stamping them to -1 in
        _add_chunk_to_buffer, so this is the instantaneous fraction the sampler sees --
        not the lifetime expectation n_score_sample/(score_interval*n_steps_per_update),
        which overstates it.
        """
        n = len(self.buffer)
        if n == 0:
            return {}
        live = self._scored_at[:n]
        scored = live >= 0
        ages = (self.q_update_count - live[scored]) if scored.any() else np.array([0])
        return {
            'scored_frac': float(scored.mean()),
            'n_scored': int(scored.sum()),
            'buffer_size': int(n),
            'buffer_full': bool(n >= self.config['buffer_capacity']),
            'staleness_p50': float(np.percentile(ages, 50)),
            'staleness_p90': float(np.percentile(ages, 90)),
            'q_updates': int(self.q_update_count),
        }

    # ------------------------------------------------------------------
    # Training step (score periodically, then a normal Q-update)
    # ------------------------------------------------------------------

    def _update(self):
        # buffer_min_samples was in the config but never read: we started learning
        # as soon as one batch existed (1024) where RP-DQN waits for 2^13 = 8192
        # (AttentionModel.py gates on can_provide_sample(buffer_min_samples)).
        # Starting 8x earlier means the first updates run on a nearly-empty,
        # highly correlated buffer.
        min_samples = max(self.batch_size, self.config.get('buffer_min_samples', 0))
        if not self.buffer.can_sample(min_samples):
            return

        self.q_update_count += 1
        if (self.q_update_count % self.score_interval == 0
                and len(self.buffer) >= self.n_score_sample):
            timer = getattr(getattr(self, 'metrics_logger', None), 'timer', None)
            if timer is not None:
                with timer('update.scoring', episode=self._current_episode):
                    self._score_buffer_transitions()
            else:
                self._score_buffer_transitions()

        transitions, indices, weights = self.buffer.sample(self.batch_size)
        states  = jnp.asarray(np.stack([t['s']  for t in transitions]), dtype=jnp.float32)
        actions = jnp.asarray([t['a']    for t in transitions], dtype=jnp.int32)
        rewards = jnp.asarray([t['r']    for t in transitions], dtype=jnp.float32)
        nexts   = jnp.asarray(np.stack([t["s'"] for t in transitions]), dtype=jnp.float32)
        dones   = jnp.asarray([t['done'] for t in transitions], dtype=jnp.float32)
        wts     = jnp.asarray(weights, dtype=jnp.float32)

        self.params, self.opt_state, td_errors = self._update_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, nexts, dones, wts,
        )
        self.buffer.update_priorities(indices, np.array(td_errors))

        if self.target_tau is not None:
            self.target_params = self._polyak(
                self.target_params, self.params, jnp.float32(self.target_tau)
            )
