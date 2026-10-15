"""
Consequence-weighted DQN (Algorithm 2) for JaxNav.

Mirror of ``agents/frozen_lake/consequence_dqn.py``. Same scoring pipeline —
store the env state at each transition, periodically roll out all actions from a
uniform sample of stored states, and weight replay priorities by the spread
(total variation) among the per-action return distributions.

Two JaxNav-specific changes, both discussed in the plan:
  * State is a pytree (not an int): stored ``jax_state`` is a full JaxNav ``State``;
    rollouts thread the observation to the network while stepping the state.
  * The rollout continuation policy is STOCHASTIC (softmax temperature
    ``cf_rollout_temperature``). JaxNav transitions are deterministic, so a greedy
    continuation would collapse each action's N rollouts to a single value and TV
    would degrade to a coarse 0/1; sampling restores graded spread. Set the
    temperature to 0 for a greedy ablation.

Variants (identical knobs to FrozenLake):
  algorithm='consequence-dqn', mu=1.0                            → CCE-only
  algorithm='consequence-dqn', priority_mixing='additive'        → CCE + TD additive (Eq. 4)
  algorithm='consequence-dqn', priority_mixing='multiplicative'  → CCE + TD multiplicative (Eq. 5)
"""

import json
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .dqn import JaxNavDQN, _MetricsLogger
from counterfactual_rl.replay.consequence_buffers import ConsequenceReplayBuffer
from counterfactual_rl.scoring.metrics import compute_consequence_metric
from counterfactual_rl.scoring.weighted_wasserstein import weighted_wasserstein_scores

# Rollout terminal-reason codes (same order/values as ~/probe_logs/probe_reasons.py).
_R_ALIVE, _R_GOAL, _R_CRASH, _R_TIMEUP = 0, 1, 2, 3


class JaxNavConsequenceDQN(JaxNavDQN):
    """Consequence-weighted DQN for JaxNav (Algorithm 2)."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        per = self.config.get('PER_parameters', {})
        self.buffer = getattr(self, 'replay_buffer_class', ConsequenceReplayBuffer)(
            capacity=self.config['buffer_capacity'],
            eps=per.get('eps', 0.01),
            beta=per.get('beta', 0.25),
            max_priority=per.get('maximum_priority', 1.0),
            mu=self.config.get('mu', 0.5),
            priority_mixing=self.config.get('priority_mixing', 'additive'),
            mu_c=self.config.get('mu_c', 1.0),
            mu_delta=self.config.get('mu_delta', 1.0),
            cce_balance=self.config.get('cce_balance', None),
            target_ess_frac=self.config.get('target_ess_frac', None),
            ess_recalib_every=self.config.get('ess_recalib_every', 50),
            normalise_priorities=self.config.get('normalise_priorities', 'none'),
        )

        self.score_interval = self.config.get('score_interval', 100)
        self.n_score_sample = self.config.get('n_score_sample', 128)
        # score-on-insert: score every arrival exactly once instead of drawing at random.
        # Default OFF so every run already on disk reproduces bit-for-bit. When ON,
        # score_interval and n_score_sample stop meaning anything -- there is nothing to
        # choose, every transition is scored once.
        self.score_on_insert = bool(self.config.get('score_on_insert', False))
        # Rows per rollout batch. 1,280 x 15 actions = 19,200 parallel envs, the widest
        # batch measured safe on a T4 (greedy_cov ran exactly this).
        self.cf_score_chunk = self.config.get('cf_score_chunk', 1280)
        self.consequence_metric = self.config.get('consequence_metric', 'total_variation')
        self.consequence_aggregation = self.config.get('consequence_aggregation', 'weighted_mean')
        self.cf_horizon = self.config.get('cf_horizon', 20)
        self.cf_n_rollouts = self.config.get('cf_n_rollouts', 20)
        self.cf_gamma = self.config.get('cf_gamma', 0.99)
        self.cf_rollout_temperature = self.config.get('cf_rollout_temperature', 0.5)
        self.cf_bootstrap = self.config.get('cf_bootstrap', False)

        self._all_actions = jnp.arange(self.n_actions, dtype=jnp.int32)
        self.q_update_count = 0
        self._compiled_rollout_fn = None
        # Diagnostics accumulators, drained by _log_priority_diagnostics at each eval.
        self._last_rollout_reasons: dict = {}
        self._last_rollout_n = 0
        self._double_sample_overlaps: list = []

        # DRIFT PROBE. A CCE score is written once and never recomputed; the measured
        # median staleness is ~1,788 Q-updates (~14 target-net refreshes). Whether that
        # MATTERS depends on how fast a score actually moves, which nothing has ever
        # measured -- the buffer snapshots hold only the priority arrays, not the
        # states, so it cannot be done after the fact. This keeps a FIXED set of
        # transitions (states copied out of the buffer so later overwrites cannot touch
        # them) and re-scores exactly those, with the current network, at every eval.
        # Correlating score vectors across time gives rho(delta) -- the answer to "is a
        # 1,788-update-old score still the same score?". TD magnitudes for the same
        # transitions are logged alongside, so CCE drift can be read against the signal
        # that IS refreshed for free. Off by default; costs one extra scoring pass per
        # probe.
        self.cf_drift_probe_n = int(self.config.get('cf_drift_probe_n', 0))
        self.cf_drift_probe_start = int(self.config.get('cf_drift_probe_start_ep', 20_000))
        self._probe = None          # dict(states=batched pytree, obs, actions)

    # ------------------------------------------------------------------
    # Counterfactual rollout
    # ------------------------------------------------------------------

    # Rollout terminal reasons. Only GOAL pays under the sparse reward, so these
    # split the "return == 0" mass into its real causes.
    _REASON_NAMES = ('alive', 'goal', 'crash', 'timeup')

    def _build_rollout_fn(self):
        """Triple-vmapped JIT rollout: (B states) x (n_actions) x (N rollouts).

        Returns ``(returns, reasons)``, both (B, A, N). ``reasons`` says HOW each
        counterfactual rollout ended -- goal / crash / timeout / still alive at the
        horizon -- because only GOAL pays (goal_rew=1.0, coll_rew=0.0, shaping
        zeroed), so the terminal reason is what splits the large "return == 0" mass
        into "the action was genuinely worthless" and "the rollout just ran out of
        room". CCE assumes these rollouts predict what really happens and that has
        never been checked; logging reasons here lets the rollout outcome
        distribution sit next to the EVAL outcome distribution from evaluate().

        The reason bookkeeping is lifted verbatim from ~/probe_logs/probe_reasons.py,
        which was itself written to mirror this function.
        """
        env = self.env
        network = self.network
        horizon = self.cf_horizon
        gamma = self.cf_gamma
        temperature = self.cf_rollout_temperature
        bootstrap = self.cf_bootstrap
        max_steps = env.max_steps

        def reason_of(st):
            g = jnp.asarray(st.goal_reached).ravel()[0]
            c = jnp.asarray(st.move_term).ravel()[0]
            t = jnp.asarray(st.step) >= max_steps
            return jnp.where(g, _R_GOAL, jnp.where(c, _R_CRASH,
                             jnp.where(t, _R_TIMEUP, _R_ALIVE)))

        def single_rollout(params, target_params, state, first_action, rng_key):
            rng_key, step_key = jax.random.split(rng_key)
            next_obs, next_state, reward, done, _ = env.step(step_key, state, first_action)
            r0 = jnp.where(done, reason_of(next_state), _R_ALIVE).astype(jnp.int32)
            init_carry = (next_state, next_obs, rng_key, reward, jnp.float32(gamma), done, r0)

            def scan_step(carry, _):
                s, obs, key, cum_ret, disc, done_flag, reason = carry
                q = network.apply(params, obs)
                key, ak, sk = jax.random.split(key, 3)
                if temperature and temperature > 0:
                    # stochastic continuation -> spreads the N rollouts (deterministic env)
                    action = jax.random.categorical(ak, q / temperature)
                else:
                    action = jnp.argmax(q)
                n_obs, ns, r, nd, _ = env.step(sk, s, action)
                masked_r = jnp.where(done_flag, 0.0, r)
                new_cum = cum_ret + disc * masked_r
                new_disc = jnp.where(done_flag, disc, disc * gamma)
                new_done = jnp.logical_or(done_flag, nd)
                # record the reason only on the transition where done FIRST flips
                new_reason = jnp.where(~done_flag & nd, reason_of(ns), reason).astype(jnp.int32)
                return (ns, n_obs, key, new_cum, new_disc, new_done, new_reason), None

            final, _ = jax.lax.scan(scan_step, init_carry, xs=None, length=horizon - 1)
            _, obs_H, _, cum, disc, done, reason = final
            if not bootstrap:
                return cum, reason  # cumulative discounted return, truncated at the horizon

            # The scan ran `horizon` env steps in total (the first action plus
            # horizon-1 more), and `disc` is frozen at gamma**horizon, so it is
            # exactly the discount this bootstrap term belongs at. `done` is only
            # true for a REAL terminal -- goal, collision or time_up -- and those
            # are correctly worth 0, so the term is masked off for them.
            q_H = network.apply(target_params, obs_H)
            if temperature and temperature > 0:
                # Value under the policy the rollout actually followed. A max()
                # bootstrap would assume greedy continuation and systematically
                # overestimate every truncated rollout.
                v_H = jnp.sum(jax.nn.softmax(q_H / temperature) * q_H)
            else:
                v_H = jnp.max(q_H)
            return cum + jnp.where(done, 0.0, disc * v_H), reason

        over_rollouts = jax.vmap(single_rollout, in_axes=(None, None, None, None, 0))  # N keys
        over_actions  = jax.vmap(over_rollouts,  in_axes=(None, None, None, 0, 0))     # A actions
        over_states   = jax.vmap(over_actions,   in_axes=(None, None, 0, None, 0))     # B states
        self._compiled_rollout_fn = jax.jit(over_states)

    def _score_buffer_transitions(self):
        """Score a uniform RANDOM sample of stored transitions -- the historical path.

        This is the coupon-collector route. A slot is scored only if a draw happens to land
        on it, and over its whole 6,250-update life it gets 0.32 expected draws, so ~73% of
        slots die carrying the running mean they inherited on insertion. See _score_indices
        for the score-on-insert path, which visits each arrival exactly once instead.
        """
        n_score = min(self.n_score_sample, len(self.buffer))
        if n_score == 0:
            return
        _, indices = self.buffer.sample_uniform(n_score)
        self._score_indices(indices)

    def _score_indices(self, indices):
        """Score exactly these buffer slots, in GPU-sized chunks.

        Chunking is NOT optional for score-on-insert: one insert burst is
        n_envs x collect_steps = 8,192 transitions, and 8,192 x 15 actions x 1 rollout is
        122,880 parallel envs -- 6.4x the widest batch ever run here (1,280 x 15 = 19,200,
        the proven-safe T4 width). Unchunked it OOMs on the first burst.
        """
        indices = np.asarray(indices)
        if indices.size == 0:
            return
        chunk = max(1, int(self.cf_score_chunk))
        for start in range(0, indices.size, chunk):
            self._score_index_chunk(indices[start:start + chunk])

    def _score_index_chunk(self, indices):
        """Score one chunk. Body unchanged from the original random-sample path; only the
        SOURCE of `indices` differs."""
        transitions = [self.buffer.get_transition(idx) for idx in indices]

        valid_states, valid_obs, valid_actions_taken, valid_indices = [], [], [], []
        for i, (t, idx) in enumerate(zip(transitions, indices)):
            s = self.buffer.get_jax_state(idx)
            if s is None:
                continue
            valid_states.append(s)                 # a JaxNav State pytree (numpy leaves)
            valid_obs.append(t['s'])                # the observation at that state (for action_probs)
            valid_actions_taken.append(int(t['a']))
            valid_indices.append(i)

        if not valid_states:
            return

        B = len(valid_states)
        N = self.cf_n_rollouts
        A = self.n_actions

        if self._compiled_rollout_fn is None:
            print("Compiling consequence rollout function (one-time cost)...")
            self._build_rollout_fn()

        # Stack the per-transition State pytrees into one batched pytree (leading axis B).
        # Stacked in numpy, then ONE transfer per leaf. The stored leaves are already numpy,
        # and jnp.stack on them dispatched an eager JAX op per element per leaf -- ~98k calls,
        # 19.9 s of a 57.6 s chunk (profiled). jnp.asarray applies the same dtype conversion,
        # so the batch is identical.
        states_batched = jax.tree.map(lambda *leaves: jnp.asarray(np.stack(leaves)), *valid_states)

        self._key, subkey = jax.random.split(self._key)
        keys_array = jax.random.split(subkey, B * A * N).reshape(B, A, N, 2)

        # CHUNK THE ROLLOUT so GPU memory does not cap n_score_sample. The triple vmap
        # runs B x A x N environments at once: at the shipped 64 x 15 x 20 that is 19,200,
        # but n_score_sample=256 asks for 76,800 and a T4 OOMs at ~20.6 GiB (measured,
        # job 274872). The lab notebook records 38,400 as near the ceiling. Coverage is
        # proportional to n_score_sample, so leaving it memory-bound would block the
        # coverage sweep (issue #22) outright. Chunking makes the two independent: the
        # scores are identical, only the number of kernel launches changes.
        max_envs = int(self.config.get('cf_score_max_envs', 20_000))
        per_state = A * N
        chunk_B = max(1, min(B, max_envs // max(1, per_state)))
        r_parts, reason_parts = [], []
        for lo in range(0, B, chunk_B):
            hi = min(lo + chunk_B, B)
            sub = jax.tree.map(lambda leaf, a=lo, b=hi: leaf[a:b], states_batched)
            r, rs = self._compiled_rollout_fn(
                self.params, self.target_params, sub, self._all_actions, keys_array[lo:hi]
            )
            r_parts.append(np.array(jax.block_until_ready(r)))
            reason_parts.append(np.array(rs))
        returns_np = np.concatenate(r_parts, axis=0)      # (B, A, N)
        reasons = np.concatenate(reason_parts, axis=0)

        # Share of rollouts ending each way, over all B x A x N of this scoring pass.
        reasons_np = reasons.ravel()
        self._last_rollout_reasons = {
            f'rollout_{name}_frac': float(np.mean(reasons_np == code))
            for code, name in enumerate(self._REASON_NAMES)
        }
        self._last_rollout_n = int(reasons_np.size)

        # action_probs for weighted_mean aggregation (fixes the bug where 'weighted_mean' was
        # requested but compute_consequence_metric silently fell back to 'max' whenever
        # action_probs was None -- see repo issue #3, which found the same bug on FrozenLake's
        # near-identical scoring code). Softmax the scored state's own Q-values (same temperature
        # as the rollout continuation policy) so every alternative gets a real nonzero weight --
        # a pure argmax-derived action_probs would put 0 on every alternative and just fall
        # through to a plain mean anyway (per the issue), so softmax is what actually makes
        # 'weighted_mean' behave as a genuine weighted mean instead of degenerating either way.
        obs_batch = jnp.asarray(np.stack(valid_obs), dtype=jnp.float32)
        q_batch = jax.vmap(self.network.apply, in_axes=(None, 0))(self.params, obs_batch)
        # These probabilities are the WEIGHTS for 'weighted_mean' aggregation, and until
        # now they shared the rollout's temperature. That couples two unrelated things:
        # at cf_rollout_temperature=0 the softmax goes one-hot (measured effective sample
        # size 1.0/14, vs 11.8-14.0/14 at T=0.5), so 'weighted_mean' silently becomes
        # 'max'. A greedy-rollout arm could then look like an aggregation change rather
        # than a rollout change. cf_action_prob_temperature decouples them; left unset it
        # falls back to cf_rollout_temperature, i.e. identical to every run already on
        # disk. 'max' aggregation ignores action_probs entirely and cannot be affected.
        ap_t = self.config.get('cf_action_prob_temperature')
        if ap_t is None:
            ap_t = self.cf_rollout_temperature
        temp = max(float(ap_t), 1e-3)
        probs_batch = np.array(jax.nn.softmax(q_batch / temp, axis=-1))  # (B, A)

        if (N == 1 and self.consequence_metric == 'wasserstein'
                and self.consequence_aggregation == 'weighted_mean'):
            scores = weighted_wasserstein_scores(
                returns_np, probs_batch, np.asarray(valid_actions_taken))
        else:
            scores = np.zeros(B)
            for i in range(B):
                taken_action = valid_actions_taken[i]
                return_distributions = {(a,): returns_np[i, a] for a in range(A)}
                action_probs = {(a,): float(probs_batch[i, a]) for a in range(A)}
                scores[i] = compute_consequence_metric(
                    action=(taken_action,),
                    return_distributions=return_distributions,
                    metric=self.consequence_metric,
                    action_probs=action_probs,
                    aggregation=self.consequence_aggregation,
                )
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        scored_indices = indices[np.array(valid_indices)]
        self.buffer.update_consequence_scores(scored_indices, scores,
                                              q_update_count=self.q_update_count)

    # ------------------------------------------------------------------
    # Drift probe
    # ------------------------------------------------------------------

    def _score_states(self, states_batched, obs_list, actions_taken):
        """Score an arbitrary batch of transitions with the CURRENT network.

        Shares the compiled rollout function and the chunking with
        _score_buffer_transitions, but reads nothing from and writes nothing to the
        buffer -- so the probe cannot perturb training.
        """
        B, A, N = len(actions_taken), self.n_actions, self.cf_n_rollouts
        if self._compiled_rollout_fn is None:
            self._build_rollout_fn()
        self._key, subkey = jax.random.split(self._key)
        keys_array = jax.random.split(subkey, B * A * N).reshape(B, A, N, 2)
        max_envs = int(self.config.get('cf_score_max_envs', 20_000))
        chunk_B = max(1, min(B, max_envs // max(1, A * N)))
        parts = []
        for lo in range(0, B, chunk_B):
            hi = min(lo + chunk_B, B)
            sub = jax.tree.map(lambda leaf, a=lo, b=hi: leaf[a:b], states_batched)
            r, _ = self._compiled_rollout_fn(
                self.params, self.target_params, sub, self._all_actions, keys_array[lo:hi])
            parts.append(np.array(jax.block_until_ready(r)))
        returns_np = np.concatenate(parts, axis=0)

        obs_batch = jnp.asarray(np.stack(obs_list), dtype=jnp.float32)
        q_batch = jax.vmap(self.network.apply, in_axes=(None, 0))(self.params, obs_batch)
        ap_t = self.config.get('cf_action_prob_temperature')
        if ap_t is None:
            ap_t = self.cf_rollout_temperature
        probs_batch = np.array(jax.nn.softmax(q_batch / max(float(ap_t), 1e-3), axis=-1))

        scores = np.zeros(B)
        for i in range(B):
            scores[i] = compute_consequence_metric(
                action=(actions_taken[i],),
                return_distributions={(a,): returns_np[i, a] for a in range(A)},
                metric=self.consequence_metric,
                action_probs={(a,): float(probs_batch[i, a]) for a in range(A)},
                aggregation=self.consequence_aggregation,
            )
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0), np.array(q_batch)

    def _maybe_init_drift_probe(self, episode: int):
        """Freeze a fixed probe set once, by COPYING states out of the buffer.

        Copied on purpose: buffer slots are overwritten every ~12 chunks, so holding
        indices would silently re-point the probe at different transitions and the
        correlation would measure the ring pointer instead of score drift.
        """
        if (self.cf_drift_probe_n <= 0 or self._probe is not None
                or episode < self.cf_drift_probe_start
                or len(self.buffer) < self.cf_drift_probe_n):
            return
        transitions, indices = self.buffer.sample_uniform(self.cf_drift_probe_n)
        states, obs, acts = [], [], []
        for t, idx in zip(transitions, indices):
            st = self.buffer.get_jax_state(idx)
            if st is None:
                continue
            states.append(jax.tree.map(lambda x: jnp.array(x), st))
            obs.append(np.array(t['s']))
            acts.append(int(t['a']))
        if not states:
            return
        self._probe = dict(
            states=jax.tree.map(lambda *l: jnp.stack(l), *states),
            obs=obs, actions=acts, episode=episode)
        print(f"[drift] probe frozen: {len(acts)} transitions at episode {episode}")

    def run_drift_probe(self, episode: int) -> Optional[dict]:
        """Re-score the frozen probe set and append one line to drift.jsonl."""
        self._maybe_init_drift_probe(episode)
        if self._probe is None:
            return None
        scores, q = self._score_states(self._probe['states'], self._probe['obs'],
                                       self._probe['actions'])
        # TD magnitude for the same transitions, from the same forward pass, so CCE
        # drift can be read against the signal that IS refreshed for free.
        q_taken = np.array([q[i, a] for i, a in enumerate(self._probe['actions'])])
        rec = dict(episode=int(episode), q_update=int(self.q_update_count),
                   n=len(scores), cce=[round(float(x), 6) for x in scores],
                   q_taken=[round(float(x), 6) for x in q_taken])
        path = os.path.join(self.metrics_logger.dir, 'drift.jsonl')
        with open(path, 'a') as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    # ------------------------------------------------------------------
    # Training step (score periodically, then a normal Q-update)
    # ------------------------------------------------------------------

    def _update(self):
        if not self.buffer.can_sample(self.batch_size):
            return

        self.q_update_count += 1
        # Under score-on-insert the scoring clock is INSERTION, not the update counter --
        # the trainer calls _score_indices on each burst. Firing here too would re-score
        # random slots on top of it and reintroduce the waste this mode exists to remove.
        if (not self.score_on_insert
                and self.q_update_count % self.score_interval == 0
                and len(self.buffer) >= self.n_score_sample):
            self._score_buffer_transitions()

        transitions, indices, weights = self.buffer.sample(self.batch_size)

        # DOUBLE SAMPLING (Jeremy 07/10, never built until now). On the same cadence
        # as scoring, draw a shadow batch from TD priority ALONE, train on nothing,
        # and record how much it overlaps the batch we really used. Overlap near the
        # uniform baseline (batch_size / |buffer|) means CCE is picking the same
        # transitions PER would have -- a direct answer, where ess_frac only reports
        # how sharp the distribution is, not which entries it chose.
        if self.q_update_count % self.score_interval == 0:
            shadow = self.buffer.sample_td_only_indices(self.batch_size)
            if shadow.size:
                overlap = len(set(indices.tolist()) & set(shadow.tolist())) / len(indices)
                self._double_sample_overlaps.append(float(overlap))

        states  = jnp.asarray(np.stack([t['s']  for t in transitions]), dtype=jnp.float32)
        actions = jnp.asarray([t['a']    for t in transitions], dtype=jnp.int32)
        rewards = jnp.asarray([t['r']    for t in transitions], dtype=jnp.float32)
        nexts   = jnp.asarray(np.stack([t["s'"] for t in transitions]), dtype=jnp.float32)
        dones   = jnp.asarray([t['done'] for t in transitions], dtype=jnp.float32)
        wts     = jnp.asarray(weights, dtype=jnp.float32)

        prev_params = self.params
        self.params, self.opt_state, td_errors = self._update_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, nexts, dones, wts,
        )
        self.buffer.update_priorities(indices, np.array(td_errors))

        qdiag = getattr(self, '_qdiag', None)
        if qdiag is not None:
            # Same batch, and the PRE-update params, so the line describes the step
            # that just ran. Reached only past the can_sample() guard at the top, so
            # updates that never happened cannot dilute the per-eval means.
            qdiag.maybe_record(prev_params, self.target_params,
                               states, actions, rewards, nexts, dones, wts)

    # ------------------------------------------------------------------
    # Sequential learn (the vectorized subclass is the workhorse)
    # ------------------------------------------------------------------

    def learn(self, n_episodes: Optional[int] = None, verbose: bool = True) -> 'JaxNavConsequenceDQN':
        n_episodes = n_episodes or self.config['n_episodes']
        eval_interval = self.config.get('eval_interval', 300)
        eval_episodes = self.config.get('eval_episodes', 100)
        save_every = self.config.get('save_every', 1000)

        self.metrics_logger = _MetricsLogger(
            config=self.config, n_episodes=n_episodes,
            eval_interval=eval_interval, eval_episodes=eval_episodes,
        )
        timer = self.metrics_logger.timer
        timer.start('total')

        last_path = os.path.join(self.metrics_logger.dir, 'last.pkl')
        best_path = os.path.join(self.metrics_logger.dir, 'best.pkl')
        best_success = -1.0

        np.random.seed(self.config.get('seed', 0))

        if verbose:
            print(f"Training Consequence-DQN on JaxNav [{self.config.get('scenario') or self.config.get('map_id')}]")
            print(f"  Priority mixing: {self.config.get('priority_mixing')}  |  mu: {self.config.get('mu')}")
            print(f"  Metric: {self.consequence_metric}  |  Score interval: {self.score_interval}")
            print(f"  CF rollouts: {self.cf_n_rollouts}  |  horizon: {self.cf_horizon}  |  temp: {self.cf_rollout_temperature}")
            print(f"  JAX backend: {jax.default_backend()}  |  Run dir: {self.metrics_logger.dir}")

        pbar = tqdm(range(n_episodes), disable=not verbose)
        for episode in pbar:
            self._current_episode = episode
            self._key, rk = jax.random.split(self._key)
            obs, state = self.env.reset(rk)
            done = False
            ep_return = 0.0
            ep_steps = 0
            while not done and ep_steps < self.env.max_steps:
                saved_state = jax.tree.map(np.array, state)   # pre-step state for rollouts
                action = self.select_action(obs)
                self._key, sk = jax.random.split(self._key)
                next_obs, next_state, reward, done, _ = self.env.step(sk, state, jnp.int32(action))
                reward = float(reward)
                done = bool(done)
                self.buffer.add(
                    {'s': np.asarray(obs, np.float32), 'a': int(action),
                     'r': reward, "s'": np.asarray(next_obs, np.float32), 'done': done},
                    jax_state=saved_state,
                )
                self.total_steps += 1
                if self.total_steps % self.n_steps_per_update == 0:
                    self._update()
                if self.total_steps % self.target_update_freq == 0:
                    self._update_target_network()
                state, obs = next_state, next_obs
                ep_return += reward
                ep_steps += 1

            self.episode_returns.append(ep_return)
            self.episode_lengths.append(ep_steps)
            decay = min(1.0, episode / self.epsilon_decay_episodes)
            self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay

            if len(self.episode_returns) >= 100:
                avg = np.mean(self.episode_returns[-100:])
                pbar.set_description(f"AvgR(100): {avg:.3f}  ε: {self.epsilon:.3f}")

            if (episode + 1) % save_every == 0:
                self.save(last_path)
            if eval_interval and (episode + 1) % eval_interval == 0:
                metrics = self.evaluate(eval_episodes)
                self.metrics_logger.log_eval(episode + 1, self.q_update_count, self.epsilon, metrics)
                self._write_qdiag(episode + 1, self.q_update_count)
                if metrics['win_rate'] > best_success:
                    best_success = metrics['win_rate']
                    self.save(best_path)

        self.save(last_path)
        timer.stop('total')
        self.metrics_logger.plot_training_curves(self.episode_returns, self.episode_lengths)
        self.metrics_logger.close()
        if verbose:
            print(f"\nTraining complete. Run saved to {self.metrics_logger.dir}")
        return self
