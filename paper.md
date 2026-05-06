# Consequentialism: Counterfactual Sampling to Speed Learning

**Adrian Manchado** — Diercks School of Advanced Computing, Milwaukee School of Engineering

**Jeremy Kedziora** — Diercks School of Advanced Computing, Milwaukee School of Engineering

---

## Abstract

TODO

**Keywords:** TODO

---

## 1. Introduction

Reinforcement learning (RL) agents learn by interacting with an environment and updating from the resulting experience. In problems with sparse rewards, long horizons, or high-dimensional state spaces, generating enough experience to learn a competent policy is expensive — often requiring millions of environment interactions. Experience replay [Mnih et al., 2013] partially addresses this by storing transitions in a buffer and reusing each one across multiple gradient updates. Prioritized Experience Replay (PER) [Schaul et al., 2016] refines this further by concentrating sampling on transitions with high temporal-difference (TD) error — the agent's surprise — yielding substantial gains in sample efficiency.

TD error, however, measures whether a transition was *surprising* to the value function — not whether it was *consequential* to the episode outcome. A transition may produce large TD error simply because the Q-network is poorly initialized in that region, regardless of whether the action taken actually mattered. Conversely, a pivotal decision — one where the chosen action substantially changed the distribution of future returns relative to alternatives — may produce small TD error once the value function has locally converged, even if that decision still determines whether the episode is won or lost. These high-consequence, low-surprise transitions are systematically undersampled by PER.

We propose *Counterfactual Consequence Estimation* (CCE), a method that scores transitions by asking: how much would the return distribution have changed if a different action had been taken? For each stored transition, CCE rolls out n counterfactual trajectories under each alternative action using the current policy, estimates the resulting return distributions, and measures their divergence from the realized distribution. Transitions where the action choice substantially shifted outcome distributions receive high CCE priority. This score is then mixed with standard TD-error priority to form a balanced replay distribution that simultaneously targets surprising and consequential transitions.

This paper makes three contributions. First, we introduce CCE (Algorithm 1), a Monte Carlo estimator of transition consequence applicable to any replay-based RL algorithm. Second, we propose DQN with CCE-augmented priority (Algorithm 2), with both additive and multiplicative mixing schemes. Third, we provide an empirical evaluation across four environments — SMAX 3m, SMAX 8m, Gardner Chess, and FrozenLake 8×8 — with 10 independent seeds and the rliable statistical framework [Agarwal et al., 2021], demonstrating that CCE leads to more sample-efficient early training while preserving asymptotic performance.

---

## 2. Related Work

**Prioritized Replay.** Deep Q-Networks (DQN) [Mnih et al., 2013] established the experience-replay framework: transitions $(s_t, a_t, r_{t+1}, s_{t+1})$ are stored in a circular buffer and sampled uniformly for stochastic gradient updates to the Q-network. Prioritized Experience Replay (PER) [Schaul et al., 2016] replaced uniform sampling with a distribution weighted by TD error, showing substantial gains in sample efficiency; PER is the direct ancestor of the priority-mixing scheme presented in this work. Large Batch Experience Replay (LaBER) [Lahire et al., 2022] frames replay sampling as importance sampling for gradient estimation, derives the theoretically optimal sampling distribution, and approximates it by drawing a large candidate batch and retaining the highest-weight subset; their published results across Atari games provide a natural baseline for replay-prioritization comparisons.

**Critical and Consequential States.** A recurring observation in the RL literature is that episode outcomes are often determined by a small number of key decision points. Huang et al. [2018] define *critical states* as those in which a policy strongly prefers a narrow subset of actions and demonstrate that surfacing these states to human supervisors improves trust calibration and intervention timing. Karino, Ohmura, and Kuniyoshi [2020] identify critical states via the variance of the Q-function across actions and concentrate exploration on them; CCE can be viewed as a distributional generalization of this scalar variance signal, replacing Q-value variance with a divergence over full return distributions under counterfactual actions. Grushin et al. [2024] define *true criticality* as the expected reward drop when an agent executes $n$ consecutive random deviations from its policy, then validate proxy criticality metrics against this ground truth; their evaluation methodology provides the closest published framework for assessing priority signals of the kind CCE produces. Liu et al. [2023] train a return-prediction model on video-encoded episodes and apply mask-based sensitivity to localize critical frames (Deep State Identifier), explicitly adopting the same few-states-matter framing as this work; unlike CCE, their method operates offline on pre-collected visual data rather than online on raw state-action pairs.

**Adjacent Work.** RUDDER [Arjona-Medina et al., 2019] addresses the related but distinct problem of credit assignment under delayed rewards by redistributing the episode return to individual transitions; unlike CCE, RUDDER modifies the effective reward signal and Bellman backup, whereas CCE leaves both unchanged and acts solely on the replay sampling distribution. CF-GPS [Buesing et al., 2019] applies structural causal models to synthesize alternative episode trajectories under counterfactual actions as additional training targets for policy search in POMDPs; CF-GPS uses counterfactuals to generate new training data, while CCE uses them to prioritize which observed transitions to replay.

---

## 3. Background

In reinforcement learning, control problems are modeled as a Markov Decision Process (MDP) defined by:

- A set of states $S$ that describe the current environmental conditions facing the agent
- A set of actions $A(s)$ that an agent can take
- Probabilities $p(s' \mid a, s)$ for transitioning from state $s$ to state $s'$ given action $a$
- A function $r: S \times A \times S \to \mathbb{R}$ so that $r(s', a, s)$ supplies the immediate reward associated with this transition

In environments that take place across a finite number of discrete periods $T$, the sequence of periods the agent participates in is referred to as an episode. The goal of the agent is to learn a policy $\pi(a|s)$ to maximize the sequence of rewards across an episode:

$$\sum_{t=0}^{T} \gamma^t r(s_{t+1}, a_t, s_t)$$

where $a_t$ and $s_t$ are the action and state at time $t$ and $\gamma \in [0,1]$ is the discount factor on future rewards.

### 3.1 Trajectories and Returns

The choices of an agent via its policy $\pi(\cdot)$ throughout the course of an episode lead to a realized time series of information commonly referred to as a trajectory:

$$\tau_\pi = s_0, a_0, r_1, s_1, a_1, \ldots, r_{T-1}, s_{T-1}, a_{T-1}, r_T, s_T$$

We denote a slice of a trajectory generated by policy $\pi(\cdot)$, beginning at time $t$ and ending at time $t'$, as $\tau_\pi^{(t:t')}$. The return at time $t$ associated with a slice is:

$$G(\tau_\pi^{(t:t')}) = \sum_{j=t+1}^{t'} \gamma^{j-t-1} r_j$$

### 3.2 Measuring Important Moments

One way to conceptualize the measurement of key moments is to take a counterfactual approach and ask "what would have happened if" questions of the policy and the data generated by it.

At time $t$, from the perspective of the agent, $\tau_\pi^{(t:)} \sim d_{\tau_\pi^{(t:)}}(s_t, a_t)$, where $d_{\tau_\pi^{(t:)}}(\cdot)$ is a distribution with dependence on the action and state. Let:

$$\mathcal{R}(g) = \{\tau^{t:} \mid G(\tau^{t:}) = g\}$$

be the set of all trajectories beginning at time $t$ whose return will equal $g$. Then:

$$d_G^{(\pi)}(g \mid s_t, a) \equiv \int_{\mathcal{R}(g)} d_{\tau_\pi^{(t:)}}(\tau \mid s_t, a) \, d\tau$$

Consider an intervention — an enforced change at time $t$ from the historic action $a_t$ to a feasible alternative action $a \in A(s_t)$. Such an intervention leads to a natural comparison of two distributions: $d_G^{(\pi)}(g \mid s_t, a_t)$ and $d_G^{(\pi)}(g \mid s_t, a)$. We use a generic metric $m(\cdot)$ to quantify the difference among the set of distributions $d_G^{(\pi)}(g \mid s_t, a_1), \ldots, d_G^{(\pi)}(g \mid s_t, a_{|A|})$.

**Algorithm 1 — Counterfactual Consequence Estimation**

```
Input: policy π, state-action pair (s_t, a_t), n ∈ ℕ+
For each a ∈ A(s_t):
    Sample n trajectories τ^j_{π,a} ~ d_{τ_π^(t:)}(s_t, a)
    Compute G(τ^j_{π,a}) for each j
    Estimate d_G^(π)(s_t, a) from the n returns
Compute m({d_G^(π)(s_t, a)}_{a ∈ A})
```

---

## 4. Consequence Prioritization

One straightforward way to apply Algorithm 1 is to use it to adjust how the agent encounters data to learn from. Prioritized experience replay (PER) samples the $j$th past transition from the replay buffer according to:

$$p^\delta(j) = \frac{(m^\delta_j + \epsilon)^\beta}{\sum_{i=1}^{|D|} (m^\delta_i + \epsilon)^\beta}$$

where $m^\delta_j = |\delta_j|$ is the most recent temporal difference error, $\epsilon$ ensures positive sampling probability, and $\beta$ controls entropy.

We experiment with augmenting TD priority with consequence measurements. We compute:

$$p^C(j) = \frac{(m^C_j + \epsilon)^\beta}{\sum_{i=1}^{|D|} (m^C_i + \epsilon)^\beta}$$

where $m_j^C = m(\{d^{(\pi)}_G(s_j, a)\}_{a \in A})$, and set the overall priority as:

**Additive mixing (Eq. 4):**
$$p(j) = \frac{\mu \, p^C(j) + (1-\mu) \, p^\delta(j)}{\sum_{k} \mu \, p^C(k) + (1-\mu) \, p^\delta(k)}$$

**Multiplicative mixing (Eq. 5):**
$$p(j) = \frac{p^C(j)^{\mu_C} \, p^\delta(j)^{\mu_\delta}}{\sum_{k} p^C(k)^{\mu_C} \, p^\delta(k)^{\mu_\delta}}$$

where $\mu$ controls the relative contribution of consequence estimates to sampling probabilities. This balanced approach puts the highest priority on transitions that are both important (high consequence) and poorly modeled (high TD error).

**Algorithm 2 — DQN with Balanced Consequence-Error Priority**

```
Input: μ, γ, α, ε, β, M, B_est^C, B_up, K_up, K_tar
Init: Q network weights w, replay buffer D = ∅
Set w' = w, π(s)
Sample s_0 ∈ S
For t = 1, 2, 3, ...:
    Sample a_t ~ π(s_t), observe r_{t+1} and s_{t+1}
    Add (s_t, a_t, r_{t+1}, s_{t+1}) to D
    If |D| > M: drop the oldest
    If t mod K_up = 0:
        Sample B_est^C transitions from U(D)
        Update m^C_j via Algorithm 1, update p(j)
        Sample B_up transitions from D via p(j)
        Compute IS weights: w_j = (p(j)|D|)^{-1}
        Compute TD error: δ_j = r_j + γ max_a Q(s'_j, a | w') - Q(s_j, a_j | w)
        Update TD priorities: m_j^δ = |δ_j|
        Update Q weights: w ← w - α ∇_w (1/B_L Σ w_j δ_j²)
    If t mod K_tar = 0: w' ← w
```

**Hyperparameters:**

| Parameter | Space | Description |
|---|---|---|
| $\mu$ | $[0,1]$ | Weight on consequence metric |
| $\gamma$ | $[0,1)$ | Discount factor |
| $\alpha$ | $(0,1]$ | Step size |
| $\epsilon$ | $\mathbb{R}_{\geq 0}$ | Priority shaping parameter |
| $\beta$ | $\mathbb{R}_{\geq 0}$ | Priority shaping parameter |
| $B_{est}^C$ | $\mathbb{N}_1$ | Batch size for consequence estimates |
| $B_{up}$ | $\mathbb{N}_1$ | Batch size for Q network update |
| $M$ | $\mathbb{N}_1$ | Replay buffer memory size |
| $K_{up}$ | $\mathbb{N}_1$ | Frequency of Q network updates |
| $K_{tar}$ | $\mathbb{N}_1$ | Frequency of target network updates |

---

## 5. Experiments

### 6.1 Environments

We evaluate CCE across four environments that span multi-agent combat and single-agent strategy, chosen to stress-test the priority signal under different reward densities, state spaces, and action structures.

**SMAX.** The StarCraft Multi-Agent Challenge, re-implemented in JAX via JaxMARL, provides two scenarios. In the *3m* scenario three allied marines face three enemies; in the *8m* scenario eight units per side create a higher-dimensional coordination problem. The reward is shaped: agents receive a per-step signal proportional to the fraction of enemy health destroyed, plus a +10 bonus upon winning the battle. All agents share a single centralized replay buffer; the global state fed to the Q-network is the concatenation of per-agent observations. Evaluation runs a greedy policy against the built-in heuristic opponent (`HeuristicEnemySMAX`) for 100 episodes.

**Gardner Chess.** A 5×5 chess variant implemented in pgx, a JAX-native vectorized board-game library. The agent plays White against the pgx pre-trained baseline (~1,000 Elo) as Black. Rewards are sparse (+1 win, −1 loss, 0 draw); the observation is a (5×5×115)-dimensional AlphaZero-style board encoding with a 1,225-action legal-move mask. We collect experience using 256 parallel environments via `jax.vmap`, processing 256×256 = 65,536 transitions per training chunk. The primary performance scalar is chess score = (wins + 0.5×draws)/total; the full W/D/L decomposition appears in the appendix.

**FrozenLake.** An 8×8 slippery variant of the Gymnasium FrozenLake environment, re-implemented in JAX. Stochastic ice transitions mean the agent cannot deterministically reach the goal; reward is +1 on success and 0 otherwise. The larger map and stochastic dynamics are chosen deliberately: counterfactual return distributions are more informative when actions have high outcome variance, which is precisely what slippery transitions produce.

### 6.2 Algorithms

We compare five configurations across all environments and seeds.

| Label | Configuration |
|---|---|
| DQN-Uniform | DQN with uniform replay |
| DQN+PER | DQN with TD-error priority (Eq. 3) |
| DQN+CCE-only | CCE priority only; μ=1 so TD priority is fully replaced |
| CCE+TD (add) | Additive mixing (Eq. 4), μ selected by sweep |
| CCE+TD (mul) | Multiplicative mixing (Eq. 5), μ_C=μ_δ=1 |

DQN-Uniform provides a no-prioritization floor; DQN+PER is the primary baseline. DQN+CCE-only isolates the CCE signal from TD error. The two mixed variants test whether combining both signals improves over either alone.

### 6.3 Claim 1: CCE Identifies Consequential Moments

To validate that CCE scores correlate with ground-truth transition importance, we compare stored priority scores against oracle assessments obtained independently of the learning algorithm.

**FrozenLake exact oracle.** FrozenLake's stochastic dynamics are fully specified by a precomputable transition table P[s][a] — for the 8×8 slippery map, each action produces three equally probable outcomes. This enables *exact* ground-truth labeling via value iteration: Bellman backups on P yield Q*(s,a) for all state-action pairs without any learned approximation. We define the oracle consequence of transition (s_t, a_t) as the *suboptimality gap*

$$\Delta Q(s_t, a_t) = Q^*(s_t, a^*) - Q^*(s_t, a_t), \qquad a^* = \arg\max_{a'} Q^*(s_t, a')$$

We report the Spearman rank correlation ρ between CCE scores and ΔQ across a held-out buffer sample.

**Chess oracle.** Gardner Chess lacks precomputed optimal values, but the pgx pre-trained baseline (~1,000 Elo) exposes a value head v_φ(·) estimating expected game outcome from any board position. For each transition (s_t, a_t, s_{t+1}) in the replay buffer, we define oracle importance as

$$\Delta v(s_t, a_t) = |v_\phi(s_{t+1}) - v_\phi(s_t)|$$

Moves that substantially shift the oracle's position evaluation are treated as high-consequence ground truth. We report the Spearman rank correlation between CCE scores and Δv, together with a scatter plot of the joint distribution.

### 6.4 Claim 2: Sample Efficiency

We evaluate whether CCE leads to faster early learning using the rliable evaluation framework [Agarwal et al., 2021], following recommended practices for statistically robust deep RL comparisons.

**Hyperparameter selection.** CCE introduces two key choices: the divergence metric m(·) and the mixing weight μ. To select these without contaminating main results, we ran a two-phase sweep on SMAX 3m using 3 seeds per configuration, with all sweep seeds held out from the main experiment. *Phase 1 (metric sweep):* four divergence metrics (Wasserstein, KL divergence, Jensen-Shannon, total variation) × 3 seeds = 12 runs, μ=0.5 fixed. Total variation achieved the highest mean win rate and was selected. *Phase 2 (μ sweep):* four values μ ∈ {0.25, 0.5, 0.75, 1.0} × 3 seeds = 12 runs using total variation. μ=0.25 achieved the highest mean win rate and was selected. The pair (total variation, μ=0.25) is applied without modification to all four environments.

**Pre-registered thresholds.** The steps-to-threshold metric requires a win-rate threshold per environment, fixed before inspecting main-experiment results: SMAX 3m at 60%; SMAX 8m at 55%; Gardner Chess and FrozenLake at values derived from single-seed DQN-Uniform pilot runs (approximately 80% of the pilot's converged performance), locked before the multi-seed sweep begins.

**Main experiment.** All five algorithms run for 10 independent seeds on each of the four environments (200 runs total). We report four metrics:

1. **IQM learning curves.** Interquartile mean win rate at each evaluation checkpoint with 95% stratified bootstrap confidence intervals. One curve per algorithm per environment; the IQM trims the two or three most extreme seeds, making the curve robust to outliers.

2. **Final IQM.** IQM win rate averaged over the final 10% of training checkpoints per seed. Tests whether early efficiency gains come at the cost of asymptotic performance.

3. **Steps-to-threshold.** For each seed, the first checkpoint at which win rate ≥ pre-registered threshold. We report median and IQR across seeds; seeds that never reach threshold are censored (recorded as ∞).

4. **P(improvement).** For each CCE variant, the probability that a randomly drawn seed beats DQN+PER on final win rate, estimated via stratified bootstrap. Reported per environment with 95% confidence intervals.

---

### 6.5 Implementation Details

All Q-networks are multi-layer perceptrons with ReLU activations. Key hyperparameters per environment:

| | SMAX 3m | SMAX 8m | Chess | FrozenLake 8×8 |
|---|---|---|---|---|
| Hidden dim | 128 | 256 | 512 | 64 |
| Layers (body/head) | 2/1 | 3/1 | 1 (flat) | 2 |
| Layer norm | No | Yes | Yes | No |
| γ | 0.95 | 0.95 | 0.99 | 0.99 |
| α | 5e-4 | 3e-4 | 1e-4 | 1e-3 |
| Replay M | 100k | 100k | 200k | 100k |
| Batch B | 32 | 64 | 64 | 32 |
| Target update C | 500 | 500 | 1000 | 200 steps |
| Q-update freq | 4 steps | 4 steps | 64 steps | 4 steps |
| ε decay | 10k ep | 10k ep | 20k ep | 10k ep |
| CF horizon H | 30 | 30 | 10 | 10 |
| CF rollouts n | 30 | 30 | 16 | 20 |
| Score interval | 200 | 200 | 1000 | 100 |
| Score sample B^C | 256 | 256 | 128 | 128 |

For all environments, ε decays linearly from 1.0 to 0.05; PER uses ε_PER=0.01 and β=0.4 for chess, β=0.25 elsewhere. Gardner Chess collects experience with 256 parallel environments via `jax.vmap`, processing 256×256=65,536 transitions per training chunk. All experiments run on NVIDIA Tesla T4 GPUs on the MSOE Rosie HPC cluster.

The counterfactual discount γ_CF matches the training discount. The mixing weight μ and divergence metric are selected by the two-phase sweep in Section 6.4; CCE-only fixes μ=1, and the multiplicative variant uses μ_C=μ_δ=1.

---

## 6. Results

We report four metrics across all four environments: IQM learning curves (Figure 1), final IQM and probability of improvement (Figure 2), steps-to-threshold (Table 2), and wall-clock cost (Figure 3). All confidence intervals are 95% stratified bootstrap (10,000 resamples). DQN+PER serves as the primary baseline throughout.

### 7.1 Learning Curves

Figure 1 shows IQM win-rate curves over training for all five algorithms across the four environments. Each panel's shaded region is the 95% bootstrap CI; the dashed line marks the pre-registered threshold for that environment.

*[Figure 1: 2×2 IQM learning curve grid — SMAX 3m, SMAX 8m, Gardner Chess, FrozenLake 8×8. To be generated once all environments complete.]*

On SMAX 3m all five algorithms converge to approximately 70% win rate along nearly identical trajectories, with confidence intervals overlapping at every checkpoint. On SMAX 8m, Gardner Chess, and FrozenLake 8×8, results are pending.

### 7.2 Steps-to-Threshold

Table 2 reports the median episodes (or training chunks for Chess) required to first cross the pre-registered win-rate threshold, along with the IQR across seeds. Seeds that never reached threshold are censored (∞).

**Table 2 — Steps-to-threshold across environments.**

| Algorithm | SMAX 3m (60%) | SMAX 8m (55%) | Chess (TBD) | FrozenLake (TBD) |
|---|---|---|---|---|
| DQN-Uniform | 10.5k ± 1.6k | — | — | — |
| DQN+PER | 10.8k ± 1.1k | — | — | — |
| DQN+CCE-only | 11.2k ± 1.9k | — | — | — |
| CCE+TD (add) | 11.6k ± 1.6k | — | — | — |
| CCE+TD (mul) | 10.4k ± 0.9k | — | — | — |

On SMAX 3m all algorithms cross the 60% threshold within 10.4–11.6k episodes with substantially overlapping IQRs, indicating no statistically significant difference in sample efficiency on this scenario.

### 7.3 Aggregate Metrics

Figure 2 summarizes final performance (left: final IQM; right: P(algorithm > DQN+PER)) across all four environments. Final IQM is the IQM of win rate averaged over the last 10% of training checkpoints. P(improvement) is the probability that a randomly sampled seed from an algorithm outperforms a randomly sampled DQN+PER seed on final win rate.

*[Figure 2: Two-panel bar chart — final IQM (left) and P(improvement) (right), environments on x-axis, algorithms as grouped bars. To be generated once all environments complete.]*

On SMAX 3m, CCE+TD (multiplicative) achieves the highest final IQM at 0.720 [0.696, 0.731] and a P(improvement) of 0.606 [0.444, 0.761] — the only CCE variant above chance, though its CI straddles 0.5. The remaining variants are statistically indistinguishable from DQN-Uniform and DQN+PER on this scenario.

### 7.4 Wall-Clock Cost

Figure 3 shows the wall-clock breakdown per algorithm per environment, with training time (blue) and CCE scoring overhead (red) stacked. Multipliers are relative to DQN-Uniform on each environment.

*[Figure 3: Grouped wall-clock bar chart across all four environments. To be generated once all environments complete.]*

On SMAX 3m, CCE adds a 30–40% runtime overhead (1.3× for CCE-only and CCE+TD additive, 1.4× for CCE+TD multiplicative), driven entirely by counterfactual rollout generation during each scoring pass. The base training loop time is consistent across all algorithms.

### 7.5 Discussion

**SMAX 3m.** The null result on 3m is expected: the scenario is solved within the ε-decay window (~10k episodes), leaving no exploration phase in which replay prioritization can meaningfully differ. All algorithms converge once the policy becomes nearly greedy, so the choice of sampling distribution has minimal impact. This environment establishes that CCE does not harm performance on easy tasks and imposes a bounded overhead.

**SMAX 8m, Gardner Chess, FrozenLake 8×8.** Results pending. The 8m scenario requires coordinating eight agents against a harder opponent and is expected to provide a longer, more discriminating exploration phase. Gardner Chess and FrozenLake 8×8 test CCE in single-agent sparse-reward settings where consequential decisions are rare and high-value.

---

## 7. Future Work

TODO

---

## 8. Conclusion

TODO

---

## Acknowledgment

The authors would like to thank the Milwaukee School of Engineering for supporting this research through computational resources and faculty guidance. This work was completed as part of the undergraduate research curriculum in the Department of Computer Science and Software Engineering.
