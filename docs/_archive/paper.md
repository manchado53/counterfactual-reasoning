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

This paper makes three contributions. First, we introduce CCE (Algorithm 1), a Monte Carlo estimator of transition consequence applicable to any replay-based RL algorithm. Second, we propose DQN with CCE-augmented priority (Algorithm 2), with both additive and multiplicative mixing schemes. Third, we provide an empirical evaluation across three environments — FrozenLake 8×8 (deterministic and stochastic variants) and SMAX 3m — with 10 independent seeds and the rliable statistical framework [Agarwal et al., 2021], demonstrating that CCE leads to more sample-efficient early training while preserving asymptotic performance.

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

We evaluate CCE across three environments spanning stochastic single-agent and multi-agent settings, chosen to stress-test the priority signal under different levels of environment noise, state spaces, and action structures.

**SMAX 3m.** The StarCraft Multi-Agent Challenge, re-implemented in JAX via JaxMARL. Three allied marines face three enemies in a combat scenario with shaped reward: agents receive a per-step signal proportional to the fraction of enemy health destroyed, plus a +10 bonus upon winning the battle. All agents share a single centralized replay buffer; the global state fed to the Q-network is the concatenation of per-agent observations. Evaluation runs a greedy policy against the built-in heuristic opponent (`HeuristicEnemySMAX`) for 100 episodes.

**FrozenLake 8×8 (Deterministic).** An 8×8 FrozenLake environment re-implemented in JAX with the slippery flag disabled. Every action moves the agent deterministically, isolating the policy-driven consequence signal from stochastic outcome noise. Reward is +1 on reaching the goal and 0 otherwise.

**FrozenLake 8×8 (Stochastic).** The same map with stochastic ice transitions: each action has a 1/3 probability of sliding perpendicular to the intended direction. This injects outcome variance that cannot be attributed to the agent's action choice, providing an expected null condition for CCE — a controlled test of whether CCE hurts when its signal is diluted.

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

To validate that CCE scores correlate with ground-truth state importance, we compare CCE priority scores against oracle assessments obtained independently of the learning algorithm.

**FrozenLake exact oracle.** FrozenLake's stochastic dynamics are fully specified by a precomputable transition table P[s][a] — for the 8×8 slippery map, each action produces three equally probable outcomes. This enables *exact* ground-truth labeling via value iteration: Bellman backups on P yield Q*(s,a) for all state-action pairs without any learned approximation. We define the oracle importance of state s as the average expected-return cost of acting suboptimally:

$$\text{Oracle}(s) = \frac{1}{|A(s)|-1} \sum_{a \,\neq\, a^*} \left[ Q^*(s, a^*) - Q^*(s, a) \right], \qquad a^* = \arg\max_{a'} Q^*(s, a')$$

A high oracle score indicates that acting suboptimally at s is costly — the state is genuinely consequential. We compute CCE scores for all 53 non-terminal states and report the Spearman rank correlation ρ between CCE scores and Oracle(s) at three training stages: untrained (ep. 150), mid-training (ep. 3,900), and fully trained (best checkpoint).

### 6.4 Claim 2: Sample Efficiency

We evaluate whether CCE leads to faster early learning using the rliable evaluation framework [Agarwal et al., 2021], following recommended practices for statistically robust deep RL comparisons.

**Hyperparameter selection.** CCE introduces two key choices: the divergence metric m(·) and the mixing weight μ. To select these without contaminating main results, we ran a two-phase sweep on SMAX 3m using 3 seeds per configuration, with all sweep seeds held out from the main experiment. *Phase 1 (metric sweep):* four divergence metrics (Wasserstein, KL divergence, Jensen-Shannon, total variation) × 3 seeds = 12 runs, μ=0.5 fixed. Total variation achieved the highest mean win rate and was selected. *Phase 2 (μ sweep):* four values μ ∈ {0.25, 0.5, 0.75, 1.0} × 3 seeds = 12 runs using total variation. μ=0.25 achieved the highest mean win rate and was selected. The pair (total variation, μ=0.25) is applied without modification to all three environments.

**Pre-registered thresholds.** The steps-to-threshold metric requires a win-rate threshold per environment, fixed before inspecting main-experiment results: SMAX 3m at 60%; FrozenLake (deterministic and stochastic) at values derived from single-seed DQN-Uniform pilot runs (approximately 80% of the pilot's converged performance), locked before the multi-seed sweep begins.

**Main experiment.** All five algorithms run for 10 independent seeds on each of the three environments (150 runs total). We report four metrics:

1. **IQM learning curves.** Interquartile mean win rate at each evaluation checkpoint with 95% stratified bootstrap confidence intervals. One curve per algorithm per environment; the IQM trims the two or three most extreme seeds, making the curve robust to outliers.

2. **Final IQM.** IQM win rate averaged over the final 10% of training checkpoints per seed. Tests whether early efficiency gains come at the cost of asymptotic performance.

3. **Steps-to-threshold.** For each seed, the first checkpoint at which win rate ≥ pre-registered threshold. We report median and IQR across seeds; seeds that never reach threshold are censored (recorded as ∞).

4. **P(improvement).** For each CCE variant, the probability that a randomly drawn seed beats DQN+PER on final win rate, estimated via stratified bootstrap. Reported per environment with 95% confidence intervals.

---

### 6.5 Implementation Details

All Q-networks are multi-layer perceptrons with ReLU activations. Key hyperparameters per environment:

| | SMAX 3m | FrozenLake 8×8 |
|---|---|---|
| Hidden dim | 128 (2 layers, 1 head) | 64 (2 layers) |
| Layer norm | No | No |
| γ | 0.95 | 0.99 |
| α | 5e-4 | 1e-3 |
| Replay M | 100k | 100k |
| Batch B | 32 | 32 |
| Target update C | 500 steps | 200 steps |
| Q-update freq | 4 steps | every step |
| ε decay | 10k ep | 10k ep |
| CF horizon H | 30 | 10 |
| CF rollouts n | 30 | 20 |
| Score interval | 200 ep | 100 ep |
| Score sample B^C | 256 | 128 |

For all environments, ε decays linearly from 1.0 to 0.05; PER uses ε_PER=0.01 and β=0.25. All experiments run on NVIDIA Tesla T4 GPUs on the MSOE Rosie HPC cluster.

The counterfactual discount γ_CF matches the training discount. The mixing weight μ and divergence metric are selected by the two-phase sweep in Section 6.4; CCE-only fixes μ=1, and the multiplicative variant uses μ_C=μ_δ=1.

---

## 6. Results

We organize results by claim. All confidence intervals are 95% stratified bootstrap (10,000 resamples).

### 6.1 Claim 1 — CCE Identifies Consequential Moments

Claim 1 asks whether CCE scores, computed purely from rollouts under the current policy, correlate with ground-truth state importance. The oracle (value iteration on the known transition table) has no access to CCE rollouts. CCE has no access to Q\*. These two signals are completely independent.

![CCE score vs. oracle importance at three training stages (seed 0). Each point is one non-terminal state; color indicates hole proximity (red = adjacent, orange = 2 steps, blue = safe). Spearman ρ rises from 0.393 (untrained) to 0.829 (mid-training) to 0.849 (fully trained).](docs/figures/real/claim1/frozen_lake/fig_c1_scatter_stages.png)

![Spearman ρ between CCE score and oracle importance across 3 seeds and 3 training stages. Mean ρ rises from 0.319 ± 0.114 (untrained) to 0.889 ± 0.031 (fully trained); all fully-trained seeds p < 0.001.](docs/figures/real/claim1/frozen_lake/fig_c1_rho_table.png)

The scatter plots and Spearman ρ table tell the same story. At episode 150 the agent is effectively untrained and the correlation is moderate (ρ = 0.319 ± 0.114); one seed is not yet statistically significant (p = 0.258). By mid-training (ep. 3,900) the policy has converged enough for CCE rollouts to yield informative return distributions, and ρ jumps to 0.765 ± 0.096 (all seeds p < 0.001). At the best checkpoint the correlation reaches **ρ = 0.889 ± 0.031** — all three seeds agree tightly and all p-values are below 0.001.

![State importance heatmaps on the 8×8 FrozenLake grid. Left: Oracle(s) from value iteration — hole-adjacent cells score highest. Center: CCE scores at episode 150 (untrained) — no clear structure. Right: CCE scores at the best checkpoint — the same hole-adjacent hot spots emerge from rollouts alone.](docs/figures/real/claim1/frozen_lake/fig_c2_grid_heatmaps.png)

The heatmaps make the correlation spatial. The oracle assigns high importance to states adjacent to holes, where acting suboptimally sends the agent to a terminal failure. Untrained CCE is noise. Trained CCE recovers the same spatial structure — identifying dangerous cells without ever being told where the holes are.

![Precision@K: fraction of the oracle's top-K% states that are also in CCE's top-K%, for the fully-trained policy (3 seeds, mean ± std). CCE achieves 3.3× random at K=5%, 6.0× at K=10%, and 3.7× at K=20%.](docs/figures/real/claim1/frozen_lake/fig_c4_precision_at_k.png)

Precision@K confirms that CCE is not just rank-correlated but identifies the *right* states in a practical sense. At K=10%, CCE achieves 0.60 precision — 6.0× the 0.10 expected from random selection. The K=5% result is weaker (3.3×) with higher variance, reflecting the small absolute count (only 2–3 states) at that threshold. Taken together, these results confirm that CCE, using only rollouts under the current policy, consistently identifies the states that are most consequential according to the exact optimal value function.

### 6.2 Claim 2 — Sample Efficiency

We evaluate sample efficiency across three environments spanning deterministic single-agent, stochastic single-agent, and stochastic multi-agent settings. All five algorithms run for 10 independent seeds per environment. We report IQM learning curves, final IQM, P(algorithm beats DQN+PER) over training, and steps-to-threshold using the rliable framework [Agarwal et al., 2021]. DQN+PER is the primary baseline throughout.

---

#### 6.2.1 FrozenLake 8×8 (Deterministic)

Deterministic ice removes all stochastic outcome noise, so every difference in return distribution across counterfactual actions is purely policy-driven. This is the cleanest setting for CCE.

![IQM win-rate learning curves, FrozenLake 8×8 deterministic (10 seeds). CCE+TD (multiplicative) and CCE+TD (additive) both pull clearly ahead of DQN+PER from mid-training onward.](docs/figures/real/claim2/FL_deterministic/fig1_iqm_frozen_lake_no_slip.png)

![Final IQM win rate, FrozenLake deterministic. CCE+TD (mul) ≈ 1.0; CCE+TD (add) ≈ 0.80; DQN+PER ≈ 0.45; DQN-Uniform ≈ 0.25.](docs/figures/real/claim2/FL_deterministic/fig2_final_iqm_frozen_lake_no_slip.png)

![P(algorithm > DQN+PER) over training, FrozenLake deterministic. CCE+TD (mul) crosses above 0.5 around 300k env steps and remains there for the rest of training.](docs/figures/real/claim2/FL_deterministic/fig4b_prob_improve_curves_frozen_lake_no_slip.png)

![Steps-to-threshold, FrozenLake deterministic. DQN+PER reaches threshold fastest (328k) but only to its low performance ceiling. CCE+TD (mul) and (add) reach it at 360k and 377k steps respectively.](docs/figures/real/claim2/FL_deterministic/fig3_steps_thresh_frozen_lake_no_slip.png)

CCE+TD (multiplicative) reaches a final IQM of ≈1.0 versus DQN+PER at ≈0.45 — more than doubling asymptotic performance. CCE+TD (additive) reaches ≈0.80. The P(improve) curve confirms this is not a convergence artifact: CCE+TD (mul) crosses above 0.5 around 300k env steps and holds there for the rest of training. DQN+PER reaches threshold fastest (328k steps) only because the threshold is set relative to its low ceiling; CCE variants take marginally longer to cross that modest goal and then far surpass it.

---

#### 6.2.2 FrozenLake 8×8 (Stochastic)

The slippery variant adds stochastic ice transitions (1/3 probability of sliding perpendicular to the intended direction), injecting outcome variance that CCE cannot distinguish from policy-driven consequence.

![IQM win-rate learning curves, FrozenLake 8×8 stochastic. All five algorithms overlap throughout training with no method dominating.](docs/figures/real/claim2/frozen_lake/fig1_iqm_frozen_lake.png)

![P(algorithm > DQN+PER) over training, FrozenLake stochastic. All CCE variants oscillate around the 0.5 chance line throughout training.](docs/figures/real/claim2/frozen_lake/fig4b_prob_improve_curves_frozen_lake.png)

The stochastic environment produces a null result: all five algorithms achieve similar final IQM (0.60–0.67) with heavily overlapping CIs, and P(CCE > DQN+PER) oscillates around 0.5 throughout training for every CCE variant. This is expected — stochastic ice transitions dilute the CCE priority signal. Critically, CCE does *not* harm performance: no CCE variant is worse than DQN+PER.

---

#### 6.2.3 SMAX 3m

SMAX 3m is a multi-agent combat scenario (three allied marines versus three enemies, JaxMARL). Reward is shaped plus a win bonus. The environment is substantially noisier than FrozenLake due to opponent stochasticity and the higher branching factor of multi-agent joint actions.

![IQM win-rate learning curves, SMAX 3m (10 seeds). Curves overlap throughout training; differences emerge only in the converged regime.](docs/figures/real/smax_3m/fig1_iqm_smax_3m.png)

![Final IQM win rate, SMAX 3m. CCE+TD (mul) leads at ≈0.72; all other algorithms cluster at 0.65–0.68.](docs/figures/real/smax_3m/fig2_final_iqm_smax_3m.png)

![P(algorithm > DQN+PER) over training, SMAX 3m. All curves oscillate 0.2–0.8 due to environment noise — contrast with FL deterministic where mul separates cleanly above 0.5.](docs/figures/real/smax_3m/fig4b_prob_improve_curves_smax_3m.png)

![Steps-to-threshold (60% win rate), SMAX 3m. All algorithms cluster in the 10–12k episode range with overlapping IQRs.](docs/figures/real/smax_3m/fig3_steps_thresh_smax_3m.png)

CCE+TD (multiplicative) leads at final IQM ≈0.72 versus ≈0.68 for DQN+PER. Notably, CCE+TD (additive) falls *below* DQN+PER — blending signals additively admits enough noise to hurt. Multiplicative mixing, which requires a transition to score high on *both* consequence and TD error, filters that noise. The P(improve) curves oscillate wildly throughout training, which contrasts directly with FL deterministic (where mul cleanly separates above 0.5) and illustrates why deterministic dynamics make the CCE signal easier to exploit. Steps-to-threshold shows no meaningful difference across algorithms.

---

#### 6.2.4 Computational Overhead

![Wall-clock cost per algorithm, FrozenLake deterministic. CCE+TD variants add 2.2–3.3× the cost of DQN-Uniform; the IQM gain (0.45 → 1.0) justifies this in deterministic settings.](docs/figures/real/claim2/FL_deterministic/fig5a_wallclock_frozen_lake_no_slip.png)

![Wall-clock cost per algorithm, SMAX 3m. CCE adds only 1.3–1.4× overhead; larger per-step environment cost dilutes the relative scoring overhead.](docs/figures/real/smax_3m/fig5a_wallclock_smax_3m.png)

CCE+TD methods cost 2.2–3.3× more than DQN-Uniform in FrozenLake deterministic, and only 1.3–1.4× more in SMAX 3m (where larger per-step rollout cost dilutes the scoring overhead). In the deterministic setting the overhead is clearly justified: doubling performance at 2–3× compute is a favourable trade. In SMAX, the modest performance gain and modest overhead make the cost-benefit ratio environment-dependent.

---

## 7. Future Work

TODO

---

## 8. Conclusion

TODO

---

## Acknowledgment

The authors would like to thank the Milwaukee School of Engineering for supporting this research through computational resources and faculty guidance. This work was completed as part of the undergraduate research curriculum in the Department of Computer Science and Software Engineering.
