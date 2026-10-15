# One learning core for every current environment

Make every current and future environment use one shared implementation for Double-DQN targets, TD errors, weighted TD loss, PER replay, CCE scoring, CCE/TD mixing, and decay. Environment modules may define networks, observations, legal-action masks, dynamics, and counterfactual rollouts, but may not copy the learning math.

Move the common TD and one-rollout weighted-Wasserstein math into `src/counterfactual_rl`. Route JaxNav, FrozenLake, DoorKey, and Connect Four through it. Add an import gate so a new current learner fails tests if it defines local replay or TD math. Keep explicitly historical learners frozen so old experiments remain reproducible.

Prove the move with exact parity fixtures for targets, TD errors, losses, priorities, CCE scores, sampled indices, and weights. Then run all tests and smoke every current environment and replay arm. No training recipe, random seed, collector, reward, or evaluation rule should change.
