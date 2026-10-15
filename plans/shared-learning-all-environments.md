# Shared learning for every environment

Make shared TD, PER, CCE scoring, mixing, and decay available to SMAX, Chess, CVRP, RP-DQN, and CVRP Budget. Keep each old historical runner frozen so past results remain reproducible. New experiment recipes will select a shared-current runner through the normal environment registry.

Use one small learner interface. Environment adapters will provide observations, legal actions, transitions, rewards, terminal flags, and counterfactual branch returns. They will not copy replay or learning math.

For each environment, first freeze parity fixtures from the old runner. Then add the shared adapter, prove targets, scores, priorities, samples, and updates match within fixed tolerances, and run a GPU smoke. Do not claim an environment migrated until its parity and smoke gates pass.
