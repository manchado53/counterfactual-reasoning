"""CCE suitability predictor — cheap rollout-based metrics that forecast/debug whether
CCE replay-prioritization will help in an environment.

Metric definitions and the GAIN×NEED framing live in docs/SUITABILITY_METRICS.md.
v1 = Option A (rollout-only, no trainer edits), FrozenLake first.
"""
