# Approved work block: migration through FrozenLake

User approved steps 1–5 on October 8, 2026. Finish the bounded JaxNav old/new
checks, make the new path installable from a fresh checkout, reproduce historical
plot data, recover source-backed JaxNav study recipes, and prove a second path
with FrozenLake. Preserve scientific behavior and all running jobs. Use the
existing migration branch and draft PR; do not merge or retire old paths yet.

Checks cross the recipe/run, log/report, and environment reset/step/branch
interfaces from the full approved plan. Use small failing tests before each new
slice. Freeze old sources and compare against them; never generate expectations
from the new implementation. Missing source or data is a blocked check, not a pass.

The running T4 gate remains a short production-size prefix with a full 1,000-map
evaluation bank. No full training sweeps are included. Local smokes are CPU-bound,
at most four cores, 180 seconds per new smoke, and fresh output folders. Freeze
the existing H100 submission; it is not evidence of a pass. Caching, policy-based
learners, other environment ports, merges, and deletion remain out of scope.

Completion is per check, with exact evidence paths and explicit remaining gaps.
If GPU parity fails, explain it and do not switch the production training path.
