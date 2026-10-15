# Shared replay for current experiments

Make the canonical shared PER buffer insert every new transition with the largest raw priority observed so far, initialized at `1.0`. This matches standard PER and replaces the misleading fixed-`1.0` behavior for new experiments.

Route every current, non-historical JaxNav, FrozenLake, DoorKey, and Connect Four replay arm through the canonical PER and CCE buffers. Keep frozen source copies and historical-only SMAX, Chess, CVRP, CVRP Budget, and RP-DQN protocols unchanged. Old completed runs remain recoverable from their saved source bundles; new runs hash and bundle the canonical replay source.

Guard the move with replay-operation parity tests, live-import tests, recipe validation, checkpoint/source-manifest checks, and DoorKey plus Connect Four smokes. The only intended behavior change is the new PER max-observed insertion rule. CCE output, ranks, samples, weights, and diagnostics must stay the same.
