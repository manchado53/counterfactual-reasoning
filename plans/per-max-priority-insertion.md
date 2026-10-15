# Standard PER insertion priority

Make the canonical shared PER buffer insert every new transition with the largest raw priority observed so far, initialized at `1.0`. This matches standard PER and replaces the misleading fixed-`1.0` behavior for new experiments.

Route current, non-historical learners through that canonical behavior. Keep frozen historical learners and fixed recipe files unchanged so old results still reproduce. Add focused tests for initialization, growth after a larger TD error, no decrease after smaller errors, uniform replay, and active learner wiring. Record the shared buffer in each new run's source hashes and run the relevant replay, learner, and smoke checks before handoff.
