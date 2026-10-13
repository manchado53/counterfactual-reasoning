# Migration trust gates

Before PR #30 leaves draft, make the new experiment layout prove that it can be
installed, tested, and used outside the migration workspace. Add a small GitHub
CI gate, run the clean test suite, and run one bounded smoke through each of the
nine environment adapters. Re-run the retained JaxNav parity checks and record
their exact result without turning the known full-size GPU mismatch into a pass.
Keep the detailed migration audit reachable through normal Git history, then
push the gate and its evidence to the existing migration branch.
