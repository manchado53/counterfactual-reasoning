# README method demos

Refresh the root README so a new reader first sees the CCE replay idea, the
strongest verified results, and how to reproduce the project. Add honest,
side-by-side policy demos for CCE, PER, and Uniform in
deterministic FrozenLake and fixed-evaluation JaxNav.

The FrozenLake demo will use the same training seed and final checkpoints for
all three arms. The JaxNav demo will use the latest matched experiment's
CCE-only arm with the fixed collector and ranked-TD decay. It will use a
checkpoint inside the measured 100k-150k lead, where the three-seed mean is
about 10 percentage points above Uniform and farther above PER. The demo will
use the same training seed for all arms and the first maps from the fixed
1,000-map evaluation bank. It will not select maps after looking at outcomes.
Each committed GIF and MP4 will have a machine-readable provenance record with
checkpoint hashes, configuration, evaluation seed, and outcomes. A reusable
generator and tests will guard the comparison rules.

The README will distinguish the strong deterministic FrozenLake result from
the JaxNav mid-training sample-efficiency gain and the later JaxNav convergence
or collapse. It will not call the JaxNav lead a final-performance win, and it
will not present an illustrative video as statistical proof.
