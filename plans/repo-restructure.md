# Repo Restructure Plan

Separate environments from algorithm implementations. The current `training/`
directory conflates JAX environment wrappers with DQN agent code, uses a
library name (`pgx/`) as a folder name, and misnames cross-environment
infrastructure as `smax/shared/`. This plan fixes all three.

## Why we are doing this

- `training/pgx/dqn_jax/chess_env.py` — an environment living inside training
- `training/smax/shared/` — used by chess too, not SMAX-specific
- `training/pgx/` — pgx is a library, not a meaningful folder name
- Adding FrozenLake as a third environment makes the mess worse without a fix

---

## Target Structure

```
src/counterfactual_rl/
  envs/                          ← NEW: JAX environment wrappers
    chess.py                     ← moved from training/pgx/dqn_jax/chess_env.py
    frozen_lake.py               ← new JAX FrozenLake (built separately)

  agents/                        ← RENAMED from training/
    shared/                      ← PROMOTED from training/smax/shared/ (truly cross-env)
      buffers.py                 ← PrioritizedReplayBuffer
      consequence_buffers.py     ← moved from training/smax/dqn_jax/
      consequence_diagnostics.py
      metrics.py                 ← MetricsLogger + TrainingTimer
      timing/
    smax/                        ← RENAMED from training/smax/dqn_jax/
      dqn.py
      consequence_dqn.py
      policies.py
      config.py
      utils.py                   ← SMAX-specific env helpers (from smax/shared/)
      experiments.py             ← SMAX experiment definitions
      run_experiments.py
      train.py
    chess/                       ← RENAMED from training/pgx/dqn_jax/
      dqn.py
      consequence_dqn.py
      policies.py
      config.py
      train.py
      generate_gif.py
    frozen_lake/                 ← NEW (built separately after restructure)
      dqn.py
      consequence_dqn.py
      policies.py
      config.py
      train.py

  validation/                    ← NEW: Claim 1 validation metrics
    metrics.py                   ← spearman, precision@k, sampling KL
    oracle_labels.py             ← chess oracle + FrozenLake value iteration

  environments/                  ← KEEP AS-IS: legacy Gymnasium pipeline
  analysis/                      ← KEEP AS-IS
  visualization/                 ← KEEP AS-IS
  utils/                         ← KEEP AS-IS
```

---

## What Moves Where

### Deleted
- `training/smax/dqn/` — old PyTorch implementation, dead code, broken imports
  (`dqn_pytorch` module it imports does not exist)

### Moved
| From | To |
|------|----|
| `training/pgx/dqn_jax/chess_env.py` | `envs/chess.py` |
| `training/smax/dqn_jax/consequence_buffers.py` | `agents/shared/consequence_buffers.py` |
| `training/smax/shared/buffers.py` | `agents/shared/buffers.py` |
| `training/smax/shared/consequence_diagnostics.py` | `agents/shared/consequence_diagnostics.py` |
| `training/smax/shared/metrics.py` | `agents/shared/metrics.py` |
| `training/smax/shared/timing/` | `agents/shared/timing/` |
| `training/smax/shared/config.py` | `agents/smax/config.py` |
| `training/smax/shared/utils.py` | `agents/smax/utils.py` |
| `training/smax/shared/experiments.py` | `agents/smax/experiments.py` |
| `training/smax/shared/run_experiments.py` | `agents/smax/run_experiments.py` |
| `training/smax/shared/train.py` | `agents/smax/train.py` |
| `training/smax/dqn_jax/dqn.py` | `agents/smax/dqn.py` |
| `training/smax/dqn_jax/consequence_dqn.py` | `agents/smax/consequence_dqn.py` |
| `training/smax/dqn_jax/policies.py` | `agents/smax/policies.py` |
| `training/pgx/dqn_jax/dqn.py` | `agents/chess/dqn.py` |
| `training/pgx/dqn_jax/consequence_dqn.py` | `agents/chess/consequence_dqn.py` |
| `training/pgx/dqn_jax/policies.py` | `agents/chess/policies.py` |
| `training/pgx/dqn_jax/config.py` | `agents/chess/config.py` |
| `training/pgx/dqn_jax/train.py` | `agents/chess/train.py` |
| `training/pgx/dqn_jax/generate_gif.py` | `agents/chess/generate_gif.py` |

### Kept as-is
- `training/smax/shared/summarize_experiment.py`
  → `agents/smax/summarize_experiment.py`
- `training/smax/shared/evals/` → `agents/smax/evals/`
- `training/smax/shared/train_smax_dqn.sh` → `agents/smax/train_smax_dqn.sh`

---

## Import Changes Required

### High priority (active code paths)

**`analysis/chess_counterfactual.py`**
```python
# before
from counterfactual_rl.training.pgx.dqn_jax.chess_env import GardnerChessEnv, CHESS_ACTIONS
# after
from counterfactual_rl.envs.chess import GardnerChessEnv, CHESS_ACTIONS
```

**`simulations/chess_baseline_analysis.py`**
```python
# before
from counterfactual_rl.training.pgx.dqn_jax.chess_env import GardnerChessEnv
# after
from counterfactual_rl.envs.chess import GardnerChessEnv
```

**`agents/chess/dqn.py`** (relative imports)
```python
# before
from ...smax.shared.buffers import PrioritizedReplayBuffer
from ...smax.shared.metrics import MetricsLogger
from .chess_env import GardnerChessEnv, CHESS_ACTIONS
# after
from ..shared.buffers import PrioritizedReplayBuffer
from ..shared.metrics import MetricsLogger
from counterfactual_rl.envs.chess import GardnerChessEnv, CHESS_ACTIONS
```

**`agents/chess/consequence_dqn.py`**
```python
# before
from ...smax.dqn_jax.consequence_buffers import ConsequenceReplayBuffer
from ...smax.shared.consequence_diagnostics import ConsequenceDiagnostics
from ...smax.shared.metrics import MetricsLogger
# after
from ..shared.consequence_buffers import ConsequenceReplayBuffer
from ..shared.consequence_diagnostics import ConsequenceDiagnostics
from ..shared.metrics import MetricsLogger
```

**`agents/smax/consequence_dqn.py`** (relative paths shift)
```python
# before
from ..shared.utils import evaluate, get_global_state, ...
# after
from .utils import evaluate, get_global_state, ...
```

**`agents/smax/dqn.py`** (relative paths shift similarly)
```python
# before  
from ..shared.config import DEFAULT_CONFIG
from ..shared.utils import create_smax_env, ...
# after
from .config import DEFAULT_CONFIG
from .utils import create_smax_env, ...
# shared buffers/metrics stay at ..shared.*
```

**`counterfactual_rl/__init__.py`**
```python
# before
from counterfactual_rl import training
# after
from counterfactual_rl import agents
```

**`training/__init__.py`** → becomes **`agents/__init__.py`**
```python
# before
from . import pgx
from . import smax
# after
from . import chess
from . import smax
```

**`tests/test_consequence_buffer.py`**
```python
# before
from counterfactual_rl.training.smax.dqn_jax.consequence_buffers import ConsequenceReplayBuffer
# after
from counterfactual_rl.agents.shared.consequence_buffers import ConsequenceReplayBuffer
```

### Pre-existing broken imports (fix during restructure)

**`training/smax/shared/train.py`** and **`training/smax/shared/visualize.py`**
- Both import `..dqn_pytorch.dqn` which does not exist
- Fix: remove the dqn_pytorch import lines; the JAX versions are the only active ones

**`tests/test_registry.py`**
- Imports `from counterfactual_rl.visualization.base import EnvironmentVisualizer`
- `visualization/base.py` does not exist
- Fix: remove or skip this test

**`tests/test_state_manager.py`**
- Imports `StateManagerFactory` which was removed
- Fix: remove or update this test

---

## Execution Order

Do this in phases so the repo stays runnable between steps:

**Phase 1 — Create new directories, copy files (no deletes yet)**
Create `envs/`, `agents/`, `agents/shared/`, `agents/smax/`, `agents/chess/`,
`validation/`. Copy (not move) all files to new locations.

**Phase 2 — Update all imports in new locations**
Update relative and absolute imports in every copied file. Verify nothing is
broken by running: `python -c "import counterfactual_rl"`.

**Phase 3 — Update external references**
Update `analysis/chess_counterfactual.py`, `simulations/chess_baseline_analysis.py`,
`__init__.py` files, tests.

**Phase 4 — Fix pre-existing broken imports**
Remove `dqn_pytorch` references, fix broken tests.

**Phase 5 — Delete old structure**
Remove `training/` once all imports verified working.

---

## What is NOT changing

- `environments/` (legacy Gymnasium pipeline) — kept as-is
- `analysis/` — no changes
- `visualization/` — no changes
- `utils/` — no changes
- `simulations/` — only `chess_baseline_analysis.py` needs one import update
