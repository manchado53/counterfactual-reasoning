"""Frozen source plus real CPU training checks for the hold-exploration port."""
import ast
import hashlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "studies/jaxnav-historical/sources/hold"
RECIPE = ROOT / "configs/experiment/jaxnav-temporal-20260930.json"


def test_hold_sources_and_all_forty_recipes():
    from experiments.recipes import load_recipe
    hashes = json.loads((SOURCE.parent.parent / "hold-source-hashes.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
    for row in json.loads(RECIPE.read_text())["rows"]:
        assert load_recipe(RECIPE, row["task"]) == row


def test_extracted_collector_matches_frozen_class():
    old = ast.parse((SOURCE / "runtime/scripts/train_temporal_replay.py").read_text())
    new = ast.parse((ROOT / "experiments/learners/jaxnav/temporal.py").read_text())
    def cls(tree):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "TemporalAgent")
    assert ast.dump(cls(old)) == ast.dump(cls(new))


def test_hold_parent_code_only_changes_imports_output_path_and_buffer_hook():
    replacements = {
        "from ..shared.buffers": "from counterfactual_rl.replay.buffers",
        "from ..shared.consequence_buffers": "from counterfactual_rl.replay.consequence_buffers",
        "from ..shared.early_stopping": "from experiments.common.early_stopping",
        "from ..shared.timing": "from experiments.common.timing",
        "from counterfactual_rl.envs.jax_nav": "from experiments.envs.jaxnav",
        "from counterfactual_rl.analysis.metrics": "from counterfactual_rl.scoring.metrics",
        "run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', job_id)":
            "run_dir = os.environ['CCE_RUN_DIR']",
        "self.buffer = ConsequenceReplayBuffer(":
            "self.buffer = getattr(self, 'replay_buffer_class', ConsequenceReplayBuffer)(",
    }
    mapping = {
        "agents/jax_nav/dqn.py": "experiments/learners/jaxnav/dqn.py",
        "agents/jax_nav/consequence_dqn.py": "experiments/learners/jaxnav/consequence_dqn.py",
        "agents/jax_nav/dqn_vectorized.py": "experiments/learners/jaxnav/dqn_vectorized_legacy.py",
        "agents/jax_nav/consequence_dqn_vectorized.py": "experiments/learners/jaxnav/consequence_dqn_vectorized_legacy.py",
        "agents/jax_nav/config.py": "experiments/learners/jaxnav/config.py",
        "agents/jax_nav/qdiag.py": "experiments/learners/jaxnav/qdiag.py",
        "agents/shared/buffers.py": "src/counterfactual_rl/replay/buffers.py",
        "agents/shared/consequence_buffers.py": "src/counterfactual_rl/replay/consequence_buffers.py",
        "agents/shared/early_stopping.py": "experiments/common/early_stopping.py",
        "analysis/metrics.py": "src/counterfactual_rl/scoring/metrics.py",
        "envs/jax_nav.py": "experiments/envs/jaxnav.py",
    }
    for old, new in mapping.items():
        code = (SOURCE / "runtime/src/counterfactual_rl" / old).read_text()
        for a, b in replacements.items():
            code = code.replace(a, b)
        if old.endswith("consequence_dqn_vectorized.py"):
            code = code.replace("from .dqn_vectorized import", "from .dqn_vectorized_legacy import")
        assert ast.dump(ast.parse(code)) == ast.dump(ast.parse((ROOT/new).read_text())), old


@pytest.mark.parametrize("task", [0, 5, 10, 15, 20, 25, 30, 35])
def test_hold_training_matches_frozen_class(tmp_path, monkeypatch, task):
    # The source equality check above proves shared parents are the same program.
    # Build the old nested class directly from its saved source, independently of
    # the new class extraction and recipe dispatch.
    import os
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import time
    from experiments.registry import get_environment
    from experiments.recipes import load_recipe
    from experiments.learners.jaxnav.dqn_vectorized_legacy import JaxNavDQNVectorized, _tree_where
    from experiments.learners.jaxnav.consequence_dqn_vectorized_legacy import JaxNavConsequenceDQNVectorized
    adapter = get_environment("jaxnav")
    cfg = load_recipe(RECIPE, task)["config"]
    cfg = dict(cfg, **adapter.smoke_overrides(cfg))
    cce = cfg["algorithm"] == "consequence-dqn"
    Parent = JaxNavConsequenceDQNVectorized if cce else JaxNavDQNVectorized
    old = ast.parse((SOURCE / "runtime/scripts/train_temporal_replay.py").read_text())
    node = next(n for n in ast.walk(old) if isinstance(n, ast.ClassDef) and n.name == "TemporalAgent")
    scope = dict(Parent=Parent, cce=cce, prefix=9 if cce else 8, jax=jax, jnp=jnp,
                 np=np, _tree_where=_tree_where, time=time, Path=Path, json=json)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "frozen-hold", "exec"), scope)
    traces = []
    for name, factory in [("old", scope["TemporalAgent"]), ("new", adapter.build_agent)]:
        out = tmp_path / name
        monkeypatch.setenv("CCE_RUN_DIR", str(out))
        agent = factory(dict(cfg))
        assert agent.config == cfg
        agent.learn(verbose=False)
        leaves = jax.tree.leaves((agent.params, agent.target_params, agent.opt_state, agent._key))
        traces.append(([np.asarray(x).copy() for x in leaves],
                       (out / "eval.jsonl").read_text(),
                       agent.buffer.consequence_scores.copy() if cce else None))
    for a, b in zip(traces[0][0], traces[1][0]):
        np.testing.assert_array_equal(a, b)
    assert traces[0][1] == traces[1][1]
    if cce:
        np.testing.assert_array_equal(traces[0][2], traces[1][2])
