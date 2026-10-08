from experiments.common.provenance import project_sources


def test_run_bundle_captures_core_and_runner_as_recoverable_paths():
    sources = project_sources()
    assert "src/counterfactual_rl/replay/decay.py" in sources
    assert "experiments/run.py" in sources
    assert all(path.is_file() for path in sources.values())
    assert all(not name.startswith("/") and ".." not in name.split("/") for name in sources)
