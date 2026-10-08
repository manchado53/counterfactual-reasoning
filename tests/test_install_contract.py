"""The installed package exposes the research runner without private paths."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_exposes_research_runner_and_optional_dependencies():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["scripts"]["cce-run"] == "experiments.run:main"
    assert data["tool"]["setuptools"]["package-dir"]["experiments"] == "experiments"
    assert "research" in data["project"]["optional-dependencies"]
    assert any(dep.startswith("rliable") for dep in data["project"]["optional-dependencies"]["research"])
    assert not any("stable-baselines3" in dep for dep in data["project"]["dependencies"])
    assert data["project"]["requires-python"] == ">=3.12"
