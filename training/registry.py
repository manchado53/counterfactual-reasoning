"""Lazy run adapters. Validation never loads a simulator or accelerator."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path


CURRENT_ENVIRONMENTS = ("jaxnav", "frozenlake", "doorkey", "connect_four", "four_rooms")
HISTORICAL_ENVIRONMENTS = ("smax", "chess", "cvrp", "rpdqn", "cvrp_budget")
ENVIRONMENTS = CURRENT_ENVIRONMENTS + HISTORICAL_ENVIRONMENTS
SHARED_LEARNING_ENVIRONMENTS = ENVIRONMENTS


@dataclass(frozen=True)
class Environment:
    name: str

    def module(self):
        return import_module(f"training.learners.{self.name}.build")

    def validate_config(self, config):
        if self.name == "jaxnav":
            from training.learners.jaxnav.protocol import validate_config
            return validate_config(config)
        return self.module().validate_config(config)

    def build_agent(self, config):
        return self.module().build_agent(config)

    def smoke_overrides(self, config):
        if self.name == "jaxnav":
            from training.smoke import JAXNAV_OVERRIDES
            overrides = dict(JAXNAV_OVERRIDES)
            if config.get("learner_variant") in ("temporal-20260930", "carry-pilot-20260930"):
                for key in ("decay_start", "decay_end"):
                    overrides.pop(key)
            if config.get("learner_variant") == "carry-pilot-20260930":
                overrides.update(blocks=4, eval_every_blocks=2, save_step_checkpoints=False)
            return overrides
        return self.module().smoke_overrides(config)

    def evaluation_axis(self, config):
        if self.name == "jaxnav":
            return "env_steps" if config.get("learner_variant") == "carry-pilot-20260930" else "episode"
        if self.name == "frozenlake":
            return "env_steps" if config["collector_variant"] == "fixed" else "episode"
        return self.module().evaluation_axis(config)

    def simulator_sources(self, config):
        if self.name == "jaxnav":
            import jaxmarl
            root = Path(jaxmarl.__file__).resolve().parent
            return {str(f.relative_to(root)): f for f in root.rglob("*.py")}
        if self.name == "frozenlake":
            from training.envs import frozenlake
            return {"frozenlake.py": Path(frozenlake.__file__).resolve()}
        return self.module().simulator_sources(config)


def get_environment(name):
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unsupported environment: {name!r}")
    return Environment(name)
