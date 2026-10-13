"""Settings validation for the frozen decay protocol. No simulator imports."""
from .config import DEFAULT_CONFIG

EXTRA_KEYS = {"decay_start", "decay_end", "decay_enabled", "collector_variant", "experiment",
              "cf_action_prob_temperature", "cf_drift_probe_n", "cf_drift_probe_start_ep",
              "cf_score_chunk", "normalise_priorities", "score_on_insert"}



def validate_config(cfg):
    if cfg.get("learner_variant") == "carry-pilot-20260930":
        return validate_carry(cfg)
    if cfg.get("learner_variant") == "temporal-20260930":
        return validate_temporal(cfg)
    unknown = set(cfg) - set(DEFAULT_CONFIG) - EXTRA_KEYS
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    # Require resolved defaults: never fill missing historical settings with new defaults.
    missing = set(DEFAULT_CONFIG) - set(cfg)
    if missing:
        raise ValueError(f"Unresolved settings: {sorted(missing)}")
    if cfg["collector_variant"] not in ("old", "fixed"):
        raise ValueError("Collector mismatch")
    if cfg["decay_end"] <= cfg["decay_start"]:
        raise ValueError("Invalid decay interval")
    if cfg["algorithm"] not in ("consequence-dqn", "dqn", "dqn-uniform"):
        raise ValueError("Unsupported algorithm")
    if cfg["algorithm"] == "consequence-dqn" and cfg["normalise_priorities"] != "rank":
        raise ValueError("This scheduled learner requires rank priorities")
    if set(cfg["PER_parameters"]) != {"eps", "beta", "maximum_priority"}:
        raise ValueError("Unknown or missing PER settings")
    if cfg["priority_mixing"] not in ("additive", "multiplicative"):
        raise ValueError("Unsupported mixing rule")
    if not isinstance(cfg["decay_enabled"], bool):
        raise ValueError("decay_enabled must be a boolean")
    if cfg["batch_size"] > cfg["buffer_capacity"]:
        raise ValueError("Batch exceeds buffer capacity")
    if not cfg["vectorized"]:
        raise ValueError("Only vectorized JaxNav has been migrated")
    for key in ("n_episodes", "n_envs", "collect_steps", "batch_size", "buffer_capacity", "eval_episodes"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    return cfg


def validate_temporal(cfg):
    """Historical hold has its own RNG and no schedule/collector override."""
    extras = (EXTRA_KEYS - {"decay_start", "decay_end", "decay_enabled", "collector_variant"}) | {
        "learner_variant", "explore_hold_steps"}
    missing = set(DEFAULT_CONFIG) - set(cfg)
    unknown = set(cfg) - set(DEFAULT_CONFIG) - extras
    if missing or unknown:
        raise ValueError(f"Unresolved settings: {sorted(missing)}; unknown: {sorted(unknown)}")
    if cfg["explore_hold_steps"] not in (1, 20) or cfg["learner_variant"] != "temporal-20260930":
        raise ValueError("Unsupported historical hold protocol")
    if cfg["algorithm"] not in ("dqn", "consequence-dqn") or not cfg["vectorized"]:
        raise ValueError("Historical replay hold suite contains PER and CCE only")
    if cfg["batch_size"] > cfg["buffer_capacity"]:
        raise ValueError("Batch exceeds capacity")
    for key in ("n_episodes", "n_envs", "collect_steps", "batch_size", "buffer_capacity", "eval_episodes"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["algorithm"] == "consequence-dqn" and cfg.get("score_on_insert") is not True:
        raise ValueError("Hold suite requires full insert scoring")
    return cfg


def validate_carry(cfg):
    extra = {"blocks", "eval_every_blocks", "collector_variant", "save_step_checkpoints"}
    missing = extra - set(cfg)
    if missing:
        raise ValueError(f"Unresolved pilot settings: {sorted(missing)}")
    if cfg["collector_variant"] not in ("old", "fixed"):
        raise ValueError("Unknown pilot collector")
    for key in ("blocks", "eval_every_blocks"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["save_step_checkpoints"]) is not bool:
        raise ValueError("save_step_checkpoints must be boolean")
    temporal = {k:v for k,v in cfg.items() if k not in extra}
    temporal["learner_variant"] = "temporal-20260930"
    if cfg["algorithm"] == "dqn-uniform":
        temporal["algorithm"] = "dqn"  # validate common settings; build retains uniform
    validate_temporal(temporal)
    return cfg
