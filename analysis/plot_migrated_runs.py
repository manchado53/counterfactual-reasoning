"""Plot saved new-run logs. Equal seed weights; no interpolation or filled tails."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

METHOD_KEYS = ("algorithm", "priority_mixing", "mu", "mu_c", "mu_delta",
               "normalise_priorities", "collector_variant", "decay_enabled")


def read_runs(paths, bin_size=5000):
    if bin_size < 1:
        raise ValueError("bin_size must be positive")
    groups, reference, seen, provenance = {}, None, set(), []
    eval_identity = None
    x_field = None
    for path in paths:
        path = Path(path)
        cfg = json.loads((path/"config.json").read_text())
        attempt = json.loads((path/"attempt.json").read_text())
        axis = attempt.get("eval_axis", "episode")
        if axis not in ("episode", "env_steps") or x_field is not None and axis != x_field:
            raise ValueError("Different or unsupported training clocks")
        x_field = axis
        eval_signature = (attempt.get("eval_bank_hash"),
                          attempt.get("simulator_source_hashes", attempt.get("jaxmarl_source_hashes")),
                          attempt.get("environment", "jaxnav"))
        if not eval_signature[0] or not eval_signature[1]:
            raise ValueError("Missing eval-bank or simulator source identity")
        if eval_identity is not None and eval_signature != eval_identity:
            raise ValueError("Evaluation bank or simulator differs")
        eval_identity = eval_signature
        common = {k:v for k,v in cfg.items() if k not in (*METHOD_KEYS, "seed")}
        if reference is not None and common != reference:
            raise ValueError(f"Incompatible non-method configs: {path}")
        reference = common
        if cfg.get("fixed_eval_seed") is None:
            raise ValueError("This plotter requires explicit fixed evaluation")
        method = tuple(cfg.get(k) for k in METHOD_KEYS)
        identity = (method, cfg["seed"])
        if identity in seen:
            raise ValueError("Duplicate method/seed: choose a canonical attempt explicitly")
        seen.add(identity)
        log = path/"eval.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        last = -1
        bins = {}
        for row in rows:
            episode, score = row[x_field], row["win_rate"]
            if type(episode) is not int or episode < 0 or episode <= last or not isinstance(score, (int,float)) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid/repeated evaluation row: {path}")
            last = episode
            bins.setdefault(episode//bin_size, []).append(score)
        if not bins:
            raise ValueError(f"No evaluations: {path}")
        groups.setdefault(method, []).append({b: float(np.mean(v)) for b,v in bins.items()})
        provenance.append(dict(path=str(path.resolve()), seed=cfg["seed"],
            eval_bank_hash=attempt["eval_bank_hash"],
            attempt_sha256=hashlib.sha256((path/"attempt.json").read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256((path/"config.json").read_bytes()).hexdigest(),
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
    results = []
    for method, seeds in groups.items():
        shared = sorted(set.intersection(*(set(seed) for seed in seeds)))
        if not shared:
            raise ValueError("No shared observed bins across seeds")
        values = np.array([[seed[b] for b in shared] for seed in seeds])
        results.append(dict(method=dict(zip(METHOD_KEYS, method)), n_seeds=len(seeds),
            bin_lower_edges=[b*bin_size for b in shared], bin_width=bin_size,
            mean=values.mean(axis=0).tolist(), median=np.median(values,axis=0).tolist()))
    return dict(x_field=x_field,
                aggregation="per-seed bin means, then equal-weight mean/median; shared bins only",
                uncertainty="none", sources=provenance, curves=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bin-size", type=int, default=5000)
    args = p.parse_args()
    data = read_runs(args.runs, args.bin_size)
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite figure/data")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,5))
    for curve in data["curves"]:
        m = curve["method"]
        label = f"{m['algorithm']} {m['priority_mixing']} mu={m['mu']} {m['collector_variant']} decay={m['decay_enabled']} n={curve['n_seeds']}"
        line, = ax.plot(curve["bin_lower_edges"],curve["mean"],label=label,marker=".")
        ax.plot(curve["bin_lower_edges"],curve["median"],color=line.get_color(),linestyle="--")
    x_label = "Environment steps" if data["x_field"] == "env_steps" else "Training episode"
    ax.set(xlabel=x_label+" (bin lower edge)",ylabel="Goal reach rate",ylim=(0,1),
           title="Mean (solid), median (dashed); no filled tails")
    ax.legend(fontsize=7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output,dpi=150)
    args.output.with_suffix(".json").write_text(json.dumps(data,indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
