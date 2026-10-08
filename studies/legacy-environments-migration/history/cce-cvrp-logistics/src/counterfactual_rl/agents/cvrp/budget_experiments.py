"""
The BUDGET DIAL sweep — does CCE beat PER once routing has tie-able outcomes and headroom?

Background. Plain CVRP (reward = -distance) produced a flat Claim-2 null across 50 runs.
Two diagnosed causes, both properties of the reward: the task was solved by episode ~750
(no headroom), and CCE's total-variation score saturated at ~100% of states (a continuous
deterministic reward makes every action's return distinct, so TV is always 1).

Budget mode fixes both: the reward becomes "customers served within travel budget B", an
integer count, so outcomes TIE and TV grades again; and B controls difficulty directly.

THE DIAL, AND THE REGISTERED PREDICTION (written BEFORE this sweep ran).
Measured on the exact oracle, the two things CCE needs move in OPPOSITE directions:

    budget_mult   stakes gini   dead states   optimal served   DQN-uniform
       0.55          0.222         12.2%          5/10          -
       0.75          0.214         11.4%          8/10          climbing at ep 1400
       0.95          0.262         16.3%          9/10          flat from ep 400
       1.30          0.369         30.3%         10/10          CEILING at ep 400

  looser B -> stakes CONCENTRATE (more FrozenLake-like, gini 0.22 -> 0.37)
  looser B -> headroom VANISHES  (solved almost immediately)

So we predict an INVERTED U: CCE's advantage over PER should peak in the MIDDLE of the
dial, not at either end. A flat result, or a monotone one, contradicts the suitability
thesis. This is falsifiable in three distinct ways, which is the point of running it.

Capacity is swept as a SECOND, independent dial (tighter load limit -> more pivotal
reload decisions), so a real effect has to show up on two axes rather than one.

Usage:
    python -m counterfactual_rl.agents.cvrp.budget_experiments --dry-run
    python -m counterfactual_rl.agents.cvrp.budget_experiments --write-manifest <path>
"""

import argparse
import json

# Replay arms. Names match the existing Claim-2 parsers.
ARMS = {
    'uniform': {'algorithm': 'dqn-uniform'},
    'per':     {'algorithm': 'dqn'},
    'cceonly': {'algorithm': 'consequence-dqn', 'mu': 1.0, 'priority_mixing': 'additive'},
    'cceadd':  {'algorithm': 'consequence-dqn', 'mu': 0.25, 'priority_mixing': 'additive'},
    'ccemul':  {'algorithm': 'consequence-dqn', 'priority_mixing': 'multiplicative',
                'mu_c': 1.0, 'mu_delta': 1.0},
}

BUDGET_MULTS = [0.60, 0.70, 0.80, 0.90, 1.00]
CAPACITIES = [10, 6]
SEEDS = list(range(12))

# The task's learning happens in the first ~1500 episodes, so eval resolution matters
# more than training length: eval every 25 episodes. Greedy policy and env are both
# deterministic, so ONE eval episode is the whole story (20 was pure waste).
BASE = {
    'n_episodes': 4000,
    'epsilon_decay_episodes': 1500,
    'eval_interval': 25,
    'eval_episodes': 1,
    'n_checkpoints': 0,
    'save_every': 100000,       # effectively off; we only need the curves
    'score_interval': 20,
    'dist_scale': 10,
}


def build_runs():
    runs = []
    for mult in BUDGET_MULTS:
        for cap in CAPACITIES:
            for arm, arm_cfg in ARMS.items():
                for seed in SEEDS:
                    cfg = dict(BASE)
                    cfg.update(arm_cfg)
                    cfg.update({'budget_mult': mult, 'capacity': cap, 'seed': seed})
                    name = f"bd_{arm}_b{int(round(mult * 100)):03d}_c{cap}_s{seed}"
                    runs.append({'name': name, 'config': cfg})
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--write-manifest', default=None)
    args = ap.parse_args(argv)

    runs = build_runs()
    print(f"{len(runs)} runs = {len(BUDGET_MULTS)} budgets x {len(CAPACITIES)} capacities "
          f"x {len(ARMS)} arms x {len(SEEDS)} seeds")
    print(f"budgets   : {BUDGET_MULTS}")
    print(f"capacities: {CAPACITIES}")
    print(f"arms      : {list(ARMS)}")
    print(f"episodes  : {BASE['n_episodes']}  eval every {BASE['eval_interval']}")

    if args.dry_run:
        for r in runs[:6]:
            print("  ", r['name'], json.dumps(r['config'], sort_keys=True))
        print("   ... (first 6 shown)")

    if args.write_manifest:
        with open(args.write_manifest, 'w') as f:
            for r in runs:
                f.write(json.dumps(r) + "\n")
        print(f"manifest -> {args.write_manifest}")
    return runs


if __name__ == '__main__':
    main()
