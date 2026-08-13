"""
Plan step 3 — falsify u. Does prioritising by measured utility beat uniform?

Everything in steps 1, 2 and 4 rests on one unproven claim: that u, the
directional derivative <grad E, grad loss_i> against exact Q*, deserves the name
"replay utility". If a sampler that draws proportional to measured u cannot beat
uniform replay in a real training run, then u is just a number, the covariances
computed against it mean nothing, and the rest of the analysis is decoration.

This is deliberately the cheapest way to kill the whole line of work, and it runs
before the expensive sweep.

Arms
----
  uniform        DQN with uniform replay                       -- the baseline
  utility-b1.0   p proportional to (u + eps)^1.0               -- the actual test
  utility-b0.25  p proportional to (u + eps)^0.25              -- deployed shaping

Why two exponents: step 1 established that beta = 0.25 caps the spread between
any two transitions at ((1+eps)/eps)^0.25 = 3.17x, which leaves the realised
sampler at 79-99% of uniform. Racing u only at the deployed beta would risk a
false negative -- u could be perfect and still fail to move the needle because
the shaping crushed it. So arm B tests u itself, and arm C vs B measures what
the deployed exponent costs. That second comparison is the same open question
sitting with Jeremy about whether beta was chosen or inherited.

u is an oracle signal: it needs exact Q*, which only exists because FrozenLake's
dynamics are fully known. This is not a deployable algorithm. It is a ceiling --
the best any utility-tracking priority could do -- and the point is whether that
ceiling is above uniform at all.

Usage
-----
    python -m counterfactual_rl.analysis.theorem3.utility_sampler --arm uniform --seed 0
    python -m counterfactual_rl.analysis.theorem3.utility_sampler --submit --dry-run
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp

from counterfactual_rl.agents.frozen_lake.consequence_dqn_vectorized import (
    FrozenLakeConsequenceDQNVectorized,
)
from counterfactual_rl.analysis.theorem3.predicate import q_star

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
LOG_DIR = os.path.expanduser("~/theorem3_logs")

ARMS = {
    # name           beta   note
    "uniform":       None,   # plain uniform replay, no priority at all
    "utility-b1.0":  1.0,
    "utility-b0.25": 0.25,
}

# Matches the graded-slip sweep exactly, because that is the configuration under
# which dqn-uniform is known to solve deterministic FrozenLake (14 of 30 seeds).
# An earlier attempt halved the budget to 8000 episodes with decay over 4000,
# keeping the same 50% ratio. That was wrong: what decides whether the agent ever
# stumbles onto the goal is how many gradient updates happen while it is still
# exploring, and halving the budget cut that from 57,344 updates to 24,576. Every
# uniform seed then finished at 0% and the baseline was useless.
BASE_CONFIG = {
    "map_name": "8x8",
    "slip_prob": 0.0,            # deterministic: where a priority effect is most visible
    "n_episodes": 15000,
    "eval_interval": 300,
    "eval_episodes": 100,
    "epsilon_decay_episodes": 7500,
    "vectorized": True,
    "n_envs": 256,
    "collect_steps": 128,
    "score_interval": 100,
    "n_score_sample": 256,
    "early_stop_win_rate": None,  # never truncate: we are comparing learning speed
    "n_checkpoints": 0,
    "priority_mixing": "additive",
    "mu": 1.0,                    # additive with mu=1 => priority is the score alone
}


class UtilitySamplerDQN(FrozenLakeConsequenceDQNVectorized):
    """Replaces the CCE score with measured replay utility u.

    Overrides only the scoring hook, so the collection loop, the update rule and
    the buffer are byte-for-byte the ones the CCE runs use. The single difference
    between this and a consequence-dqn run is what gets written into the score
    array.
    """

    def __init__(self, config=None):
        super().__init__(config)
        gamma = float(self.config.get("gamma", 0.99))
        self.Qstar, self.nt_list = q_star(self.env, gamma)
        self._nt = jnp.array(self.nt_list, dtype=jnp.int32)
        self._Qstar_nt = jnp.array(self.Qstar[np.array(self.nt_list)])
        self._u_clip_frac = []      # fraction of sampled u that came out negative

    def _global_err_grad(self):
        net = self.network

        def err(p):
            q = jax.vmap(lambda s: net.apply(p, s))(self._nt)
            return jnp.mean(jnp.abs(q - self._Qstar_nt))

        return jax.grad(err)(self.params), float(err(self.params))

    def _score_buffer_transitions(self):
        n_score = min(self.n_score_sample, len(self.buffer))
        if n_score == 0:
            return
        timer = self.metrics_logger.timer
        ep = self._current_episode

        with timer("update.scoring.sample", episode=ep):
            transitions, indices = self.buffer.sample_uniform(n_score)
            s = np.array([int(t["s"]) for t in transitions], dtype=np.int32)
            a = np.array([int(t["a"]) for t in transitions], dtype=np.int32)
            r = np.array([float(t["r"]) for t in transitions], dtype=np.float32)
            ns = np.array([int(t["s'"]) for t in transitions], dtype=np.int32)
            done = np.array([bool(t["done"]) for t in transitions])

        with timer("update.scoring.metrics", episode=ep):
            gamma = float(self.gamma)
            net = self.network
            # signed TD off the target net -- what training actually bootstraps from
            q_sa = np.array(jax.vmap(lambda x: net.apply(self.params, x))(jnp.array(s)))
            q_sa = q_sa[np.arange(len(s)), a]
            q_next = np.array(jax.vmap(lambda x: net.apply(self.target_params, x))(jnp.array(ns)))
            delta = r + gamma * np.where(done, 0.0, q_next.max(axis=1)) - q_sa

            # G_i = -2 <grad E, grad Q(s_i,a_i)>, so u_i = delta_i * G_i exactly.
            # One JVP for the whole batch; no Adam, no per-transition clones.
            grad_E, _ = self._global_err_grad()
            js, ja = jnp.array(s), jnp.array(a)

            def q_taken(p):
                q = jax.vmap(lambda x: net.apply(p, x))(js)
                return q[jnp.arange(q.shape[0]), ja]

            _, dq = jax.jvp(q_taken, (self.params,), (grad_E,))
            u = delta * (-2.0 * np.array(dq))
            u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            self._u_clip_frac.append(float(np.mean(u < 0)))
            scores = np.maximum(u, 0.0)     # priorities must be non-negative
            m = scores.max()
            if m > 0:
                scores = scores / m         # keep in [0,1] like a CCE score

        with timer("update.scoring.buffer_update", episode=ep):
            self.buffer.update_consequence_scores(indices, scores)


def build_config(arm, seed):
    cfg = dict(BASE_CONFIG)
    cfg["seed"] = seed
    beta = ARMS[arm]
    if arm == "uniform":
        cfg["algorithm"] = "dqn-uniform"
    else:
        cfg["algorithm"] = "consequence-dqn"
        per = dict(cfg.get("PER_parameters", {}))
        per.update({"eps": 0.01, "beta": beta, "maximum_priority": 1.0})
        cfg["PER_parameters"] = per
    return cfg


def run(arm, seed):
    cfg = build_config(arm, seed)
    if arm == "uniform":
        from counterfactual_rl.agents.frozen_lake.dqn_vectorized import (
            FrozenLakeDQNVectorized,
        )
        agent = FrozenLakeDQNVectorized(cfg)
    else:
        agent = UtilitySamplerDQN(cfg)
    print(f"[{arm} seed {seed}] starting, slip={agent.env.slip_prob}", flush=True)
    agent.learn()   # metrics_logger is created inside learn(), not the constructor
    out = dict(arm=arm, seed=seed, run_dir=agent.metrics_logger.dir,
               beta=ARMS[arm], slip=float(agent.env.slip_prob))
    if isinstance(agent, UtilitySamplerDQN) and agent._u_clip_frac:
        out["mean_frac_u_negative"] = float(np.mean(agent._u_clip_frac))
    print("RESULT " + json.dumps(out), flush=True)
    return out


SBATCH = """#!/bin/bash
#SBATCH --job-name=thm3_u_race
#SBATCH --output={log}/%j.out
#SBATCH --error={log}/%j.err
#SBATCH --partition=teaching
#SBATCH --account=undergrad_research
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --exclude=dh-node12,dh-node16,dh-node17,dh-node18
#SBATCH --nice=10000

export MPLBACKEND=Agg
export PYTHONPATH={src}:$PYTHONPATH
cd {repo}
{python} -m counterfactual_rl.analysis.theorem3.utility_sampler --arm {arm} --seed {seed}
"""


def submit(seeds, dry_run=True, max_concurrent=6):
    os.makedirs(LOG_DIR, exist_ok=True)
    python = os.path.expanduser("~/.conda/envs/counterfactual/bin/python")
    jobs = [(arm, s) for arm in ARMS for s in seeds]
    print(f"{len(jobs)} jobs: {len(ARMS)} arms x {len(seeds)} seeds  "
          f"(deterministic FL, {BASE_CONFIG['n_episodes']} episodes)")
    print(f"logs -> {LOG_DIR}   (no spaces in the path, per the cluster rule)")
    for arm, seed in jobs:
        script = SBATCH.format(log=LOG_DIR, src=os.path.join(_REPO, "src"),
                               repo=_REPO, python=python, arm=arm, seed=seed)
        if dry_run:
            print(f"  DRY-RUN  sbatch  arm={arm:14s} seed={seed}")
            continue
        p = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
        print(f"  {arm:14s} seed={seed}  {p.stdout.strip() or p.stderr.strip()}")
    if dry_run:
        print("\nnothing submitted. rerun with --submit --no-dry-run to launch.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = ap.parse_args()
    if args.submit:
        submit(args.seeds, dry_run=args.dry_run)
    elif args.arm:
        run(args.arm, args.seed)
    else:
        ap.error("pass --arm to run one job, or --submit to launch the race")
