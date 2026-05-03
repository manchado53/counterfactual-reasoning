# Counterfactual Reasoning Research

## Overview
Undergraduate research project on counterfactual reasoning in multi-agent reinforcement learning, using the SMAX environment from JaxMARL.

## Project Structure
- Main research directory: `~/UR-RL/counterfactual-reasoning/`
- SMAX examples: `~/UR-RL/SmaxExample.py`, `~/UR-RL/SMAX.SH`
- Related SMAC work: `~/UR-RL/playing-with-smac/`
- Transition notes: `~/UR-RL/transition-smax.md`
- Key docs: `~/UR-RL/counterfactual-reasoning/docs/`

## SMAX / JaxMARL
When working with SMAX, JaxMARL, or related multi-agent RL environments:
1. Use the WebFetch tool to check https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax for relevant documentation
2. Follow any links on that page pertinent to the specific question
3. Base answers on the fetched documentation rather than general knowledge

## Tech Stack
- JAX, JaxMARL, SMAX environment
- Multi-agent reinforcement learning
- Counterfactual reasoning and explainability methods

## Gardner Chess (pgx) — Planned Environment
We are adding Gardner chess (5×5 chess variant) as a second environment to demonstrate the counterfactual consequence reasoning algorithm beyond SMAX.

**Library**: `pgx` — JAX-native, vectorized board game environments (NeurIPS 2023)
**Environment name**: `"gardner_chess"` via `pgx.make("gardner_chess")`

When working with pgx or Gardner chess:
1. Use the WebFetch tool to check https://www.sotets.uk/pgx/gardner_chess/ for the environment spec
2. Check https://www.sotets.uk/pgx/api/ for the full API reference
3. Check https://www.sotets.uk/pgx/api_usage/ for usage patterns and vmap examples
4. Base answers on the fetched documentation rather than general knowledge

**Key facts (verify against docs before using):**
- Observation shape: `(5, 5, 115)` float32 — AlphaZero-style board representation
- Action space: 1,225 discrete actions (`legal_action_mask` is a `(1225,)` bool on the state)
- Rewards: sparse ±1 at game end only, `state.rewards` shape `(2,)` one per player
- Perspective: observations auto-flip for the current player — pgx handles this internally
- Pre-trained opponent: `pgx.make_baseline_model("gardner_chess_v0")` (~1000 Elo AlphaZero baseline)
- Vectorized API: `jax.vmap(env.init)`, `jax.vmap(env.step)` — no key needed for step (deterministic)
- Open issue #1174: pawn move behavior under investigation — check before relying on pawn logic

## Cluster (Rosie / SLURM)
- Check running jobs: `squeue -u $USER`
- **After every `sbatch` submission, immediately start a Monitor** on the job's `.out` log file without waiting to be asked. Use a grep filter that catches: training progress, eval metrics, errors, and completion. Keep it persistent so it runs for the life of the session.
