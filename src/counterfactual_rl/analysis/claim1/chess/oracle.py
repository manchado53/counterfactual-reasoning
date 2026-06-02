"""
AlphaZero value-head oracle for Gardner chess Claim 1.

Oracle label per position:
    oracle_score(s) = mean_{a ≠ a_chosen} |v_chosen - v_a|

where v_a = -value_head(step(s, a).observation) (negated: pgx returns
current player's perspective; after white moves it's black's turn, so negate
to get white's value).

a_chosen = white's chosen action (record.action, already the baseline's greedy pick).

Implementation: batched. All next-state observations are collected in one pass
using vmap over legal actions per position, then the value head is called in
chunks of `chunk_size` to avoid GPU OOM.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List


def compute_oracle_for_records(
    records,
    pgx_env,
    baseline_model,
    chunk_size: int = 1024,
) -> List[float]:
    """
    Compute oracle importance score for each ChessConsequenceRecord.

    Requires store_states=True in ChessCounterfactualAnalyzer so that
    record.pgx_state is populated.

    Parameters
    ----------
    records        : List[ChessConsequenceRecord]
    pgx_env        : raw pgx Gardner chess env (pgx.make("gardner_chess"))
    baseline_model : pgx baseline model (pgx.make_baseline_model("gardner_chess_v0"))
    chunk_size     : model call batch size (lower if OOM)

    Returns
    -------
    list[float] — oracle score per record (same order as input)
    """
    # --- Phase 1: collect all next-state observations ---
    # Use jax.jit(pgx_env.step) in a plain Python loop — traces once on the first
    # call (fixed state shape + int32 scalar action), then runs in microseconds.
    # vmap with variable K per position causes repeated retracing (one per unique K),
    # which is much slower than a single jit trace reused 8,200 times.
    step_jit = jax.jit(pgx_env.step)

    all_obs = []     # list of (K_i, 5, 5, 115) arrays
    all_legal = []   # list of (K_i,) int32 action arrays
    all_chosen = []  # chosen action per position

    for i, record in enumerate(records):
        state = record.pgx_state
        if state is None:
            raise ValueError(
                f"record {i} has pgx_state=None; "
                "use store_states=True in ChessCounterfactualAnalyzer"
            )
        legal_mask = np.array(state.legal_action_mask)
        legal_actions = np.where(legal_mask)[0]

        next_obs_i = []
        for a in legal_actions:
            s_next = step_jit(state, jnp.int32(int(a)))
            next_obs_i.append(s_next.observation)
        all_obs.append(jnp.stack(next_obs_i))        # (K_i, 5, 5, 115)
        all_legal.append(legal_actions)
        all_chosen.append(int(record.action))

        if (i + 1) % 100 == 0:
            print(f'  oracle phase 1: {i + 1}/{len(records)} positions stepped')

    # --- Phase 2: one batched model call (chunked) ---
    obs_batch = jnp.concatenate(all_obs, axis=0)    # (N_total, 5, 5, 115)
    n_total = obs_batch.shape[0]
    print(f'  oracle phase 2: calling value head on {n_total} observations '
          f'(chunk_size={chunk_size})')

    value_chunks = []
    for start in range(0, n_total, chunk_size):
        _, v = baseline_model(obs_batch[start:start + chunk_size])
        value_chunks.append(np.array(v))

    values_np = -np.concatenate(value_chunks)        # negate → white's perspective

    # --- Phase 3: reconstruct oracle scores per position ---
    oracle_scores = []
    offset = 0
    for legal_actions, chosen in zip(all_legal, all_chosen):
        K = len(legal_actions)
        values = values_np[offset:offset + K]
        offset += K

        idx = np.where(legal_actions == chosen)[0]
        chosen_idx = int(idx[0]) if len(idx) > 0 else int(np.argmax(values))

        v_chosen = values[chosen_idx]
        alt_values = np.delete(values, chosen_idx)
        score = float(np.mean(np.abs(v_chosen - alt_values))) if len(alt_values) > 0 else 0.0
        oracle_scores.append(score)

    return oracle_scores
