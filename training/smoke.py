"""Explicit small test profile. Never used implicitly for research runs."""
JAXNAV_OVERRIDES = dict(n_envs=2, collect_steps=4, batch_size=4, buffer_capacity=17,
    max_steps=8, n_episodes=6, eval_episodes=2, eval_interval=2,
    n_steps_per_update=4, target_update_freq=8, n_checkpoints=0, n_buffer_snapshots=0,
    decay_start=2, decay_end=4, qdiag_every=1, cf_score_chunk=8)
