# Sparse Four Rooms, twice the room width

Stop only the original Four Rooms array and its monitor, retaining all raw data. Reuse the reviewed Four Rooms code on a fresh branch from origin/master. Double each room's two interior dimensions while keeping walls and doors one cell wide, the same room connections, and corresponding fixed start/goal locations. This is a scaled Sparse Four Rooms variant, not the original benchmark.

Keep the three replay arms, five matched seeds, training budget, noise, sparse reward and 200-step cap. Verify the cap with a known-map shortest-path policy before training. Reuse the same learner and CCE settings; only observation size grows with the map. Add scale/map/parity tests, rerun CPU and T4 smoke checks, then dry-run and submit one capped array. Keep the old study intact, generate a new map and separate reports, start a quiet monitor, push and open an unmerged PR.
