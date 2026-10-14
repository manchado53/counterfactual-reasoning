# DoorKey 12x12 VLM-RB paper-parity experiment

This is the corrected full-state DoorKey transfer. It is separate from the
`procedural-12x12` partial-observation stress test.

Matched from arXiv:2602.01915 Appendix B, E.2, and E.4:

- full `12 x 12 x 3` symbolic grid;
- five actions: left, right, forward, pickup, toggle;
- per-channel embeddings and a three-block residual dilated CNN;
- Double DQN, learning rate `4e-5`, gamma `0.95`, batch size `128`;
- learning starts at 500 steps, train every 4 steps, hard target copy every 1,000;
- epsilon `1.0 -> 0.05` over 500,000 of 1,000,000 steps;
- official sparse, time-scaled MiniGrid reward and 1,440-step cap.

The paper does not publish its replay capacity or exact embedding/CNN widths.
This recipe pins those unresolved choices to a 100,000-row buffer, embedding
widths `[16, 8, 8]`, 128 CNN channels, and a 128-unit DQN head.

Evaluation uses the same fixed bank of 1,000 maps for every arm and seed. This
is stricter than the paper's 32-episode evaluation and does not affect training.

Run `smoke.sbatch` first. Then run `gate.sbatch`, which trains only Uniform seed
0 for the full one-million-step budget. Submit `train.sbatch` only if the gate
shows real learning.
