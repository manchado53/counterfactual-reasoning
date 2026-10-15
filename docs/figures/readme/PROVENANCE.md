# README figure provenance

Figures copied from unmerged branches so the README can show them. Each file is byte-identical to the source path at the listed commit.

| File | Branch @ commit | Source path |
|---|---|---|
| `jaxnav_collapse_500k.png` | `research/jaxnav-greedy-coverage` @ `1596514` | `docs/figures/real/claim2/jaxnav/fig_jaxnav_collapse_500k.png` |
| `jaxnav_why_null.png` | `research/jaxnav-greedy-coverage` @ `1596514` | `docs/figures/real/claim2/jaxnav/fig_why_null.png` |
| `jaxnav_why_small_buffer_fails.png` | `research/jaxnav-greedy-coverage` @ `1596514` | `docs/figures/real/claim2/jaxnav/fig_why_small_buffer_fails.png` |
| `doorkey_iqm.png` | `experiment/cce-doorkey` @ `ee12113` | `docs/figures/real/claim2/doorkey/fig1_iqm_doorkey.png` |
| `cvrp_routing_summary.png` | `research/cvrp-claim1-and-c2-audit` @ `c2bdb00` | `docs/figures/real/claim2/fig_routing_summary.png` |

Equation images in `eq/` were rendered from `paper/paper.tex` (Eqs. for return, return distribution, priorities, mixing, IS weight, TD target, oracle) with matplotlib mathtext.

## Generated figures

Made by [`analysis/readme/make_readme_figs.py`](../../../analysis/readme/make_readme_figs.py):

| File | Source |
|---|---|
| `cce_core.png` | Fresh rollouts on FrozenLake 8x8 slippery: exact dynamics, greedy policy from value iteration, 1,000 rollouts per action; bars = share reaching the goal, score = TV over 50 return bins; seed 7 |
| `cce_priority.png` | Same setup, all 53 non-terminal states (300 rollouts each), ε = 0.01 |
| `jaxnav_curve.png` | Raw `eval.jsonl` from runs 277734–277736 (Uniform), 277751–277753 (PER), 281092, 281093, 280874 (CCE + ranked-TD decay). SHA-256 matches `docs/media/demos/jaxnav_learning_curve.json`. Recomputed window means differ from that file by ≤0.4 points (binning); the README table quotes the file. |

```bash
python analysis/readme/make_readme_figs.py <dir with <run-id>.jsonl eval logs> docs/figures/readme
```
