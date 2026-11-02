# Ablations

Two reward ablations of the Stage-3 self-evolve loop. Both start from the RRS init
policy, use the identical recipe (schedule v3, `batch=16`, `rollout.n=8`, one task
type per step, KL-anchored), and change exactly **one** thing in the Round-2 reward.
No core trainer code is modified — the ablations are isolated under
`self_evolve/ablations/`.

Baseline Round-2 (Solver) reward:

```
r = 0.1 * format + (0.9 - w) * accuracy(answer, engine_GT) + w * Φ
```

where `Φ` is the engine-verified **process reward** (maze path-fidelity / VSI
grounding IoU) gated by `w = VERSE_VSI_GROUNDING_WEIGHT` (or the maze weight).

| Run | What changes | How to launch |
|---|---|---|
| **Main (process reward)** | `w = 0.2` | `MAX_STEPS=100 bash scripts/train_self_evolve.sh` |
| **Ablation 1 — no process reward** | `w = 0` → pure outcome | `bash scripts/train_ablation_no_process.sh` |
| **Ablation 2 — no GT (majority vote)** | GT answer → self-consistency pseudo-label | `bash scripts/train_ablation_no_gt.sh` |

## Ablation 1 — no process reward (outcome-only)
Sets both process-reward weights to 0, so `r = 0.1*format + 0.9*accuracy`. GT is
still used. This is just the base reward at `w=0`, so it needs no new code — the
launcher only forces the weights to 0. Use it as the `w=0` vs `w=0.2` contrast to
isolate the contribution of the engine-verified process reward.

## Ablation 2 — no GT (majority-voting / TTRL-style)
The engine still verifies validity and renders the question (so questions stay
well-formed and MC options are intact), but its ground-truth **answer** is dropped
from the Solver reward. Within each GRPO group (one `uid` = one verified question,
`n=8` rollouts) a pseudo-label is estimated by majority voting:

- multiple-choice / maze → mode of the option letters
- `object_counting` → mode of the integer answers
- other numeric tasks → median of parsed floats
- otherwise → mode of normalized answer strings

Each rollout is then scored (via the same scorer as production) against the group's
pseudo-label. Diagnostics logged (never in the reward): `majority_vote/consensus_ratio`
and `majority_vote/pseudo_vs_true_acc` (how well the vote matches the engine GT).

Code: `self_evolve/ablations/no_gt_majority_vote/reward_function/`
(`agentic_solver_majority_reward.py` reuses every helper of the production
`agentic_solver_reward.py`; `unified_reward_majority.py` routes Round-1 to the
unchanged Proposer reward and Round-2 to the majority reward). The launcher just
overrides `worker.reward.reward_function` to point at it.

CPU unit tests (no GPU/network):
```bash
conda run -n verse python self_evolve/ablations/no_gt_majority_vote/test_majority_reward.py
```
