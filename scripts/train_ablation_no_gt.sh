#!/bin/bash
# Ablation — self-evolve WITHOUT ground-truth answers (majority-voting GRPO / TTRL).
# The deterministic engine still VERIFIES question validity and RENDERS the question,
# but its ground-truth ANSWER is discarded for the Solver reward. Instead, within each
# GRPO group (one verified question, n=8 rollouts) a self-consistency pseudo-label is
# formed by majority voting (mode of MC letters / count ints, median of numerics) and
# each rollout is rewarded by agreement with it. Only the Round-2 reward changes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAJ="${HERE}/self_evolve/ablations/no_gt_majority_vote/reward_function/unified_reward_majority.py"
export EXP_NAME="${EXP_NAME:-verse_no_gt_majority}"
export MAX_STEPS="${MAX_STEPS:-100}"
# Ablate GT only; keep process reward off (matches the outcome-only base recipe).
export VERSE_VSI_GROUNDING_WEIGHT="${VERSE_VSI_GROUNDING_WEIGHT:-0.0}"
export VERSE_MAZE_FIDELITY_WEIGHT="${VERSE_MAZE_FIDELITY_WEIGHT:-0.0}"
echo "[ablation:no-GT] Round-2 reward: engine GT answer -> majority-vote self-consistency"
exec bash "${HERE}/scripts/train_self_evolve.sh" \
    "worker.reward.reward_function=${MAJ}:compute_score" "$@"
