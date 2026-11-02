#!/bin/bash
# Ablation — NO process reward (outcome-only). Same recipe as the main self-evolve
# run, but the engine-verified process (fidelity/grounding) reward weight = 0, so
# the Round-2 reward is pure outcome:  r = 0.1*format + 0.9*accuracy(answer, GT).
# This is the direct w=0 counterpart of the main run (w=0.2). GT is still used.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EXP_NAME="${EXP_NAME:-verse_no_process}"
export MAX_STEPS="${MAX_STEPS:-100}"
export VERSE_VSI_GROUNDING_WEIGHT=0.0
export VERSE_MAZE_FIDELITY_WEIGHT=0.0
echo "[ablation:no-process] outcome-only reward (process/fidelity weight = 0)"
exec bash "${HERE}/scripts/train_self_evolve.sh" "$@"
