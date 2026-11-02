#!/bin/bash
set -u
XINRAN="/data/pengtianshuo/self_evolve/Xinran"
REPO_VI="${XINRAN}/ViLaSR"
REPO_SE="${XINRAN}/ViLaSR-self-evolve"
LOGDIR="${XINRAN}/logs"
LOG="${LOGDIR}/eval_then_smoke_chain.log"
mkdir -p "${LOGDIR}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

log "================ EVAL -> SMOKE CHAIN ARMED ================"

# 1. Wait for selfevolve200 VSI-Bench eval to finish
log "waiting for selfevolve200 VSI-Bench eval to finish..."
while pgrep -f "eval/infer.py.*selfevolve200.*vsi_bench" >/dev/null 2>&1; do
    sleep 60
done
log "VSI-Bench eval finished. waiting 120s for GPU release..."
sleep 120

# 2. Run rrs-744 maze eval
log "launching maze eval on rrs-744..."
bash "${REPO_VI}/run_scripts/eval_maze.sh" rrs744 "${XINRAN}/checkpoints/vilasr_reflective_4gpu" 2>&1 | tee -a "${LOG}"
log "maze eval finished."

# 3. Wait for maze eval shards to fully exit + GPU release
sleep 120

# 4. Launch maze-only smoke
log "launching maze-only smoke (10 steps, pick-letter mode)..."
cd "${REPO_SE}"
MAX_STEPS=10 SAVE_FREQ=-1 EXP_NAME=maze_pick_smoke \
    bash run_scripts/run_maze_only_pick.sh 2>&1 | tee -a "${LOG}"
log "================ EVAL -> SMOKE CHAIN DONE ================"
