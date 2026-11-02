#!/bin/bash
# Chain (2026-08-16): wait for the imitate-0811-no-spar step-200 eval to finish,
# aggregate official-metric scores.json, then automatically launch a 50-step
# selection-scheme training run (resample + skip-R1-update + maze fidelity
# w=0.2 + trajectory bank) on GPU 4-7.
#
# Completion detection follows the 6.6 lesson: wait for the "ALL EVAL DONE" /
# "EVAL ABORTED" log markers, NOT pgrep gaps between benchmarks. A dead eval
# session without a marker is treated as crashed (logged, chain continues to
# training only after confirming no infer.py processes remain).
set -u

XINRAN="/data/pengtianshuo/self_evolve/Xinran"
EVOLVED="${XINRAN}/ViLaSR-self-evolve"
EVAL_LOG="${XINRAN}/logs/eval_imitate0811_no_spar.log"
CHAIN_LOG="${XINRAN}/logs/chain_eval0811_then_selection50.log"
PY="${XINRAN}/envs/Xinran_VERSE/bin/python"

EXP_NAME="${EXP_NAME:-vilasr-se-selection-50step-0816}"
MAX_STEPS="${MAX_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-25}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${CHAIN_LOG}"; }

log "======== chain started: waiting for eval_0811 to finish ========"
log "eval log: ${EVAL_LOG}"

EVAL_STATUS="unknown"
while true; do
    if grep -q "ALL EVAL DONE" "${EVAL_LOG}" 2>/dev/null; then
        EVAL_STATUS="done"
        break
    fi
    if grep -q "EVAL ABORTED" "${EVAL_LOG}" 2>/dev/null; then
        EVAL_STATUS="aborted"
        break
    fi
    if ! tmux has-session -t eval_0811 2>/dev/null; then
        if ! pgrep -f "eval/infer.py" >/dev/null 2>&1; then
            EVAL_STATUS="crashed"
            break
        fi
    fi
    LAST=$(tail -1 "${EVAL_LOG}" 2>/dev/null | cut -c1-120)
    log "eval still running... last: ${LAST}"
    sleep 300
done
log "eval finished with status: ${EVAL_STATUS}"

# Belt-and-braces: make sure no eval inference processes remain, let VRAM drain.
for i in $(seq 1 60); do
    pgrep -f "eval/infer.py" >/dev/null 2>&1 || break
    log "waiting for lingering infer.py processes to exit..."
    sleep 30
done
sleep 60

# ---- Official-metric scores.json (CPU, harmless even after abort/crash) ----
log "generating official-metric scores.json..."
${PY} "${EVOLVED}/run_scripts/gen_imitate0811_scores.py" 2>&1 | tee -a "${CHAIN_LOG}" || \
    log "WARNING: scores aggregation failed (eval may be incomplete); continuing"

# ---- Launch the 50-step selection training ----
log "launching ${EXP_NAME} (MAX_STEPS=${MAX_STEPS}, SAVE_FREQ=${SAVE_FREQ}) on GPU 4-7"
cd "${EVOLVED}"
MAX_STEPS="${MAX_STEPS}" SAVE_FREQ="${SAVE_FREQ}" SAVE_LIMIT=3 \
    EXP_NAME="${EXP_NAME}" \
    bash run_scripts/run_phase2_imitate_selection.sh 2>&1 | tee -a "${CHAIN_LOG}"
RC=$?
log "training launcher exited rc=${RC}"
log "======== chain finished ========"
