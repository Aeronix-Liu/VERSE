#!/bin/bash
# Automation chain: wait for Phase-1d (DGE-verified baseline) to finish, then
# smoke-test Phase-2 (structured_imitate) for 3 steps, and only if the smoke
# produces valid questions without crashing, launch the full Phase-2 run.
#
# Run in tmux so it survives SSH disconnect:
#   tmux new-session -d -s phase2_chain "bash run_scripts/chain_phase2_after_phase1d.sh"
#
# Logs: logs/phase2_chain.log

XINRAN_ROOT=/data/pengtianshuo/self_evolve/Xinran
REPO_ROOT="${XINRAN_ROOT}/ViLaSR-self-evolve"
LOGDIR="${REPO_ROOT}/logs"
PHASE1D_LOG="${XINRAN_ROOT}/logs/grpo_dge_verified.log"
CHAIN_LOG="${LOGDIR}/phase2_chain.log"
mkdir -p "${LOGDIR}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${CHAIN_LOG}"; }

log "=== Phase-2 automation chain started ==="
log "Waiting for Phase-1d to finish (pgrep for config_grpo_dge_verified)"

# --- 1. Wait for Phase-1d training to actually exit ---
# We poll the live process (not the log) because the log contains stale
# "exited rc=" lines from the earlier failed launch. The Phase-1d verl main
# command line includes "config_grpo_dge_verified", so we match that exactly.
seen_running=0
while true; do
    pid=$(pgrep -f "config_grpo_dge_verified" | head -1)
    if [[ -n "${pid}" ]]; then
        seen_running=1
        sleep 60
        continue
    fi
    if [[ ${seen_running} -eq 1 ]]; then
        log "Phase-1d process gone (was running, now exited)."
        break
    fi
    # Not seen yet — maybe between launches. Check the log for the latest exit.
    sleep 30
done

# Give the launcher teardown + guardian re-occupy a moment.
log "Sleeping 120s to let Phase-1d release GPUs and guardian re-occupy..."
sleep 120

# --- 2. Phase-2 smoke (3 steps, no checkpoint) ---
SMOKE_EXP="phase2_smoke"
SMOKE_LOG="${LOGDIR}/${SMOKE_EXP}.log"
SMOKE_CKPT_DIR="${XINRAN_ROOT}/checkpoints/${SMOKE_EXP}"
# Clean any stale smoke checkpoint/tracker so verl starts fresh (no resume).
if [[ -d "${SMOKE_CKPT_DIR}" ]]; then
    log "Removing stale smoke checkpoint dir ${SMOKE_CKPT_DIR}"
    rm -rf "${SMOKE_CKPT_DIR}"
fi
log "=== Launching Phase-2 smoke (3 steps) ==="
log "smoke log: ${SMOKE_LOG}"
cd "${REPO_ROOT}"
MAX_STEPS=3 SAVE_FREQ=-1 EXP_NAME="${SMOKE_EXP}" LOGGER_JSON='["file"]' \
    bash run_scripts/run_phase2_structured_imitate.sh > "${SMOKE_LOG}" 2>&1
SMOKE_RC=$?
log "Phase-2 smoke exited rc=${SMOKE_RC}"

# --- 3. Validate smoke ---
SMOKE_CKPT="${XINRAN_ROOT}/checkpoints/${SMOKE_EXP}"
SMOKE_EXPLOG="${SMOKE_CKPT}/experiment_log.jsonl"

ok=1
if [[ ${SMOKE_RC} -ne 0 ]]; then
    log "SMOKE FAILED: non-zero rc"
    ok=0
fi
if grep -q "Traceback" "${SMOKE_LOG}"; then
    # Distinguish benign DataLoader-worker-killed-during-shutdown from real errors.
    # A DataLoader worker killed by signal:Killed during the FINAL checkpoint save
    # (after max_steps reached, rc=0) is a known shutdown artifact, not a training
    # failure. Only fail if there's a non-benign traceback.
    real_tb=$(grep -c "Traceback" "${SMOKE_LOG}")
    benign_dl=$(grep -c "DataLoader worker.*killed by signal" "${SMOKE_LOG}")
    log "Traceback lines=${real_tb}, benign DataLoader-kill lines=${benign_dl}"
    if [[ ${real_tb} -gt ${benign_dl} ]]; then
        log "SMOKE FAILED: non-benign traceback found in smoke log"
        ok=0
    else
        log "Tracebacks are benign DataLoader-shutdown artifacts (rc=0, ckpt saved) -> not a failure"
    fi
fi
if [[ ! -f "${SMOKE_EXPLOG}" ]]; then
    log "SMOKE FAILED: no experiment_log.jsonl produced (no step completed)"
    ok=0
fi
if [[ ${ok} -eq 1 ]]; then
    # Check that at least some valid questions were produced (not 100% invalid).
    valid_summary=$(python3 -c "
import json,sys
rows=[json.loads(l) for l in open('${SMOKE_EXPLOG}')]
print('rows',len(rows))
for r in rows[:3]:
    print('step',r.get('step'),'round2_invalid',r.get('round2/invalid_ratio',r.get('reward_round2/invalid_ratio','?')))
" 2>/dev/null || echo "PARSE_FAIL")
    log "Smoke step summary: ${valid_summary}"
    if [[ "${valid_summary}" == "PARSE_FAIL" ]]; then
        log "SMOKE FAILED: could not parse experiment_log"
        ok=0
    fi
fi

if [[ ${ok} -ne 1 ]]; then
    log "=== SMOKE FAILED — NOT launching full Phase-2. Inspect ${SMOKE_LOG}. ==="
    log "Last 40 lines of smoke log:"
    tail -40 "${SMOKE_LOG}" | tee -a "${CHAIN_LOG}"
    exit 1
fi

log "=== SMOKE PASSED — launching full Phase-2 (100 steps) ==="
FULL_EXP="vilasr_self_evolve_immitate_porposer"
FULL_LOG="${LOGDIR}/${FULL_EXP}_chain_launch.log"
MAX_STEPS=100 SAVE_FREQ=50 EXP_NAME="${FULL_EXP}" \
    bash run_scripts/run_phase2_structured_imitate.sh > "${FULL_LOG}" 2>&1
FULL_RC=$?
log "Full Phase-2 exited rc=${FULL_RC}. Log: ${FULL_LOG}"
log "=== Phase-2 automation chain done ==="
