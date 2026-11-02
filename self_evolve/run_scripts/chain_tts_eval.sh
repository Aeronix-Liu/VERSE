#!/bin/bash
# 2026-08-18 auto-chain: wait for the VGS-GRPO training (tmux tts_run) to finish,
# then merge + eval the TTS checkpoint on VSI-Bench + MAZE, using the same
# shard-guard protocol as run_eval_step50_compare.sh. Designed to survive SSH
# disconnect (run inside tmux `tts_eval`). Non-destructive: only starts eval
# AFTER training has released the GPUs.
#
# Compare against (same budget, already evaluated):
#   benchmark_result/vilasr-se-selection-50step-0816/selfevolve_step50   (VSI 39.05 / MAZE 89.65)
#   benchmark_result/vilasr-self-evolve-imitate-0811-no-spar/... step50   (VSI 39.79 / MAZE 90.20)
set -u

XINRAN="/data/pengtianshuo/self_evolve/Xinran"
REPO="${XINRAN}/ViLaSR"
EVOLVED="${XINRAN}/ViLaSR-self-evolve"
MERGE_PY="${EVOLVED}/easy_r1/scripts/model_merger.py"
PY="/root/miniconda3/envs/xinran_verl_demo/bin/python"
LOGDIR="${XINRAN}/logs"
GPUS=(4 5 6 7); NSHARD=4
DATADIR="${REPO}/benchmark"
GUARDIAN="/data/pengtianshuo/self_evolve/dlc_shell"

CKPT_ROOT="${XINRAN}/checkpoints/vilasr-se-tts-search-50step"
OUTROOT="${XINRAN}/benchmark_result/vilasr-se-tts-search-50step"
TRAIN_LOG="$(cat ${EVOLVED}/logs/latest_vilasr-se-tts-search-50step.logpath 2>/dev/null)"
STEPS_TO_EVAL="${STEPS_TO_EVAL:-50 25}"   # reverse order: newest first

LOG="${LOGDIR}/chain_tts_eval.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

MAX_RETRY=${MAX_RETRY:-3}
export HF_ENDPOINT=https://hf-mirror.com
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
mkdir -p "${LOGDIR}" "${OUTROOT}"

release_gpus() {
    python3 -c "
import sys; sys.path.insert(0, '${GUARDIAN}')
import gpu_guardian as g
g.stop_occupy_on_gpus([4,5,6,7])
" 2>&1 | tail -2
    sleep 3
}
wait_gpus_clean() {
    for v in $(seq 1 12); do
        busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
               | awk -F', *' '$1>=4 && $1<=7 && $2>35000 {c++} END{print c+0}')
        [ "${busy}" = "0" ] && return 0
        sleep 5
    done
    log "  WARNING: GPUs not clean after 60s, proceeding anyway"; return 1
}
merge_step() {
    local ckpt_root="$1" s="$2"
    local SHARD_DIR="${ckpt_root}/global_step_${s}/actor"
    local MERGED_DIR="${ckpt_root}/global_step_${s}_merged"
    if ls "${MERGED_DIR}"/model*.safetensors >/dev/null 2>&1; then log "already merged: ${MERGED_DIR}"; return 0; fi
    if ! ls "${SHARD_DIR}/model_world_size_4_rank_0.pt" >/dev/null 2>&1; then log "ERROR: no shards at ${SHARD_DIR}"; return 1; fi
    log "Merging ${SHARD_DIR} -> ${MERGED_DIR}"
    mkdir -p "${MERGED_DIR}"
    cp -rf "${SHARD_DIR}/huggingface/"* "${MERGED_DIR}/" 2>/dev/null
    ${PY} "${MERGE_PY}" --local_dir "${SHARD_DIR}" 2>&1 | tee -a "${LOG}"
    cp -f "${SHARD_DIR}/huggingface/"*.safetensors "${MERGED_DIR}/" 2>/dev/null
    cp -f "${SHARD_DIR}/huggingface/"*.json "${MERGED_DIR}/" 2>/dev/null
    ls "${MERGED_DIR}"/model*.safetensors >/dev/null 2>&1
}
compute_expected() {
    local djson="$1" split_idx="$2"; local qfile="${DATADIR}/${djson}"; local total
    [ ! -f "${qfile}" ] && { echo 0; return; }
    total=$(${PY} -c "import json; print(len(json.load(open('${qfile}'))))" 2>/dev/null || echo 0)
    local st=$(( (total * split_idx) / NSHARD )); local ed=$(( (total * (split_idx + 1)) / NSHARD ))
    echo $(( ed - st ))
}
run_shard_with_retry() {
    local i="$1" G="$2" bkey="$3" djson="$4" ckpt="$5" mdir="$6" expected="$7" lbl="$8"
    local shdir="${mdir}/${i}_${NSHARD}"; local shlog="${LOGDIR}/ttseval_${lbl}_${bkey}_shard${i}.log"; local attempt=0 rc=1
    while [ "${attempt}" -lt "${MAX_RETRY}" ]; do
        attempt=$((attempt + 1)); log "      shard${i} (GPU${G}) attempt ${attempt}/${MAX_RETRY}"
        release_gpus >/dev/null 2>&1; wait_gpus_clean >/dev/null 2>&1; : > "${shlog}"
        CUDA_VISIBLE_DEVICES=${G} ${PY} "${REPO}/eval/infer.py" \
            --model-path "${ckpt}" --model-name Qwen2.5_VL_7B --dataset "${bkey}" \
            --input-file "${DATADIR}/${djson}" --image-folder "${DATADIR}" \
            --output-dir "${mdir}" --temperature 0.75 --max-frames 16 --max-pixels 351232 \
            --split "${i}" --all "${NSHARD}" > "${shlog}" 2>&1 &
        local pid=$!; wait "${pid}" 2>/dev/null; rc=$?
        kill -TERM "${pid}" 2>/dev/null; sleep 3; kill -9 "${pid}" 2>/dev/null
        local nlines=0; [ -f "${shdir}/results.jsonl" ] && nlines=$(wc -l < "${shdir}/results.jsonl")
        if [ "${rc}" -eq 0 ] && [ "${nlines}" -ge "${expected}" ]; then log "      shard${i} OK"; return 0; fi
        log "      shard${i} FAILED attempt ${attempt}: rc=${rc} lines=${nlines}/${expected}"
    done
    return 1
}

# ============================================================
# Phase 1: wait for training to finish (log marker primary; pgrep secondary)
# ============================================================
log "================ WAIT FOR TTS TRAINING (log=${TRAIN_LOG}) ================"
while true; do
    if [ -n "${TRAIN_LOG}" ] && grep -q "Training process exited rc=" "${TRAIN_LOG}" 2>/dev/null; then
        log "Detected 'Training process exited' marker. Proceeding."; break
    fi
    if ! pgrep -f "train_main.py.*vilasr-se-tts-search" >/dev/null 2>&1; then
        if [ -d "${CKPT_ROOT}/global_step_50" ] || [ -d "${CKPT_ROOT}/global_step_25" ]; then
            log "Training process gone and a checkpoint exists. Proceeding."; break
        fi
        log "Training process gone but no ckpt yet; waiting 120s (may be mid-bootstrap)..."
    else
        LASTSTEP=$(ls -d ${CKPT_ROOT}/global_step_* 2>/dev/null | grep -oE 'step_[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
        log "Training still running (latest ckpt step=${LASTSTEP:-none}). Waiting 120s..."
    fi
    sleep 120
done
sleep 60   # let the final checkpoint save flush

# ============================================================
# Phase 2+3: merge + eval each step on VSI + MAZE
# ============================================================
BENCHES=( "vsi_bench|VSI-Bench|vsi_bench.json" "maze|MAZE|maze.json" )
ABORT=0
for s in ${STEPS_TO_EVAL}; do
    if ! merge_step "${CKPT_ROOT}" "${s}"; then log "SKIP step-${s}: merge failed/absent"; continue; fi
    CKPT="${CKPT_ROOT}/global_step_${s}_merged"
    LABEL="selfevolve_step${s}"
    log "======== EVAL ${LABEL} (${CKPT}) ========"
    for job in "${BENCHES[@]}"; do
        IFS='|' read -r bkey bfolder djson <<< "${job}"
        MDIR="${OUTROOT}/${LABEL}/${bfolder}"
        SUMM="${MDIR}/scores.jsonl"
        [ -f "${SUMM}" ] && { log "  SKIP ${LABEL}/${bfolder} (scores exist)"; continue; }
        log "  ----- ${LABEL}/${bfolder} -----"; mkdir -p "${MDIR}"
        release_gpus; wait_gpus_clean
        PIDS=()
        for i in "${!GPUS[@]}"; do
            G=${GPUS[$i]}; SHLOG="${LOGDIR}/ttseval_${LABEL}_${bkey}_shard${i}.log"; : > "${SHLOG}"
            CUDA_VISIBLE_DEVICES=${G} ${PY} "${REPO}/eval/infer.py" \
                --model-path "${CKPT}" --model-name Qwen2.5_VL_7B --dataset "${bkey}" \
                --input-file "${DATADIR}/${djson}" --image-folder "${DATADIR}" \
                --output-dir "${MDIR}" --temperature 0.75 --max-frames 16 --max-pixels 351232 \
                --split "${i}" --all "${NSHARD}" > "${SHLOG}" 2>&1 &
            PIDS+=($!); sleep 2
        done
        log "    shards launched: ${PIDS[*]}"
        SHARD_RC=(); for p in "${PIDS[@]}"; do wait "$p"; SHARD_RC+=($?); done
        for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done; sleep 5
        for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done
        ALL_OK=1
        for i in "${!GPUS[@]}"; do
            G=${GPUS[$i]}; shdir="${MDIR}/${i}_${NSHARD}"; expected=$(compute_expected "${djson}" "${i}")
            nlines=0; [ -f "${shdir}/results.jsonl" ] && nlines=$(wc -l < "${shdir}/results.jsonl")
            if [ "${SHARD_RC[$i]}" -eq 0 ] && [ "${nlines}" -ge "${expected}" ]; then continue; fi
            log "    shard${i} incomplete -> retrying"
            run_shard_with_retry "$i" "$G" "$bkey" "${djson}" "$CKPT" "$MDIR" "${expected}" "${LABEL}" || ALL_OK=0
        done
        wait_gpus_clean
        if [ "${ALL_OK}" -ne 1 ]; then log "    !! ${LABEL}/${bfolder} INCOMPLETE — aborting this bench"; ABORT=1; continue; fi
        MERGED="${MDIR}/results.jsonl"; : > "${MERGED}"
        for ((i=0; i<NSHARD; i++)); do SH="${MDIR}/${i}_${NSHARD}/results.jsonl"; [ -f "${SH}" ] && cat "${SH}" >> "${MERGED}"; done
        log "    ${LABEL}/${bfolder}: $(wc -l < ${MERGED}) results merged."
        PYTHONPATH="${REPO}" ${PY} "${REPO}/eval/evaluate.py" \
            --dataset "${bkey}" --question-file "${MERGED}" --result-file "${MERGED}" \
            --output-result "${MDIR}/scores.jsonl" >> "${LOG}" 2>&1 || true
        log "    ${LABEL}/${bfolder}: scored."
    done
done

log "================ TTS EVAL DONE (abort_flag=${ABORT}) ================"
log "VSI official overall + MAZE avg are printed by evaluate.py above; grep '${LOG}' for 'Overall' and 'avg score'."
log "Raw results: ${OUTROOT}/selfevolve_step*/{VSI-Bench,MAZE}/"
