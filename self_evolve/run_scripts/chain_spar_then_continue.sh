#!/bin/bash
# 2026-08-19 supervisor: (1) wait for the running MAZE step-50 eval to finish,
# (2) stop the old tts_eval chain so it does NOT start step-25 eval,
# (3) eval SPAR-Bench on TTS step-50, (4) continue training +25 steps (50->75)
# by resuming the same experiment WITH optimizer state saved at step-75
# (save_model_only=false), (5) merge step-75 + eval VSI-Bench on step-75.
# Survives SSH disconnect (run in tmux tts_next).
set -u

XINRAN="/data/pengtianshuo/self_evolve/Xinran"
REPO="${XINRAN}/ViLaSR"
EVOLVED="${XINRAN}/ViLaSR-self-evolve"
SCRIPTS_DIR="${EVOLVED}/run_scripts"
MERGE_PY="${EVOLVED}/easy_r1/scripts/model_merger.py"
PY="/root/miniconda3/envs/xinran_verl_demo/bin/python"
LOGDIR="${XINRAN}/logs"
GPUS=(4 5 6 7); NSHARD=4
DATADIR="${REPO}/benchmark"
GUARDIAN="/data/pengtianshuo/self_evolve/dlc_shell"

CKPT_ROOT="${XINRAN}/checkpoints/vilasr-se-tts-search-50step"
OUTROOT="${XINRAN}/benchmark_result/vilasr-se-tts-search-50step"
MAZE_SCORES="${OUTROOT}/selfevolve_step50/MAZE/scores.jsonl"

LOG="${LOGDIR}/chain_spar_then_continue.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
MAX_RETRY=${MAX_RETRY:-3}
export HF_ENDPOINT=https://hf-mirror.com
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
mkdir -p "${LOGDIR}" "${OUTROOT}"

release_gpus() { python3 -c "
import sys; sys.path.insert(0, '${GUARDIAN}')
import gpu_guardian as g; g.stop_occupy_on_gpus([4,5,6,7])" 2>&1 | tail -1; sleep 3; }
wait_gpus_clean() {
    for v in $(seq 1 24); do
        busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
               | awk -F', *' '$1>=4 && $1<=7 && $2>35000 {c++} END{print c+0}')
        [ "${busy}" = "0" ] && return 0; sleep 5
    done; log "  WARNING: GPUs not clean, proceeding"; return 1
}
compute_expected() {
    local djson="$1" i="$2"; local q="${DATADIR}/${djson}"; local t
    [ ! -f "${q}" ] && { echo 0; return; }
    t=$(${PY} -c "import json;print(len(json.load(open('${q}'))))" 2>/dev/null || echo 0)
    echo $(( (t*(i+1))/NSHARD - (t*i)/NSHARD ))
}
run_shard_retry() {
    local i="$1" G="$2" bkey="$3" djson="$4" ckpt="$5" mdir="$6" exp="$7"
    local shdir="${mdir}/${i}_${NSHARD}"; local shlog="${LOGDIR}/sparcont_${bkey}_shard${i}.log"; local a=0 rc=1
    while [ "${a}" -lt "${MAX_RETRY}" ]; do
        a=$((a+1)); log "      shard${i}(GPU${G}) try ${a}/${MAX_RETRY}"; release_gpus>/dev/null 2>&1; wait_gpus_clean>/dev/null 2>&1; : > "${shlog}"
        CUDA_VISIBLE_DEVICES=${G} ${PY} "${REPO}/eval/infer.py" \
            --model-path "${ckpt}" --model-name Qwen2.5_VL_7B --dataset "${bkey}" \
            --input-file "${DATADIR}/${djson}" --image-folder "${DATADIR}" \
            --output-dir "${mdir}" --temperature 0.75 --max-frames 16 --max-pixels 351232 \
            --split "${i}" --all "${NSHARD}" > "${shlog}" 2>&1 &
        local pid=$!; wait "${pid}" 2>/dev/null; rc=$?; kill -9 "${pid}" 2>/dev/null
        local n=0; [ -f "${shdir}/results.jsonl" ] && n=$(wc -l < "${shdir}/results.jsonl")
        [ "${rc}" -eq 0 ] && [ "${n}" -ge "${exp}" ] && { log "      shard${i} OK (${n})"; return 0; }
        log "      shard${i} FAIL try ${a}: rc=${rc} lines=${n}/${exp}"
    done; return 1
}
merge_step() {
    local s="$1"
    local SHARD_DIR="${CKPT_ROOT}/global_step_${s}/actor"
    local MERGED_DIR="${CKPT_ROOT}/global_step_${s}_merged"
    if ls "${MERGED_DIR}"/model*.safetensors >/dev/null 2>&1; then log "step-${s} already merged"; return 0; fi
    if ! ls "${SHARD_DIR}/model_world_size_${NSHARD}_rank_0.pt" >/dev/null 2>&1; then log "ERROR: no shards at ${SHARD_DIR}"; return 1; fi
    log "Merging step-${s} -> ${MERGED_DIR}"; mkdir -p "${MERGED_DIR}"
    cp -rf "${SHARD_DIR}/huggingface/"* "${MERGED_DIR}/" 2>/dev/null
    ${PY} "${MERGE_PY}" --local_dir "${SHARD_DIR}" 2>&1 | tee -a "${LOG}"
    cp -f "${SHARD_DIR}/huggingface/"*.safetensors "${MERGED_DIR}/" 2>/dev/null
    cp -f "${SHARD_DIR}/huggingface/"*.json "${MERGED_DIR}/" 2>/dev/null
    ls "${MERGED_DIR}"/model*.safetensors >/dev/null 2>&1
}
eval_bench() {
    # eval_bench <ckpt> <bkey> <bfolder> <djson> <mdir>
    local ckpt="$1" bkey="$2" bfolder="$3" djson="$4" mdir="$5"
    mkdir -p "${mdir}"
    if [ -f "${mdir}/scores.jsonl" ]; then log "  ${bfolder}: scores exist, skip"; return 0; fi
    log "  ----- EVAL ${bfolder} (${ckpt}) -----"
    release_gpus; wait_gpus_clean
    local PIDS=()
    for i in "${!GPUS[@]}"; do
        local G=${GPUS[$i]}; local SHLOG="${LOGDIR}/sparcont_${bkey}_shard${i}.log"; : > "${SHLOG}"
        CUDA_VISIBLE_DEVICES=${G} ${PY} "${REPO}/eval/infer.py" \
            --model-path "${ckpt}" --model-name Qwen2.5_VL_7B --dataset "${bkey}" \
            --input-file "${DATADIR}/${djson}" --image-folder "${DATADIR}" \
            --output-dir "${mdir}" --temperature 0.75 --max-frames 16 --max-pixels 351232 \
            --split "${i}" --all "${NSHARD}" > "${SHLOG}" 2>&1 &
        PIDS+=($!); sleep 2
    done
    log "  ${bfolder} shards launched: ${PIDS[*]}"
    local RC=(); for p in "${PIDS[@]}"; do wait "$p"; RC+=($?); done
    for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done; sleep 5
    for i in "${!GPUS[@]}"; do
        local exp=$(compute_expected "${djson}" "${i}"); local shdir="${mdir}/${i}_${NSHARD}"
        local n=0; [ -f "${shdir}/results.jsonl" ] && n=$(wc -l < "${shdir}/results.jsonl")
        if [ "${RC[$i]}" -eq 0 ] && [ "${n}" -ge "${exp}" ]; then continue; fi
        log "  ${bfolder} shard${i} incomplete (${n}/${exp}) -> retry"
        run_shard_retry "$i" "${GPUS[$i]}" "$bkey" "${djson}" "$ckpt" "$mdir" "${exp}" || log "  ${bfolder} shard${i} still failing"
    done
    wait_gpus_clean
    local MERGED="${mdir}/results.jsonl"; : > "${MERGED}"
    for ((i=0;i<NSHARD;i++)); do local SH="${mdir}/${i}_${NSHARD}/results.jsonl"; [ -f "${SH}" ] && cat "${SH}" >> "${MERGED}"; done
    log "  ${bfolder}: $(wc -l < ${MERGED}) results merged."
    PYTHONPATH="${REPO}" ${PY} "${REPO}/eval/evaluate.py" --dataset "${bkey}" \
        --question-file "${MERGED}" --result-file "${MERGED}" --output-result "${mdir}/scores.jsonl" >> "${LOG}" 2>&1 || true
    log "  ${bfolder}: scored (official overall printed above in ${LOG})."
}

# ---------- Phase 1: wait for MAZE step-50 to be scored ----------
log "================ WAIT FOR MAZE step-50 scores ================"
while [ ! -f "${MAZE_SCORES}" ]; do sleep 60; done
log "MAZE step-50 scored."

# ---------- Phase 2: stop old chain + stray step-25 shards ----------
pkill -f "chain_tts_eval.sh" 2>/dev/null && log "killed old chain_tts_eval.sh" || log "no old chain running"
sleep 2
pkill -f "eval/infer.py" 2>/dev/null && log "killed stray infer.py (step-25 nascent)" || true
sleep 5
release_gpus; wait_gpus_clean

# ---------- Phase 3: SPAR-Bench eval on step-50 ----------
log "================ SPAR-Bench eval (step-50) ================"
eval_bench "${CKPT_ROOT}/global_step_50_merged" "spar_bench" "SPAR-Bench" "spar_bench.json" \
           "${OUTROOT}/selfevolve_step50/SPAR-Bench"

# ---------- Phase 4: continue training 50 -> 75 (save optimizer at 75) ----------
release_gpus; wait_gpus_clean
log "================ CONTINUE TRAINING 50 -> 75 (save_model_only=false) ================"
MAX_STEPS=75 SAVE_FREQ=25 SAVE_LIMIT=8 EXP_NAME=vilasr-se-tts-search-50step \
    EXTRA_OVERRIDES="trainer.load_checkpoint_path=${CKPT_ROOT}/global_step_50 trainer.save_model_only=false" \
    bash "${SCRIPTS_DIR}/run_phase2_imitate_tts.sh" 2>&1 | tee -a "${LOG}"
log "================ CONTINUE TRAINING DONE ================"
if [ -d "${CKPT_ROOT}/global_step_75/actor" ]; then
    log "step-75 saved. optimizer shards present: $(ls ${CKPT_ROOT}/global_step_75/actor/optim_world_size_${NSHARD}_rank_*.pt 2>/dev/null | wc -l)/4"
else
    log "ERROR: global_step_75 not found; aborting step-75 eval."; exit 1
fi

# ---------- Phase 5: merge step-75 + eval VSI-Bench ----------
release_gpus; wait_gpus_clean
log "================ MERGE + VSI-Bench eval (step-75) ================"
if merge_step 75; then
    eval_bench "${CKPT_ROOT}/global_step_75_merged" "vsi_bench" "VSI-Bench" "vsi_bench.json" \
               "${OUTROOT}/selfevolve_step75/VSI-Bench"
else
    log "ERROR: step-75 merge failed; skipping VSI eval."
fi
log "================ ALL DONE (SPAR-50 + train75 + VSI-75) ================"
