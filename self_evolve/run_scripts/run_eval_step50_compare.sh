#!/bin/bash
# Matched-budget comparison eval (2026-08-18): step-50 checkpoints of
#   A) vilasr-se-selection-50step-0816  (resample + skip-R1-update + maze
#      fidelity w=0.2 + traj bank)
#   B) vilasr-self-evolve-imitate-0811-no-spar (baseline, same data v4 +
#      schedule v3 + base rrs-744)
# on VSI-Bench + MAZE (the benches covered by this training mix), GPU 4-7,
# v3 shard-guard protocol. ~6.3h per checkpoint.
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

LOG="${LOGDIR}/eval_step50_compare.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

MAX_RETRY=${MAX_RETRY:-3}

export HF_ENDPOINT=https://hf-mirror.com
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

mkdir -p "${LOGDIR}"

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
    log "  WARNING: GPUs not clean after 60s, proceeding anyway"
    return 1
}

merge_step() {
    local ckpt_root="$1" s="$2"
    local SHARD_DIR="${ckpt_root}/global_step_${s}/actor"
    local MERGED_DIR="${ckpt_root}/global_step_${s}_merged"
    if ls "${MERGED_DIR}"/model*.safetensors >/dev/null 2>&1; then
        log "already merged: ${MERGED_DIR}"
        return 0
    fi
    if ! ls "${SHARD_DIR}/model_world_size_4_rank_0.pt" >/dev/null 2>&1; then
        log "ERROR: no shards at ${SHARD_DIR}"
        return 1
    fi
    log "Merging ${SHARD_DIR} -> ${MERGED_DIR}"
    mkdir -p "${MERGED_DIR}"
    cp -rf "${SHARD_DIR}/huggingface/"* "${MERGED_DIR}/" 2>/dev/null
    ${PY} "${MERGE_PY}" --local_dir "${SHARD_DIR}" 2>&1 | tee -a "${LOG}"
    cp -f "${SHARD_DIR}/huggingface/"*.safetensors "${MERGED_DIR}/" 2>/dev/null
    cp -f "${SHARD_DIR}/huggingface/"*.json "${MERGED_DIR}/" 2>/dev/null
    ls "${MERGED_DIR}"/model*.safetensors >/dev/null 2>&1
}

compute_expected() {
    local djson="$1" split_idx="$2"
    local qfile="${DATADIR}/${djson}"
    local total
    if [ ! -f "${qfile}" ]; then echo 0; return; fi
    total=$(${PY} -c "import json; print(len(json.load(open('${qfile}'))))" 2>/dev/null || echo 0)
    local st=$(( (total * split_idx) / NSHARD ))
    local ed=$(( (total * (split_idx + 1)) / NSHARD ))
    echo $(( ed - st ))
}

run_shard_with_retry() {
    local i="$1" G="$2" bkey="$3" djson="$4" ckpt="$5" mdir="$6" expected="$7"
    local shdir="${mdir}/${i}_${NSHARD}"
    local shlog="${LOGDIR}/eval50cmp_${LABEL}_${bkey}_shard${i}.log"
    local attempt=0 rc=1
    while [ "${attempt}" -lt "${MAX_RETRY}" ]; do
        attempt=$((attempt + 1))
        log "      shard${i} (GPU${G}) attempt ${attempt}/${MAX_RETRY}"
        release_gpus >/dev/null 2>&1
        wait_gpus_clean >/dev/null 2>&1
        : > "${shlog}"
        CUDA_VISIBLE_DEVICES=${G} ${PY} "${REPO}/eval/infer.py" \
            --model-path "${ckpt}" --model-name Qwen2.5_VL_7B --dataset "${bkey}" \
            --input-file "${DATADIR}/${djson}" --image-folder "${DATADIR}" \
            --output-dir "${mdir}" --temperature 0.75 \
            --max-frames 16 --max-pixels 351232 \
            --split "${i}" --all "${NSHARD}" \
            > "${shlog}" 2>&1 &
        local pid=$!
        wait "${pid}" 2>/dev/null
        rc=$?
        kill -TERM "${pid}" 2>/dev/null; sleep 3
        kill -9 "${pid}" 2>/dev/null
        local nlines=0
        [ -f "${shdir}/results.jsonl" ] && nlines=$(wc -l < "${shdir}/results.jsonl")
        if [ "${rc}" -eq 0 ] && [ "${nlines}" -ge "${expected}" ]; then
            log "      shard${i} OK"
            return 0
        fi
        log "      shard${i} FAILED attempt ${attempt}: rc=${rc} lines=${nlines}/${expected}"
    done
    return 1
}

BENCHES=(
    "vsi_bench|VSI-Bench|vsi_bench.json"
    "maze|MAZE|maze.json"
)

# label|ckpt_root|outroot
MODELS=(
    "selection50|${XINRAN}/checkpoints/vilasr-se-selection-50step-0816|${XINRAN}/benchmark_result/vilasr-se-selection-50step-0816"
    "baseline0811s50|${XINRAN}/checkpoints/vilasr-self-evolve-imitate-0811-no-spar|${XINRAN}/benchmark_result/vilasr-self-evolve-imitate-0811-no-spar"
)

ABORT=0
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name ckpt_root outroot <<< "${entry}"
    if ! merge_step "${ckpt_root}" 50; then
        log "SKIP ${name}: merge failed"
        continue
    fi
    CKPT="${ckpt_root}/global_step_50_merged"
    LABEL="selfevolve_step50"
    log "======== EVAL ${name} (${CKPT}) ========"
    for job in "${BENCHES[@]}"; do
        IFS='|' read -r bkey bfolder djson <<< "${job}"
        MDIR="${outroot}/${LABEL}/${bfolder}"
        log "  ----- ${name}/${bfolder} -----"
        release_gpus
        wait_gpus_clean
        mkdir -p "${MDIR}"
        PIDS=()
        for i in "${!GPUS[@]}"; do
            G=${GPUS[$i]}
            SHLOG="${LOGDIR}/eval50cmp_${name}_${bkey}_shard${i}.log"
            : > "${SHLOG}"
            CUDA_VISIBLE_DEVICES=${G} ${PY} "${REPO}/eval/infer.py" \
                --model-path "${CKPT}" --model-name Qwen2.5_VL_7B --dataset "${bkey}" \
                --input-file "${DATADIR}/${djson}" --image-folder "${DATADIR}" \
                --output-dir "${MDIR}" --temperature 0.75 \
                --max-frames 16 --max-pixels 351232 \
                --split "${i}" --all "${NSHARD}" \
                > "${SHLOG}" 2>&1 &
            PIDS+=($!)
            sleep 2
        done
        log "    shards launched: ${PIDS[*]}"
        SHARD_RC=()
        for p in "${PIDS[@]}"; do wait "$p"; SHARD_RC+=($?); done
        for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
        sleep 5
        for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done
        ALL_OK=1
        for i in "${!GPUS[@]}"; do
            G=${GPUS[$i]}
            shdir="${MDIR}/${i}_${NSHARD}"
            expected=$(compute_expected "${djson}" "${i}")
            nlines=0; [ -f "${shdir}/results.jsonl" ] && nlines=$(wc -l < "${shdir}/results.jsonl")
            if [ "${SHARD_RC[$i]}" -eq 0 ] && [ "${nlines}" -ge "${expected}" ]; then
                continue
            fi
            log "    shard${i} incomplete -> retrying"
            if ! run_shard_with_retry "$i" "$G" "$bkey" "${djson}" "$CKPT" "$MDIR" "${expected}"; then
                ALL_OK=0
            fi
        done
        wait_gpus_clean
        if [ "${ALL_OK}" -ne 1 ]; then
            log "    !! ${name}/${bfolder} INCOMPLETE — aborting."
            ABORT=1
            break
        fi
        MERGED="${MDIR}/results.jsonl"
        : > "${MERGED}"
        for ((i=0; i<NSHARD; i++)); do
            SH="${MDIR}/${i}_${NSHARD}/results.jsonl"
            [ -f "${SH}" ] && cat "${SH}" >> "${MERGED}"
        done
        log "    ${name}/${bfolder}: $(wc -l < ${MERGED}) results merged."
        PYTHONPATH="${REPO}" ${PY} "${REPO}/eval/evaluate.py" \
            --dataset "${bkey}" \
            --question-file "${MERGED}" \
            --result-file "${MERGED}" \
            --output-result "${MDIR}/scores.jsonl" >> "${LOG}" 2>&1 || true
        log "    ${name}/${bfolder}: scored."
    done
    [ "${ABORT}" -ne 0 ] && break
done

[ "${ABORT}" -ne 0 ] && { log "======== COMPARE EVAL ABORTED ========"; exit 1; }
log "======== COMPARE EVAL DONE ========"
