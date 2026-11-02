#!/bin/bash
# Merge + eval for vilasr-self-evolve-imitate-0811-no-spar (finished 2026-08-15,
# 200 steps, VSI+MAZE only training). Evaluates VSI-Bench / SPAR-Bench / MAZE on
# GPU 4-7 with the v3 shard-guard protocol. SPAR is still evaluated even though
# it was not trained — it tells us whether dropping SPAR training hurt SPAR.
#
# Usage:  bash run_eval_imitate0811_no_spar.sh          # step 200 only
#         STEPS="200 100" bash run_eval_imitate0811_no_spar.sh
set -u

XINRAN="/data/pengtianshuo/self_evolve/Xinran"
REPO="${XINRAN}/ViLaSR"
EVOLVED="${XINRAN}/ViLaSR-self-evolve"
CKPT_ROOT="${XINRAN}/checkpoints/vilasr-self-evolve-imitate-0811-no-spar"
MERGE_PY="${EVOLVED}/easy_r1/scripts/model_merger.py"
PY="/root/miniconda3/envs/xinran_verl_demo/bin/python"
LOGDIR="${XINRAN}/logs"
OUTROOT="${XINRAN}/benchmark_result/vilasr-self-evolve-imitate-0811-no-spar"
GPUS=(4 5 6 7); NSHARD=4
DATADIR="${REPO}/benchmark"
GUARDIAN="/data/pengtianshuo/self_evolve/dlc_shell"

LOG="${LOGDIR}/eval_imitate0811_no_spar.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

MAX_RETRY=${MAX_RETRY:-3}
STEPS_DEFAULT="200"
STEPS_STR="${STEPS:-${STEPS_DEFAULT}}"
read -r -a STEPS <<< "${STEPS_STR}"

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
    log "  WARNING: GPUs not clean after 60s, proceeding anyway"
    return 1
}

# ============================================================
# Phase 1: merge requested steps (idempotent)
# ============================================================
for s in "${STEPS[@]}"; do
    SHARD_DIR="${CKPT_ROOT}/global_step_${s}/actor"
    MERGED_DIR="${CKPT_ROOT}/global_step_${s}_merged"
    if ls "${MERGED_DIR}"/model*.safetensors >/dev/null 2>&1; then
        log "step-${s} already merged."
        continue
    fi
    if ! ls "${SHARD_DIR}/model_world_size_4_rank_0.pt" >/dev/null 2>&1; then
        log "SKIP step-${s}: no sharded checkpoint"
        continue
    fi
    log "Merging step-${s} -> ${MERGED_DIR}"
    mkdir -p "${MERGED_DIR}"
    cp -rf "${SHARD_DIR}/huggingface/"* "${MERGED_DIR}/" 2>/dev/null
    ${PY} "${MERGE_PY}" --local_dir "${SHARD_DIR}" 2>&1 | tee -a "${LOG}"
    if [ -d "${SHARD_DIR}/huggingface" ]; then
        cp -f "${SHARD_DIR}/huggingface/"*.safetensors "${MERGED_DIR}/" 2>/dev/null
        cp -f "${SHARD_DIR}/huggingface/"*.json "${MERGED_DIR}/" 2>/dev/null
    fi
    if ls "${MERGED_DIR}"/model*.safetensors >/dev/null 2>&1; then
        log "  step-${s} merged OK"
    else
        log "  ERROR: step-${s} merge failed"
    fi
done

# ============================================================
# Phase 2: eval (same shard-guard protocol as run_eval_imitate0808_step100.sh)
# ============================================================
compute_expected() {
    local djson="$1" split_idx="$2"
    local qfile="${DATADIR}/${djson}"
    local total
    if [ ! -f "${qfile}" ]; then echo 0; return; fi
    if [[ "${qfile}" == *.jsonl ]]; then
        total=$(wc -l < "${qfile}")
    else
        total=$(${PY} -c "import json; print(len(json.load(open('${qfile}'))))" 2>/dev/null || echo 0)
    fi
    local st=$(( (total * split_idx) / NSHARD ))
    local ed=$(( (total * (split_idx + 1)) / NSHARD ))
    echo $(( ed - st ))
}

run_shard_with_retry() {
    local i="$1" G="$2" bkey="$3" bfolder="$4" djson="$5" ckpt="$6" mdir="$7" expected="$8"
    local shdir="${mdir}/${i}_${NSHARD}"
    local shlog="${LOGDIR}/eval0811_${LABEL}_${bkey}_shard${i}.log"
    local attempt=0
    local rc=1
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
            log "      shard${i} OK (rc=0, lines=${nlines}, expected>=${expected})"
            return 0
        fi
        log "      shard${i} FAILED attempt ${attempt}: rc=${rc} lines=${nlines}/${expected}"
    done
    log "      shard${i} GIVING UP after ${MAX_RETRY} attempts (last rc=${rc})"
    return 1
}

BENCHES=(
    "vsi_bench|VSI-Bench|vsi_bench.json"
    "spar_bench|SPAR-Bench|spar_bench.json"
    "maze|MAZE|maze.json"
)

ABORT=0
for s in "${STEPS[@]}"; do
    CKPT="${CKPT_ROOT}/global_step_${s}_merged"
    if ! ls "${CKPT}"/model*.safetensors >/dev/null 2>&1; then
        log "SKIP step-${s}: no merged checkpoint"
        continue
    fi
    LABEL="selfevolve_step${s}"
    log "======== EVAL ${LABEL} ========"
    for job in "${BENCHES[@]}"; do
        IFS='|' read -r bkey bfolder djson <<< "${job}"
        MDIR="${OUTROOT}/${LABEL}/${bfolder}"
        log "  ----- ${LABEL}/${bfolder} -----"
        release_gpus
        wait_gpus_clean
        log "  GPUs ready"
        DJSON="${djson}"
        mkdir -p "${MDIR}"
        PIDS=()
        for i in "${!GPUS[@]}"; do
            G=${GPUS[$i]}
            SHLOG="${LOGDIR}/eval0811_${LABEL}_${bkey}_shard${i}.log"
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
        log "    inference done. rc=[${SHARD_RC[*]}]"
        for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
        sleep 5
        for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done
        ALL_OK=1
        for i in "${!GPUS[@]}"; do
            G=${GPUS[$i]}
            shdir="${MDIR}/${i}_${NSHARD}"
            expected=$(compute_expected "${DJSON}" "${i}")
            nlines=0; [ -f "${shdir}/results.jsonl" ] && nlines=$(wc -l < "${shdir}/results.jsonl")
            if [ "${SHARD_RC[$i]}" -eq 0 ] && [ "${nlines}" -ge "${expected}" ]; then
                log "    shard${i} OK (lines=${nlines}/${expected})"
                continue
            fi
            log "    shard${i} incomplete (rc=${SHARD_RC[$i]}, lines=${nlines}/${expected}) -> retrying"
            if ! run_shard_with_retry "$i" "$G" "$bkey" "$bfolder" "${DJSON}" "$CKPT" "$MDIR" "${expected}"; then
                ALL_OK=0
            fi
        done
        wait_gpus_clean
        if [ "${ALL_OK}" -ne 1 ]; then
            log "    !! ${LABEL}/${bfolder} INCOMPLETE — aborting."
            ABORT=1
            break
        fi
        MERGED="${MDIR}/results.jsonl"
        : > "${MERGED}"
        for ((i=0; i<NSHARD; i++)); do
            SH="${MDIR}/${i}_${NSHARD}/results.jsonl"
            [ -f "${SH}" ] && cat "${SH}" >> "${MERGED}"
        done
        total=$(wc -l < "${MERGED}")
        log "    ${LABEL}/${bfolder}: ${total} results merged."
        if [ "${total}" -gt 0 ]; then
            PYTHONPATH="${REPO}" ${PY} "${REPO}/eval/evaluate.py" \
                --dataset "${bkey}" \
                --question-file "${MERGED}" \
                --result-file "${MERGED}" \
                --output-result "${MDIR}/scores.jsonl" >> "${LOG}" 2>&1 || true
            log "    ${LABEL}/${bfolder}: scored."
        fi
    done
    [ "${ABORT}" -ne 0 ] && break
done

if [ "${ABORT}" -ne 0 ]; then
    log "================ EVAL ABORTED ================"
    exit 1
fi
log "================ ALL EVAL DONE ================"
