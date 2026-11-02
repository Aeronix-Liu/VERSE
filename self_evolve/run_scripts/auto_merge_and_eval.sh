#!/bin/bash
# Auto-merge checkpoints and run 5-benchmark eval in reverse order (step100→25).
# Waits for training to finish first.
set -u

XINRAN="/data/pengtianshuo/self_evolve/Xinran"
REPO="${XINRAN}/ViLaSR"
EVOLVED="${XINRAN}/ViLaSR-self-evolve"
CKPT_ROOT="${XINRAN}/checkpoints/vilasr_self_evolve_immitate_porposer"
MERGE_PY="${EVOLVED}/easy_r1/scripts/model_merger.py"
EVAL_SCRIPT="${REPO}/run_scripts/eval_fullbench.sh"
PY="/root/miniconda3/envs/xinran_verl_demo/bin/python"
LOGDIR="${XINRAN}/logs"
OUTROOT="${XINRAN}/benchmark_result/vilasr_self_evolve_immitate_porposer"
GPUS=(4 5 6 7); NSHARD=4
DATADIR="${REPO}/benchmark"

LOG="${LOGDIR}/auto_merge_and_eval.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

export HF_ENDPOINT=https://hf-mirror.com
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

mkdir -p "${LOGDIR}" "${OUTROOT}"

# ============================================================
# Phase 1: Wait for training to finish
# ============================================================
log "================ WAITING FOR TRAINING TO FINISH ================"
while true; do
    if ! pgrep -f "train_main.py.*config_phase2" >/dev/null 2>&1; then
        log "Training process not found. Checking if step-100 checkpoint exists..."
        if [ -d "${CKPT_ROOT}/global_step_100" ]; then
            log "Step-100 checkpoint exists. Training finished."
            break
        fi
        # Maybe training crashed; check if step-75 exists at least
        if [ -d "${CKPT_ROOT}/global_step_75" ]; then
            log "WARNING: No step-100 checkpoint, but step-75 exists. Training may have ended early."
            break
        fi
        log "No training process and no step-75+ checkpoint. Waiting 60s..."
    else
        STEP=$(grep "modality-aware.*step" "$(cat ${EVOLVED}/logs/latest_vilasr_self_evolve_immitate_porposer.logpath 2>/dev/null)" 2>/dev/null | tail -1 | grep -oP 'step \K[0-9]+' || echo "?")
        log "Training still running (step ${STEP}). Waiting 120s..."
    fi
    sleep 120
done

# Give a few seconds for final checkpoint save to complete
sleep 30

# ============================================================
# Phase 2: Merge all checkpoints
# ============================================================
log "================ MERGING CHECKPOINTS ================"
STEPS_TO_MERGE=()
for s in 100 75 50 25; do
    if [ -d "${CKPT_ROOT}/global_step_${s}/actor" ] && ls "${CKPT_ROOT}/global_step_${s}/actor/model_world_size_4_rank_0.pt" >/dev/null 2>&1; then
        STEPS_TO_MERGE+=($s)
    else
        log "SKIP step-${s}: no sharded checkpoint found"
    fi
done

for s in "${STEPS_TO_MERGE[@]}"; do
    SHARD_DIR="${CKPT_ROOT}/global_step_${s}/actor"
    MERGED_DIR="${CKPT_ROOT}/global_step_${s}_merged"
    log "Merging step-${s} -> ${MERGED_DIR}"
    if [ -f "${MERGED_DIR}/model.safetensors" ]; then
        log "  Already merged, skipping."
        continue
    fi
    mkdir -p "${MERGED_DIR}"
    # Copy huggingface config/tokenizer files
    cp -rf "${SHARD_DIR}/huggingface/"* "${MERGED_DIR}/" 2>/dev/null
    # Merge shards
    ${PY} "${MERGE_PY}" --local_dir "${SHARD_DIR}" 2>&1 | tee -a "${LOG}"
    # model_merger.py saves to huggingface/ subdir; move to merged dir
    if [ -d "${SHARD_DIR}/huggingface" ]; then
        cp -f "${SHARD_DIR}/huggingface/"*.safetensors "${MERGED_DIR}/" 2>/dev/null
        cp -f "${SHARD_DIR}/huggingface/"*.json "${MERGED_DIR}/" 2>/dev/null
    fi
    if ls "${MERGED_DIR}"/model-*.safetensors >/dev/null 2>&1 || [ -f "${MERGED_DIR}/model.safetensors" ]; then
        log "  Step-${s} merged OK: $(ls ${MERGED_DIR}/model-*.safetensors 2>/dev/null | wc -l) safetensors files"
    else
        log "  ERROR: Step-${s} merge failed (no safetensors found)"
    fi
done

# ============================================================
# Phase 3: Eval on 5 benchmarks in reverse order (step100 -> step25)
# ============================================================
log "================ STARTING 5-BENCHMARK EVAL ================"
BENCHES=(
    "vsi_bench|VSI-Bench|vsi_bench.json"
    "spar_bench|SPAR-Bench|spar_bench.json"
    "maze|MAZE|maze.json"
    "SpatialEval_spatialreal|SpatialEval-Real|SpatialEval_spatialreal.json"
    "mmsi_bench|MMSI-Bench|mmsi_bench.json"
)

for s in "${STEPS_TO_MERGE[@]}"; do
    CKPT="${CKPT_ROOT}/global_step_${s}_merged"
    if ! ls "${CKPT}"/model-*.safetensors >/dev/null 2>&1 && [ ! -f "${CKPT}/model.safetensors" ]; then
        log "SKIP eval step-${s}: merged checkpoint missing"
        continue
    fi
    LABEL="selfevolve_step${s}"
    log "======== EVAL ${LABEL} (ckpt: ${CKPT}) ========"

    for job in "${BENCHES[@]}"; do
        IFS='|' read -r bkey bfolder djson <<< "${job}"
        MDIR="${OUTROOT}/${LABEL}/${bfolder}"
        SUMM="${MDIR}/${LABEL}_summary.json"
        [ -f "${SUMM}" ] && { log "  SKIP ${LABEL}/${bfolder} (summary exists)"; continue; }

        log "  ----- ${LABEL}/${bfolder} -----"
        mkdir -p "${MDIR}"

        # Wait for GPUs to be free
        for v in $(seq 1 60); do
            busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                   | awk -F', *' '$1>=4 && $1<=7 && $2>12000 {c++} END{print c+0}')
            [ "${busy}" = "0" ] && break
            sleep 5
        done

        PIDS=()
        for i in "${!GPUS[@]}"; do
            G=${GPUS[$i]}
            SHLOG="${LOGDIR}/eval_${LABEL}_${bkey}_shard${i}.log"
            # Use agentic protocol (infer.py) for self-evolve models
            CUDA_VISIBLE_DEVICES=${G} ${PY} "${REPO}/eval/infer.py" \
                --model-path "${CKPT}" --model-name Qwen2.5_VL_7B --dataset "${bkey}" \
                --input-file "${DATADIR}/${djson}" --image-folder "${DATADIR}" \
                --output-dir "${MDIR}" --temperature 0.75 \
                --max-frames 16 --max-pixels 351232 \
                --split "${i}" --all "${NSHARD}" \
                > "${SHLOG}" 2>&1 &
            PIDS+=($!)
            sleep 5
        done
        log "    shards launched: ${PIDS[*]}"
        for p in "${PIDS[@]}"; do wait "$p"; done
        log "    ${LABEL}/${bfolder}: inference done."

        # Merge shards and score
        MERGED="${MDIR}/results.jsonl"
        : > "${MERGED}"
        for ((i=0; i<NSHARD; i++)); do
            SH="${MDIR}/${i}_${NSHARD}/results.jsonl"
            if [ -f "${SH}" ]; then cat "${SH}" >> "${MERGED}"; fi
        done
        PYTHONPATH="${REPO}" ${PY} "${REPO}/eval/evaluate.py" \
            --dataset "${bkey}" \
            --question-file "${MERGED}" \
            --result-file "${MERGED}" \
            --output-result "${MDIR}/scores.jsonl" >> "${LOG}" 2>&1 || true
        log "    ${LABEL}/${bfolder}: scored."
    done
done

log "================ ALL EVAL DONE ================"
log "Results in: ${OUTROOT}"
