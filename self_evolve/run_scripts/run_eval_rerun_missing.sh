#!/bin/bash
# Run 5-benchmark eval, releasing guardian before EACH benchmark to prevent re-occupation.
# Only re-runs missing shards. Uses a background CUDA loop to keep GPUs busy
# between shard launch and vLLM init (bridges the ~30s guardian check window).
set -u

XINRAN="/data/pengtianshuo/self_evolve/Xinran"
REPO="${XINRAN}/ViLaSR"
EVOLVED="${XINRAN}/ViLaSR-self-evolve"
CKPT_ROOT="${XINRAN}/checkpoints/vilasr_self_evolve_immitate_porposer"
PY="/root/miniconda3/envs/xinran_verl_demo/bin/python"
LOGDIR="${XINRAN}/logs"
OUTROOT="${XINRAN}/benchmark_result/vilasr_self_evolve_immitate_porposer"
GPUS=(4 5 6 7); NSHARD=4
DATADIR="${REPO}/benchmark"
GUARDIAN="/data/pengtianshuo/self_evolve/dlc_shell"

LOG="${LOGDIR}/eval_rerun_missing.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

export HF_ENDPOINT=https://hf-mirror.com
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

BENCHES=(
    "vsi_bench|VSI-Bench|vsi_bench.json"
    "spar_bench|SPAR-Bench|spar_bench.json"
    "maze|MAZE|maze.json"
    "SpatialEval_spatialreal|SpatialEval-Real|SpatialEval_spatialreal.json"
    "mmsi_bench|MMSI-Bench|mmsi_bench.json"
)

STEPS=(100 75 50 25)

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

        # Check which shards are missing
        MISSING_SHARDS=()
        for i in "${!GPUS[@]}"; do
            SH="${MDIR}/${i}_${NSHARD}/results.jsonl"
            if [ ! -f "$SH" ] || [ "$(wc -l < "$SH")" = "0" ]; then
                MISSING_SHARDS+=($i)
            fi
        done

        if [ ${#MISSING_SHARDS[@]} = 0 ]; then
            log "  SKIP ${LABEL}/${bfolder} (all 4 shards present)"
            continue
        fi

        log "  ${LABEL}/${bfolder}: re-running missing shards: ${MISSING_SHARDS[*]}"

        # KEY FIX: Release guardian RIGHT NOW, right before launching shards
        release_gpus
        log "  Guardian released for ${LABEL}/${bfolder}"

        mkdir -p "${MDIR}"
        PIDS=()
        for i in "${MISSING_SHARDS[@]}"; do
            G=${GPUS[$i]}
            SHLOG="${LOGDIR}/eval_${LABEL}_${bkey}_shard${i}_rerun.log"
            # Use lower GPU memory util if orphaned memory exists (~27GB free)
            GPU_MEM_UTIL=0.3 CUDA_VISIBLE_DEVICES=${G} ${PY} "${REPO}/eval/infer.py" \
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
        for p in "${PIDS[@]}"; do wait "$p"; done
        log "    ${LABEL}/${bfolder}: inference done."

        # Re-merge all shards
        MERGED="${MDIR}/results.jsonl"
        : > "${MERGED}"
        for ((i=0; i<NSHARD; i++)); do
            SH="${MDIR}/${i}_${NSHARD}/results.jsonl"
            if [ -f "${SH}" ]; then cat "${SH}" >> "${MERGED}"; fi
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
done

log "================ ALL MISSING SHARDS DONE ================"
