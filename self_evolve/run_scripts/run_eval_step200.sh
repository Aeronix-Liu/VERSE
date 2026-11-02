#!/bin/bash
# Full eval v3: 4 steps × 5 benchmarks × 4 shards, ALL re-run.
# Key fixes:
#   1. Release guardian before EACH benchmark
#   2. Use SIGTERM (not SIGKILL) for graceful vLLM shutdown
#   3. Wait for GPU memory to be fully released before next benchmark
#   4. Verify GPU memory < 5GB before launching shards
#   5. SHARD COMPLETENESS GUARD (added 2026-08-04):
#      - Each shard must exit 0 AND produce expected line count
#      - Failed/incomplete shards are auto-retried up to MAX_RETRY times
#      - If any shard still incomplete after retries, the whole benchmark
#        is marked FAILED (no merge, no score) and the run aborts.
#   6. STEPS override: set STEPS="100 75 50" to skip step25.
set -u

XINRAN="/data/pengtianshuo/self_evolve/Xinran"
REPO="${XINRAN}/ViLaSR"
CKPT_ROOT="${XINRAN}/checkpoints/vilasr_self_evolve_imitate_porposer_step200"
PY="/root/miniconda3/envs/xinran_verl_demo/bin/python"
LOGDIR="${XINRAN}/logs"
OUTROOT="${XINRAN}/benchmark_result/vilasr_self_evolve_imitate_porposer_step200"
GPUS=(4 5 6 7); NSHARD=4
DATADIR="${REPO}/benchmark"
GUARDIAN="/data/pengtianshuo/self_evolve/dlc_shell"

LOG="${LOGDIR}/eval_step200.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

# Max retries per shard on crash / incomplete output.
MAX_RETRY=${MAX_RETRY:-3}
# Which steps to eval. Default: skip step25 (user request 2026-08-04).
STEPS_DEFAULT="200 150"
STEPS_STR="${STEPS:-${STEPS_DEFAULT}}"
read -r -a STEPS <<< "${STEPS_STR}"

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

wait_gpus_clean() {
    # Wait for all GPU 4-7 to have < 5GB used
    for v in $(seq 1 120); do
        busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
               | awk -F', *' '$1>=4 && $1<=7 && $2>5000 {c++} END{print c+0}')
        [ "${busy}" = "0" ] && return 0
        sleep 5
    done
    log "  WARNING: GPUs not clean after 10min, proceeding anyway"
    return 1
}

# Expected number of questions for a specific shard.  infer.py splits data as
#   st, ed = len*split//all, len*(split+1)//all   -> count = ed - st
# So per-shard counts differ by at most 1; using a global ceil would falsely
# flag the smaller shards as incomplete.  We compute the exact per-shard count.
# Args: djson split_idx
compute_expected() {
    local djson="$1" split_idx="$2"
    local qfile="${DATADIR}/${djson}"
    local total
    if [ ! -f "${qfile}" ]; then
        echo 0; return
    fi
    if [[ "${qfile}" == *.jsonl ]]; then
        total=$(wc -l < "${qfile}")
    else
        total=$(${PY} -c "import json; print(len(json.load(open('${qfile}'))))" 2>/dev/null || echo 0)
    fi
    local st=$(( (total * split_idx) / NSHARD ))
    local ed=$(( (total * (split_idx + 1)) / NSHARD ))
    echo $(( ed - st ))
}

# Run ONE shard, retry on crash/incomplete. Returns 0 only if shard is complete.
# Args: shard_idx gpu_id bkey bfolder djson ckpt outdir expected
run_shard_with_retry() {
    local i="$1" G="$2" bkey="$3" bfolder="$4" djson="$5" ckpt="$6" mdir="$7" expected="$8"
    local shdir="${mdir}/${i}_${NSHARD}"
    local shlog="${LOGDIR}/eval_${LABEL}_${bkey}_shard${i}.log"
    local attempt=0
    local rc=1

    while [ "${attempt}" -lt "${MAX_RETRY}" ]; do
        attempt=$((attempt + 1))
        log "      shard${i} (GPU${G}) attempt ${attempt}/${MAX_RETRY}"

        # Make sure GPUs are clean before (re)launching this shard.
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

        # Graceful cleanup of any lingering vLLM procs from this shard.
        kill -TERM "${pid}" 2>/dev/null; sleep 3
        kill -9 "${pid}" 2>/dev/null

        local nlines=0
        [ -f "${shdir}/results.jsonl" ] && nlines=$(wc -l < "${shdir}/results.jsonl")

        if [ "${rc}" -eq 0 ] && [ "${nlines}" -ge "${expected}" ]; then
            log "      shard${i} OK (rc=0, lines=${nlines}, expected>=${expected})"
            return 0
        fi
        log "      shard${i} FAILED attempt ${attempt}: rc=${rc} lines=${nlines}/${expected}"
        # If the shard file exists but is incomplete, infer.py will resume from
        # its existing line count (append mode) on the next attempt — good.
        # If it crashed at startup (e.g. OOM), the partial file (if any) stays
        # and resume continues.
    done
    log "      shard${i} GIVING UP after ${MAX_RETRY} attempts (last rc=${rc})"
    return 1
}

BENCHES=(
    "vsi_bench|VSI-Bench|vsi_bench.json"
    "spar_bench|SPAR-Bench|spar_bench.json"
    "maze|MAZE|maze.json"
    "SpatialEval_spatialreal|SpatialEval-Real|SpatialEval_spatialreal.json"
    "mmsi_bench|MMSI-Bench|mmsi_bench.json"
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

        # Step 1: Release guardian
        release_gpus
        log "  Guardian released"

        # Step 2: Wait for GPUs to be clean (< 5GB used)
        wait_gpus_clean
        log "  GPUs verified clean"

        # Expected lines per shard (per-shard, since infer.py splits by st/ed)
        # We'll compute per shard inside the loop; here just log dataset size.
        DJSON="${djson}"
        log "    dataset: ${djson}"

        mkdir -p "${MDIR}"

        # Step 3: Launch 4 shards in parallel (first attempt)
        PIDS=()
        for i in "${!GPUS[@]}"; do
            G=${GPUS[$i]}
            SHLOG="${LOGDIR}/eval_${LABEL}_${bkey}_shard${i}.log"
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

        # Step 4: Wait for all shards to complete
        SHARD_RC=()
        for p in "${PIDS[@]}"; do wait "$p"; SHARD_RC+=($?); done
        log "    ${LABEL}/${bfolder}: inference done. rc=[${SHARD_RC[*]}]"

        # Graceful cleanup
        for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
        sleep 5
        for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done

        # Step 5: Check each shard's completeness; retry failures one by one.
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

        # Step 6: Wait for GPU memory to be released before next benchmark
        log "    Waiting for GPU memory release..."
        wait_gpus_clean
        log "    GPUs clean after ${LABEL}/${bfolder}"

        if [ "${ALL_OK}" -ne 1 ]; then
            log "    !! ${LABEL}/${bfolder} INCOMPLETE — NOT merging/scoring, aborting eval."
            ABORT=1
            break
        fi

        # Step 7: Merge shards and score
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

    [ "${ABORT}" -ne 0 ] && break
done

if [ "${ABORT}" -ne 0 ]; then
    log "================ EVAL ABORTED (incomplete shard) ================"
    exit 1
fi
log "================ ALL EVAL DONE ================"
