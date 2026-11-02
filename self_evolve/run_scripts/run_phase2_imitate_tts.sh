#!/bin/bash
# imitate-TTS: self-evolve GRPO + Verifier-Guided rollout Search (VGS-GRPO).
#
# This is run_phase2_imitate_selection.sh (same base=rrs-744, data v4, schedule
# v3, and the 2026-08-15 selection/process-reward/bank flags) PLUS a train-time
# search stage on the Round-2 solver:
#
#   SPATIALEVO_TTS_SEARCH=1  — for every trained step whose task is in
#     SPATIALEVO_TTS_TASKS, widen the solver frontier from rollout.n to
#     rollout.n * SPATIALEVO_TTS_BRANCH i.i.d. agentic rollouts per context,
#     score each with the DETERMINISTIC engine (answer correctness + maze
#     Path-Tracer trajectory fidelity), then engine-SELECT rollout.n rollouts
#     per group (class-balanced + fidelity + variance fallback; see
#     easy_r1/verl/trainer/tts_search.py) to seed the standard GRPO update.
#     GRPO itself (KL anchor rrs-744, asymmetric clip, token-mean) is unchanged.
#
# Clean ablation partner: run_phase2_imitate_selection.sh (identical minus
# SPATIALEVO_TTS_SEARCH), i.e. the selection-50 run.
#
# Env overrides: MAX_STEPS / SAVE_FREQ / EXP_NAME / GPUS / SPATIALEVO_TTS_* ...

XINRAN_ROOT=/data/pengtianshuo/self_evolve/Xinran
REPO_ROOT="${XINRAN_ROOT}/ViLaSR-self-evolve"
EASY_R1="${REPO_ROOT}/easy_r1"
SCRIPTS_DIR="${REPO_ROOT}/run_scripts"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TRAIN_CONFIG="${CONFIG:-examples/config_phase2_structured_imitate.yaml}"
TRAIN_JSONL="${TRAIN_JSONL:-./dataset/full_selfevolve_train_v4.jsonl}"
EXP_NAME="${EXP_NAME:-vilasr-se-tts-search-50step}"
LOGGER_JSON="${LOGGER_JSON:-[\"file\",\"wandb\"]}"
CKPT_ROOT="${XINRAN_ROOT}/checkpoints/${EXP_NAME}"

MAX_STEPS="${MAX_STEPS:-50}"
SAVE_FREQ="${SAVE_FREQ:-25}"
SAVE_LIMIT="${SAVE_LIMIT:-6}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${XINRAN_ROOT}/envs/Xinran_VERSE"

# --- GPU selection (default 4-7; override GPUS for debug, e.g. GPUS=0) ---
export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-4,5,6,7}}"
NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

export SCHEDULER="round_robin"
export SCHEDULE_FILE="${SCHEDULE_FILE:-${EASY_R1}/dataset/full_selfevolve_schedule_v3.json}"
export MAZE_PROPOSER_MODE="${MAZE_PROPOSER_MODE:-round_robin}"
export PROPOSER_MODE="${PROPOSER_MODE:-structured_imitate}"

# --- 2026-08-15 selection + process-reward + bank flags (same as selection-50) ---
export SPATIALEVO_QUESTION_RESAMPLE="${SPATIALEVO_QUESTION_RESAMPLE:-1}"
export SPATIALEVO_QUESTION_RESAMPLE_TRIES="${SPATIALEVO_QUESTION_RESAMPLE_TRIES:-8}"
export SPATIALEVO_ROUND1_SKIP_UPDATE="${SPATIALEVO_ROUND1_SKIP_UPDATE:-1}"
export SPATIALEVO_MAZE_FIDELITY_WEIGHT="${SPATIALEVO_MAZE_FIDELITY_WEIGHT:-0.2}"
export SPATIALEVO_TRAJ_BANK="${SPATIALEVO_TRAJ_BANK:-1}"
export SPATIALEVO_TRAJ_BANK_ACC_THRESHOLD="${SPATIALEVO_TRAJ_BANK_ACC_THRESHOLD:-0.8}"
export SPATIALEVO_TRAJ_BANK_FIDELITY_THRESHOLD="${SPATIALEVO_TRAJ_BANK_FIDELITY_THRESHOLD:-0.6}"

# --- 2026-08-18 Verifier-Guided rollout Search (VGS-GRPO) flags ---
export SPATIALEVO_TTS_SEARCH="${SPATIALEVO_TTS_SEARCH:-1}"
export SPATIALEVO_TTS_BRANCH="${SPATIALEVO_TTS_BRANCH:-2}"
export SPATIALEVO_TTS_TASKS="${SPATIALEVO_TTS_TASKS:-maze,vsi}"
export SPATIALEVO_TTS_FIDELITY_WEIGHT="${SPATIALEVO_TTS_FIDELITY_WEIGHT:-0.5}"

echo "==> Scheduler: ${SCHEDULER} (${SCHEDULE_FILE})"
echo "==> PROPOSER_MODE=${PROPOSER_MODE} / MAZE_PROPOSER_MODE=${MAZE_PROPOSER_MODE}"
echo "==> RESAMPLE=${SPATIALEVO_QUESTION_RESAMPLE} SKIP_R1_UPDATE=${SPATIALEVO_ROUND1_SKIP_UPDATE}"
echo "==> MAZE_FIDELITY_WEIGHT=${SPATIALEVO_MAZE_FIDELITY_WEIGHT} TRAJ_BANK=${SPATIALEVO_TRAJ_BANK}"
echo "==> TTS_SEARCH=${SPATIALEVO_TTS_SEARCH} BRANCH=${SPATIALEVO_TTS_BRANCH} TASKS=${SPATIALEVO_TTS_TASKS} FID_W=${SPATIALEVO_TTS_FIDELITY_WEIGHT}"

# --- Release guardian occupy on ONLY the training GPUs (never --stop) ---
GUARDIAN_DIR="/data/pengtianshuo/self_evolve/dlc_shell"
if [[ -f "${GUARDIAN_DIR}/gpu_guardian.py" ]]; then
    echo "==> Releasing occupy on ONLY GPU(s) ${CUDA_VISIBLE_DEVICES}"
    GUARDIAN_DIR="${GUARDIAN_DIR}" TRAIN_GPUS="${CUDA_VISIBLE_DEVICES}" python3 - <<'PYEOF' || true
import os, sys
sys.path.insert(0, os.environ["GUARDIAN_DIR"])
import gpu_guardian as g
gpus = [int(x) for x in os.environ["TRAIN_GPUS"].split(",") if x.strip()]
g.stop_occupy_on_gpus(gpus)
print(f"[run] stopped occupy on GPUs {gpus}")
PYEOF
fi
sleep 2

# --- Hold GPUs during bootstrap ---
RESERVE_PID=""
_stop_reservation() {
    if [[ -n "${RESERVE_PID}" ]] && kill -0 "${RESERVE_PID}" 2>/dev/null; then
        kill -TERM "${RESERVE_PID}" 2>/dev/null || true
        wait "${RESERVE_PID}" 2>/dev/null || true
    fi
    RESERVE_PID=""
}
trap _stop_reservation EXIT
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    python3 "${SCRIPTS_DIR}/gpu_training_reservation.py" &
RESERVE_PID=$!
echo "==> GPU reservation holder pid=${RESERVE_PID}"
sleep 5

export SWANLAB_MODE=disabled
export WANDB_DIR="${REPO_ROOT}/logs/wandb"
export WANDB_SILENT=true
export WANDB_START_METHOD=thread
mkdir -p "${WANDB_DIR}"
[[ -f "${REPO_ROOT}/config/wandb.env" ]] && source "${REPO_ROOT}/config/wandb.env"
export SPATIALEVO_ATTN_IMPLEMENTATION=sdpa
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export FLASH_ATTENTION_SKIP_CUDA_CHECK=1
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}"
export RAY_TMPDIR=/dev/shm/ray_tmp
mkdir -p "${RAY_TMPDIR}"
export TQDM_DISABLE=1
export TQDM_MININTERVAL=3600
export SPATIALEVO_SAMPLE_LOG_FREQ="${SPATIALEVO_SAMPLE_LOG_FREQ:-20}"

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TS}.log"
echo "${LOG_FILE}" > "${LOG_DIR}/latest_${EXP_NAME}.logpath"

OVERRIDES=()
OVERRIDES+=("trainer.max_steps=${MAX_STEPS}")
OVERRIDES+=("trainer.save_freq=${SAVE_FREQ}")
OVERRIDES+=("trainer.save_limit=${SAVE_LIMIT}")
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    OVERRIDES+=(${EXTRA_OVERRIDES})
fi

echo "==> Launching imitate-TTS self-evolve GRPO (VGS)"
echo "    jsonl     : ${TRAIN_JSONL}"
echo "    gpus      : ${CUDA_VISIBLE_DEVICES} (n=${NGPUS})"
echo "    ckpt      : ${CKPT_ROOT}"
echo "    overrides : ${OVERRIDES[*]:-<none>}"
echo "    log       : ${LOG_FILE}"

cd "${EASY_R1}"

python3 "${SCRIPTS_DIR}/train_main.py" \
    config="${TRAIN_CONFIG}" \
    data.train_files="${TRAIN_JSONL}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.save_checkpoint_path="${CKPT_ROOT}" \
    trainer.n_gpus_per_node="${NGPUS}" \
    "trainer.logger=${LOGGER_JSON}" \
    "${OVERRIDES[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
TRAIN_RC=${PIPESTATUS[0]}

echo "==> Training process exited rc=${TRAIN_RC}"
echo "==> Checkpoints under: ${CKPT_ROOT}"
ls -d "${CKPT_ROOT}"/global_step_*/ 2>/dev/null || echo "    (no global_step_* dirs)"
echo "==> Done (log: ${LOG_FILE})."
