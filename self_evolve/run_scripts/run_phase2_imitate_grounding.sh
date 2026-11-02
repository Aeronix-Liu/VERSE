#!/bin/bash
# grounding: 0811-no-spar structured_imitate self-evolve GRPO + engine-verified
# VSI grounding process reward, on GPU 4-7.
#
# This is the imitate-0811-no-spar recipe with ONE added ingredient: a dense
# process-reward term for scene/VSI tasks that scores how faithfully the Solver's
# Object-Mapper boxes localize the queried objects (GT per-frame bboxes). VSI is
# otherwise outcome-only; this targets the "draw-then-guess" decoupling.
#   scene valid: r = 0.1*fmt + (0.9-w)*acc + w*grounding   (w=SPATIALEVO_VSI_GROUNDING_WEIGHT)
# Everything else is IDENTICAL to 0811-no-spar: batch_size=16, rollout.n=8,
# one task per step, proposer update ON, KL anchored to rrs-744, NO selection/
# search switches. Base = rrs-744 unless LOAD_CKPT is set (warm-start).
#
# Env overrides:
#   MAX_STEPS (100) / SAVE_FREQ (25) / SAVE_LIMIT (8) / EXP_NAME
#   SPATIALEVO_VSI_GROUNDING_WEIGHT (0.2)   -- 0 = pure diagnostic
#   LOAD_CKPT   -- optional: resume/warm-start from this checkpoint dir
#   SCHEDULE_FILE / TRAIN_JSONL / EXTRA_OVERRIDES / CUDA_VISIBLE_DEVICES

XINRAN_ROOT=/data/pengtianshuo/self_evolve/Xinran
REPO_ROOT="${XINRAN_ROOT}/ViLaSR-self-evolve"
EASY_R1="${REPO_ROOT}/easy_r1"
SCRIPTS_DIR="${REPO_ROOT}/run_scripts"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TRAIN_CONFIG="${CONFIG:-examples/config_phase2_structured_imitate.yaml}"
TRAIN_JSONL="${TRAIN_JSONL:-./dataset/full_selfevolve_train_v4.jsonl}"
EXP_NAME="${EXP_NAME:-vilasr-se-grounding-100step}"
LOGGER_JSON="${LOGGER_JSON:-[\"file\",\"wandb\"]}"
CKPT_ROOT="${XINRAN_ROOT}/checkpoints/${EXP_NAME}"

# --- Defaults for this run ---
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_FREQ="${SAVE_FREQ:-25}"
SAVE_LIMIT="${SAVE_LIMIT:-8}"

# --- Grounding process reward (the only new ingredient) ---
export SPATIALEVO_VSI_GROUNDING_WEIGHT="${SPATIALEVO_VSI_GROUNDING_WEIGHT:-0.2}"
echo "==> SPATIALEVO_VSI_GROUNDING_WEIGHT=${SPATIALEVO_VSI_GROUNDING_WEIGHT}"

# --- Activate conda env ---
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${XINRAN_ROOT}/envs/Xinran_VERSE"

# --- GPU selection: train on GPU 4-7 ONLY ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

# --- Scheduler: deterministic round-robin over the v3 no-SPAR schedule ---
export SCHEDULER="round_robin"
export SCHEDULE_FILE="${SCHEDULE_FILE:-${EASY_R1}/dataset/full_selfevolve_schedule_v3.json}"
echo "==> Scheduler: ${SCHEDULER}, schedule: ${SCHEDULE_FILE}"

# --- Maze Proposer mode: round_robin ---
export MAZE_PROPOSER_MODE="${MAZE_PROPOSER_MODE:-round_robin}"

# --- Phase-2: structured_imitate Proposer ---
export PROPOSER_MODE="${PROPOSER_MODE:-structured_imitate}"
echo "==> PROPOSER_MODE=${PROPOSER_MODE}"

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
if [[ -n "${LOAD_CKPT:-}" ]]; then
    OVERRIDES+=("trainer.load_checkpoint_path=${LOAD_CKPT}")
    echo "==> WARM-START / RESUME from: ${LOAD_CKPT}"
fi
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    OVERRIDES+=(${EXTRA_OVERRIDES})
fi

echo "==> Launching grounding self-evolve GRPO (0811-no-spar + VSI grounding process reward)"
echo "    repo      : ${REPO_ROOT}"
echo "    config    : ${TRAIN_CONFIG}"
echo "    jsonl     : ${TRAIN_JSONL}"
echo "    scheduler : ${SCHEDULER} (${SCHEDULE_FILE})"
echo "    proposer  : PROPOSER_MODE=${PROPOSER_MODE} / MAZE_PROPOSER_MODE=${MAZE_PROPOSER_MODE}"
echo "    grounding : w=${SPATIALEVO_VSI_GROUNDING_WEIGHT}"
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
