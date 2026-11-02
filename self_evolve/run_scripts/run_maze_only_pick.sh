#!/bin/bash
# Maze-only GRPO self-evolve (pick-letter Proposer) on 4xA800 (GPU 4-7).
#
#   Policy init : checkpoints/vilasr_reflective_4gpu (RRS final flat-HF)
#   Mode        : MAZE_PROPOSER_MODE=pick_letter (Questioner only picks A/B/C/D;
#                 system fills the verified shortest path automatically)
#   Data        : dataset/maze_only_train.jsonl (300 maze contexts)
#   Loop        : two-round self-evolve (Questioner picks letter -> MazeEngine
#                 constructs valid question -> agentic Solver answers)
#
# Env overrides: MAX_STEPS / SAVE_FREQ / EXP_NAME / EXTRA_OVERRIDES
# Smoke: MAX_STEPS=10 SAVE_FREQ=-1 EXP_NAME=maze_pick_smoke bash run_scripts/run_maze_only_pick.sh

XINRAN_ROOT=/data/pengtianshuo/self_evolve/Xinran
REPO_ROOT="${XINRAN_ROOT}/ViLaSR-self-evolve"
EASY_R1="${REPO_ROOT}/easy_r1"
SCRIPTS_DIR="${REPO_ROOT}/run_scripts"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TRAIN_CONFIG="${CONFIG:-examples/config_maze_only.yaml}"
TRAIN_JSONL="${TRAIN_JSONL:-./dataset/maze_only_train.jsonl}"
EXP_NAME="${EXP_NAME:-vilasr_maze_only_pick}"
LOGGER_JSON="${LOGGER_JSON:-[\"file\",\"wandb\"]}"
CKPT_ROOT="${XINRAN_ROOT}/checkpoints/${EXP_NAME}"

# --- Activate conda env ---
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${XINRAN_ROOT}/envs/Xinran_VERSE"

# --- GPU selection: train on GPU 4-7 ONLY ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')

# --- Maze Proposer mode: pick_letter (Questioner only picks A/B/C/D) ---
export MAZE_PROPOSER_MODE="${MAZE_PROPOSER_MODE:-pick_letter}"

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
[[ -n "${MAX_STEPS:-}" ]]  && OVERRIDES+=("trainer.max_steps=${MAX_STEPS}")
[[ -n "${SAVE_FREQ:-}" ]]  && OVERRIDES+=("trainer.save_freq=${SAVE_FREQ}")
[[ -n "${SAVE_LIMIT:-}" ]] && OVERRIDES+=("trainer.save_limit=${SAVE_LIMIT}")
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    OVERRIDES+=(${EXTRA_OVERRIDES})
fi

echo "==> Launching maze-only GRPO self-evolve (pick-letter mode)"
echo "    repo      : ${REPO_ROOT}"
echo "    config    : ${TRAIN_CONFIG}"
echo "    jsonl     : ${TRAIN_JSONL}"
echo "    maze mode : ${MAZE_PROPOSER_MODE}"
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
