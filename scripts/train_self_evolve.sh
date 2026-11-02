#!/bin/bash
# VERSE Stage 3 — Self-Evolve GRPO. The model proposes questions over 3D
# contexts, a deterministic engine (DGE for ScanNet / BFS for maze) verifies them
# and supplies ground truth, the model then solves them with agentic drawing, and
# GRPO updates on the outcome (+ optional engine-verified process rewards:
# maze path-fidelity and VSI grounding). Init policy = the RRS checkpoint; KL is
# anchored to it. batch=16, rollout.n=8, one task type per step.
#
# Requires the `verse` env. Runs on ${VERSE_GPUS}.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/configs/paths.env"
# Load API keys if the user created configs/api_keys.env (see api_keys.env.example).
[[ -f "${HERE}/configs/api_keys.env" ]] && set -a && source "${HERE}/configs/api_keys.env" && set +a
SE="${HERE}/self_evolve"

EXP_NAME="${EXP_NAME:-verse_self_evolve}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-25}"
INIT_POLICY="${INIT_POLICY:-${VERSE_CKPT}/verse_rrs}"
TRAIN_JSONL="${TRAIN_JSONL:-${VERSE_CONTEXT}/full_selfevolve_train_v4.jsonl}"
CKPT_OUT="${VERSE_CKPT}/${EXP_NAME}"
NGPUS="$(echo "${VERSE_GPUS}" | awk -F, '{print NF}')"

# --- Self-evolve knobs (env-gated; defaults reproduce the base self-evolve run) ---
export SCHEDULER="${SCHEDULER:-round_robin}"
export SCHEDULE_FILE="${SCHEDULE_FILE:-${SE}/easy_r1/dataset/full_selfevolve_schedule_v3.json}"
export PROPOSER_MODE="${PROPOSER_MODE:-structured_imitate}"
export MAZE_PROPOSER_MODE="${MAZE_PROPOSER_MODE:-round_robin}"
# Engine-verified process rewards. Official self-evolve defaults w=0.2
# (maze path-fidelity + VSI grounding). Ablation launchers force w=0.
export VERSE_MAZE_FIDELITY_WEIGHT="${VERSE_MAZE_FIDELITY_WEIGHT:-0.2}"
export VERSE_VSI_GROUNDING_WEIGHT="${VERSE_VSI_GROUNDING_WEIGHT:-0.2}"
# Back-compat with the internal SPATIALEVO_* env names still read by the trainer:
export SPATIALEVO_MAZE_FIDELITY_WEIGHT="${VERSE_MAZE_FIDELITY_WEIGHT}"
export SPATIALEVO_VSI_GROUNDING_WEIGHT="${VERSE_VSI_GROUNDING_WEIGHT}"

export CUDA_VISIBLE_DEVICES="${VERSE_GPUS}"
export HF_ENDPOINT="${HF_ENDPOINT}"
export PYTHONPATH="${SE}/run_scripts:${PYTHONPATH:-}"
mkdir -p "${CKPT_OUT}" "${VERSE_OUTPUT}"

# --- Reward-judge API key (from configs/api_keys.env) injected as config overrides ---
KEY_OVERRIDES=()
VLM_KEY="${VERSE_VLM_API_KEY:-}"
if [[ -n "${VLM_KEY}" ]]; then
    RK="worker.reward.reward_function_kwargs"
    for grp in question_quality_kwargs majority_correctness_kwargs; do
        KEY_OVERRIDES+=("${RK}.${grp}.api_key=${VLM_KEY}")
        KEY_OVERRIDES+=("${RK}.${grp}.model=${VERSE_VLM_MODEL:-gpt-4o-mini}")
        KEY_OVERRIDES+=("${RK}.${grp}.service_url=${VERSE_VLM_BASE_URL:-https://api.openai.com/v1}")
    done
    echo "[stage3] LLM reward judges: ENABLED (model=${VERSE_VLM_MODEL:-gpt-4o-mini})"
else
    echo "[stage3] LLM reward judges: DISABLED (no VERSE_VLM_API_KEY; judge terms -> 0)"
fi

# --- Logger: add wandb only if a key is present ---
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
    LOGGER='trainer.logger=["file","wandb"]'
    echo "[stage3] W&B logging: ENABLED"
else
    LOGGER='trainer.logger=["file"]'
    echo "[stage3] W&B logging: disabled (file logger only)"
fi

echo "[stage3] self-evolve GRPO | exp=${EXP_NAME} | init=${INIT_POLICY} | gpus=${VERSE_GPUS} | steps=${MAX_STEPS}"
echo "[stage3] grounding_w=${VERSE_VSI_GROUNDING_WEIGHT}  maze_fidelity_w=${VERSE_MAZE_FIDELITY_WEIGHT}"

cd "${SE}/easy_r1"
conda run -n verse --no-capture-output python3 "${SE}/run_scripts/train_main.py" \
    config=examples/config_phase2_structured_imitate.yaml \
    data.train_files="${TRAIN_JSONL}" \
    worker.actor.model.model_path="${INIT_POLICY}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.save_checkpoint_path="${CKPT_OUT}" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    "${LOGGER}" \
    "${KEY_OVERRIDES[@]}" \
    "$@"
echo "[stage3] done -> ${CKPT_OUT}"
