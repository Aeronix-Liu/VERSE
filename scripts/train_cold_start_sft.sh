#!/bin/bash
# VERSE Stage 1 — Cold-start SFT (LLaMA-Factory full SFT, DeepSpeed ZeRO-3).
# Renders sft/cold_start.yaml with paths from configs/paths.env, then launches.
# Requires the `verse-sft` env + LLaMA-Factory (see scripts/setup_env.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/configs/paths.env"

LF_DIR="${LLAMAFACTORY_DIR:-${VERSE_ROOT}/third_party/LLaMA-Factory}"
NPROC="${NPROC_PER_NODE:-$(echo "${VERSE_GPUS}" | awk -F, '{print NF}')}"
RENDERED="${VERSE_OUTPUT}/sft_cold_start.rendered.yaml"
mkdir -p "${VERSE_OUTPUT}"

# Fill __VAR__ placeholders from the environment (no hard-coded machine paths).
sed -e "s|__VERSE_BASE_MODEL__|${VERSE_BASE_MODEL}|g" \
    -e "s|__VERSE_ROOT__|${VERSE_ROOT}|g" \
    -e "s|__VERSE_DATA__|${VERSE_DATA}|g" \
    -e "s|__VERSE_CKPT__|${VERSE_CKPT}|g" \
    "${HERE}/sft/cold_start.yaml" > "${RENDERED}"

echo "[stage1] cold-start SFT | gpus=${VERSE_GPUS} | config=${RENDERED}"
export CUDA_VISIBLE_DEVICES="${VERSE_GPUS}"
export FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="${NPROC}"
conda run -n verse-sft --no-capture-output \
    llamafactory-cli train "${RENDERED}" "$@"
echo "[stage1] done -> ${VERSE_CKPT}/verse_cold_start"
