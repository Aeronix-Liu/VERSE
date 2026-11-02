#!/bin/bash
# VERSE Stage 2 — Reflective Rejection Sampling SFT (continues from Stage 1).
# Renders sft/reflective.yaml from configs/paths.env, then launches LLaMA-Factory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/configs/paths.env"

NPROC="${NPROC_PER_NODE:-$(echo "${VERSE_GPUS}" | awk -F, '{print NF}')}"
RENDERED="${VERSE_OUTPUT}/sft_reflective.rendered.yaml"
mkdir -p "${VERSE_OUTPUT}"

sed -e "s|__VERSE_BASE_MODEL__|${VERSE_BASE_MODEL}|g" \
    -e "s|__VERSE_ROOT__|${VERSE_ROOT}|g" \
    -e "s|__VERSE_DATA__|${VERSE_DATA}|g" \
    -e "s|__VERSE_CKPT__|${VERSE_CKPT}|g" \
    "${HERE}/sft/reflective.yaml" > "${RENDERED}"

echo "[stage2] reflective SFT | gpus=${VERSE_GPUS} | config=${RENDERED}"
export CUDA_VISIBLE_DEVICES="${VERSE_GPUS}"
export FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="${NPROC}"
conda run -n verse-sft --no-capture-output \
    llamafactory-cli train "${RENDERED}" "$@"
echo "[stage2] done -> ${VERSE_CKPT}/verse_rrs  (this is the self-evolve init policy)"
