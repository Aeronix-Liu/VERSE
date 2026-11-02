#!/bin/bash
# Download the VERSE self-evolve CONTEXT data: the 3D contexts the proposer
# builds questions from — ScanNet frame subsets + per-frame metadata (for the
# deterministic engine) and the maze image pool — plus the training manifest
# (full_selfevolve_train_v4.jsonl) and round-robin schedule (schedule_v3.json).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/configs/paths.env"
mkdir -p "${VERSE_CONTEXT}"

echo "[context] endpoint=${HF_ENDPOINT}  repo=${VERSE_HF_CTX_REPO}  ->  ${VERSE_CONTEXT}"
hf download "${VERSE_HF_CTX_REPO}" --repo-type dataset --local-dir "${VERSE_CONTEXT}"

echo "[context] done. Expected layout:"
echo "  ${VERSE_CONTEXT}/ScanNetData/{train,metadata}/...   (ScanNet frames + per-frame GT bboxes)"
echo "  ${VERSE_CONTEXT}/maze_pool/...                       (maze images + metadata)"
echo "  ${VERSE_CONTEXT}/full_selfevolve_train_v4.jsonl      (training manifest: image paths only)"
echo "  ${VERSE_CONTEXT}/full_selfevolve_schedule_v3.json    (per-step task schedule)"
echo "[context] The manifest references images by path; point the trainer's data.image_root at ${VERSE_CONTEXT}."
