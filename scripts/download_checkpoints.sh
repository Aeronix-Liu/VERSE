#!/bin/bash
# Download the released VERSE base checkpoints (model-only, HF format):
#   - cold-start SFT  (Stage 1 output)  -> $VERSE_CKPT/verse_cold_start
#   - RRS SFT         (Stage 2 output)  -> $VERSE_CKPT/verse_rrs
# RRS is the init policy (pi_ref) for the self-evolve GRPO stage.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/configs/paths.env"
mkdir -p "${VERSE_CKPT}"

WHICH="${1:-all}"

get () {  # repo  dest
  echo "[ckpt] ${1} -> ${2}  (endpoint=${HF_ENDPOINT})"
  hf download "${1}" --local-dir "${2}"
}

if [[ "${WHICH}" == "cold_start" || "${WHICH}" == "all" ]]; then
  get "${VERSE_HF_COLDSTART_REPO}" "${VERSE_CKPT}/verse_cold_start"
fi
if [[ "${WHICH}" == "rrs" || "${WHICH}" == "all" ]]; then
  get "${VERSE_HF_RRS_REPO}" "${VERSE_CKPT}/verse_rrs"
fi
echo "[ckpt] done."
