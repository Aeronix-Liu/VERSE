#!/bin/bash
# Download VERSE SFT data (cold-start + reflective rejection sampling).
#
# We release the CLEANED SFT ANNOTATIONS (sharegpt json, *_clean.json) + the
# LLaMA-Factory dataset_info.json. The underlying IMAGES are ViLaSR's public
# datasets (SR_91k / GPT4Scene / VQA / maze); fetch them with
# scripts/download_vilasr_images.sh and place them so the relative "images"
# paths in the annotations resolve under $VERSE_DATA.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/configs/paths.env"
mkdir -p "${VERSE_DATA}"

echo "[data] endpoint=${HF_ENDPOINT}  repo=${VERSE_HF_DATA_REPO}  ->  ${VERSE_DATA}"
hf download "${VERSE_HF_DATA_REPO}" --repo-type dataset --local-dir "${VERSE_DATA}"

# Ensure the LLaMA-Factory registry is present (also shipped in the repo).
cp -f "${HERE}/sft/dataset_info.json" "${VERSE_DATA}/dataset_info.json"

echo "[data] SFT annotations ready under ${VERSE_DATA}"
echo "[data] NEXT: bash scripts/download_vilasr_images.sh   (fetch the images the annotations reference)"
