#!/bin/bash
# Fetch the SFT images referenced by the VERSE annotations. These are the
# PUBLIC ViLaSR training datasets (SR_91k, GPT4Scene, VQA, maze) — we do not
# re-host them. See the ViLaSR release for the authoritative source:
#   https://gh-proxy.com/https://github.com/AntResearchNLP/ViLaSR
#   HuggingFace: search "ViLaSR" datasets by the ViLaSR authors.
#
# After download, arrange the images so the relative "images" paths inside
# $VERSE_DATA/{cold_start,reflective_rejection_sampling}/*_clean.json
# resolve, i.e. under:
#   $VERSE_DATA/cold_start/{SR_91k,GPT4Scene,vqa,maze}/...
#   $VERSE_DATA/reflective_rejection_sampling/{SR_91k,GPT4Scene,vqa,maze}/...
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/configs/paths.env"
mkdir -p "${VERSE_DATA}"

# Set VILASR_HF_DATA_REPO to the ViLaSR image dataset repo id to auto-download,
# or place the images manually as described above.
VILASR_HF_DATA_REPO="${VILASR_HF_DATA_REPO:-}"
if [[ -z "${VILASR_HF_DATA_REPO}" ]]; then
  echo "[images] VILASR_HF_DATA_REPO not set."
  echo "[images] Obtain the ViLaSR SFT images from the official ViLaSR release and place them"
  echo "         under ${VERSE_DATA}/{cold_start,reflective_rejection_sampling}/{SR_91k,GPT4Scene,vqa,maze}/"
  echo "[images] Then verify a few paths resolve, e.g.:"
  echo "         python - <<'PY'"
  echo "         import json,os; d=json.load(open('${VERSE_DATA}/cold_start/vqa_clean.json'))"
  echo "         p=os.path.join('${VERSE_DATA}', d[0]['images'][0]); print(p, os.path.exists(p))"
  echo "         PY"
  exit 0
fi

echo "[images] downloading ViLaSR images from ${VILASR_HF_DATA_REPO} (endpoint=${HF_ENDPOINT})"
hf download "${VILASR_HF_DATA_REPO}" --repo-type dataset --local-dir "${VERSE_DATA}"
echo "[images] done."
