#!/bin/bash
# Extract DGE metadata for the small ScanNet subset (CPU-only, guardian-safe).
# Run AFTER prep_scannet_subset.py has extracted frames/pose/depth from the 25k zip
# and the per-scene mesh files (vh_clean_2.ply + segs + aggregation) are downloaded.
set -euo pipefail

XINRAN_ROOT=/data/pengtianshuo/self_evolve/Xinran
REPO_ROOT="${XINRAN_ROOT}/ViLaSR-self-evolve"
SCANNET_DATA="${REPO_ROOT}/ScanNetData_local"
TRAIN_ROOT="${SCANNET_DATA}/train"
LOG="${REPO_ROOT}/logs/metadata_extraction.log"
mkdir -p "${REPO_ROOT}/logs"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${XINRAN_ROOT}/envs/Xinran_VERSE"

READY_SCENES="${SCANNET_DATA}/downloads/scenes_ready_for_metadata.txt"
: > "${READY_SCENES}"
for scene_dir in "${TRAIN_ROOT}"/scene*; do
    [[ -d "${scene_dir}" ]] || continue
    sid=$(basename "${scene_dir}")
    ply="${scene_dir}/${sid}_vh_clean_2.ply"
    segs="${scene_dir}/${sid}_vh_clean_2.0.010000.segs.json"
    if [[ -f "${ply}" && $(stat -c%s "${ply}") -ge 1000 && -f "${segs}" && $(stat -c%s "${segs}") -ge 100 \
          && -d "${scene_dir}/frame_processed" && -d "${scene_dir}/pose" ]]; then
        echo "${sid}" >> "${READY_SCENES}"
    fi
done
sort -u -o "${READY_SCENES}" "${READY_SCENES}"
echo "Scenes ready for metadata extraction: $(wc -l < "${READY_SCENES}")"

python3 "${REPO_ROOT}/run_scripts/metadata_extractor_small_mesh.py" \
    --dataset-root "${SCANNET_DATA}" \
    --split train \
    --frame-type frame_processed \
    --scene-ids-file "${READY_SCENES}" \
    --label-map-file "${REPO_ROOT}/data/scannet_process/scannet_metadata/scannetv2-labels-v2.tsv" \
    --workers 8 \
    2>&1 | tee -a "${LOG}"

echo "Metadata dirs: $(ls "${SCANNET_DATA}/metadata" 2>/dev/null | wc -l)"
echo "Frame metadata jsons: $(find "${SCANNET_DATA}/metadata" -name '*.json' 2>/dev/null | wc -l)"
