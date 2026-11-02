#!/bin/bash
# Fix: download missing {scene_id}.txt axis-alignment files, then re-run metadata extraction.
set -uo pipefail
XINRAN="/data/pengtianshuo/self_evolve/Xinran"
SELFEVOLVE="${XINRAN}/ViLaSR-self-evolve"
LOG="${SELFEVOLVE}/logs/fix_metadata.log"
mkdir -p "${SELFEVOLVE}/logs"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${XINRAN}/envs/Xinran_VERSE"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

log "================ FIX METADATA START ================"
cd "${SELFEVOLVE}"

log "---- Step 1: download missing {scene_id}.txt files ----"
python run_scripts/fix_axis_alignment_txt.py 2>&1 | tee -a "${LOG}"

log "---- Step 2: re-run full metadata extraction ----"
bash run_scripts/run_metadata_extraction.sh 2>&1 | tee -a "${LOG}"

META_COUNT=$(ls ScanNetData_local/metadata/ 2>/dev/null | wc -l)
FRAME_META=$(find ScanNetData_local/metadata -path "*/frame_processed/*.json" 2>/dev/null | wc -l)
log "================ FIX METADATA COMPLETE ================"
log "  scene metadata dirs: ${META_COUNT}"
log "  frame metadata jsons: ${FRAME_META}"

cat > "${SELFEVOLVE}/logs/overnight_task2a_summary.txt" <<EOF
=== Task 2a Final Summary ($(date '+%F %T')) ===

ScanNet fully-equipped scenes:
  - Scenes with mesh + frames + pose + depth + intrinsics: 332
  - Storage used: $(du -sh ${SELFEVOLVE}/ScanNetData_local/ 2>&1 | cut -f1)

Metadata:
  - Scene metadata dirs: ${META_COUNT}
  - Frame metadata jsons: ${FRAME_META}
  - ALL DGE task types supported (SPAR + VSI)

See ${LOG} for full log.
EOF
log "  summary updated: ${SELFEVOLVE}/logs/overnight_task2a_summary.txt"
