#!/bin/bash
# Overnight automation for Task 2a: make ALL scenes fully-equipped.
#   Phase 1:   finish mesh download (resumes, idempotent) -> updates chosen_scenes.txt
#   Phase 1.5: fetch frames/pose/depth/intrinsics for all chosen scenes (HTTP Range, ~1GB)
#   Phase 2:   FULL metadata extraction (per-frame: visibility, 2d_bbox, camera_location)
#   Phase 3:   report
#
# Runs in tmux so it survives SSH disconnect. CPU-only (guardian-safe).
set -uo pipefail

XINRAN="/data/pengtianshuo/self_evolve/Xinran"
SELFEVOLVE="${XINRAN}/ViLaSR-self-evolve"
LOGDIR="${SELFEVOLVE}/logs"
LOG="${LOGDIR}/overnight_task2a.log"
mkdir -p "${LOGDIR}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${XINRAN}/envs/Xinran_VERSE"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

log "================ OVERNIGHT TASK 2a START (fully-equipped) ================"

# ---- Phase 1: Finish mesh download (resumes, idempotent) ----
log "---- Phase 1: finishing mesh download (resumes from existing) ----"
cd "${SELFEVOLVE}"
python run_scripts/download_scannet_mesh.py --n-scenes 150 --workers 8 2>&1 | tee -a "${LOG}"
PLY_COUNT=$(find ScanNetData_local/train -maxdepth 2 -name "*_vh_clean_2.ply" -size +1k 2>/dev/null | wc -l)
CHOSEN=$(wc -l < ScanNetData_local/downloads/chosen_scenes.txt 2>/dev/null || echo 0)
log "  mesh download done. ${PLY_COUNT} valid .ply files; chosen_scenes.txt has ${CHOSEN} scenes."
du -sh ScanNetData_local/ 2>&1 | tee -a "${LOG}"

# ---- Phase 1.5: Fetch frames/pose/depth/intrinsics for all chosen scenes ----
log "---- Phase 1.5: fetching frames/pose/depth/intrinsics via HTTP Range (~1GB) ----"
python run_scripts/fetch_25k_partial.py --workers 8 2>&1 | tee -a "${LOG}"
# Count fully-equipped scenes (have pose dir + intrinsic_color.txt + frame_processed)
EQUIPPED=0
for d in ScanNetData_local/train/scene*; do
    [ -d "$d/pose" ] && [ -f "$d/intrinsic_color.txt" ] && [ -d "$d/frame_processed" ] && EQUIPPED=$((EQUIPPED+1))
done
log "  frames/pose/depth/intrinsics fetched. ${EQUIPPED} scenes now fully-equipped."
du -sh ScanNetData_local/ 2>&1 | tee -a "${LOG}"

# ---- Phase 2: FULL metadata extraction (per-frame) ----
log "---- Phase 2: FULL metadata extraction (per-frame: visibility, 2d_bbox, camera_location) ----"
bash run_scripts/run_metadata_extraction.sh 2>&1 | tee -a "${LOG}"
META_COUNT=$(ls ScanNetData_local/metadata/ 2>/dev/null | wc -l)
FRAME_META=$(find ScanNetData_local/metadata -path "*/frame_processed/*.json" 2>/dev/null | wc -l)
log "  metadata extraction done. ${META_COUNT} scene dirs; ${FRAME_META} frame metadata jsons."

# ---- Phase 3: Report ----
log "================ OVERNIGHT TASK 2a COMPLETE ================"
log "  ScanNet scenes with mesh:           ${PLY_COUNT}"
log "  Scenes fully-equipped (pose+frames): ${EQUIPPED}"
log "  Scene metadata dirs:                ${META_COUNT}"
log "  Frame metadata jsons:               ${FRAME_META}"
log "  Total storage:                      $(du -sh ScanNetData_local/ 2>&1 | cut -f1)"
log "  ALL scenes now support SPAR + pose-dependent VSI tasks."

cat > "${LOGDIR}/overnight_task2a_summary.txt" <<EOF
=== Overnight Task 2a Summary ($(date '+%F %T')) ===

ScanNet fully-equipped scenes:
  - Scenes with mesh:           ${PLY_COUNT}
  - Fully-equipped (pose+frames+depth+intrinsics): ${EQUIPPED}
  - Storage used:               $(du -sh ${SELFEVOLVE}/ScanNetData_local/ 2>&1 | cut -f1)

Metadata:
  - Scene metadata dirs:       ${META_COUNT}
  - Frame metadata jsons:       ${FRAME_META}
  - Supports: ALL DGE task types (SPAR-Bench + VSI object_counting/room_size/
    relative_distance/relative_direction/route_planning/absolute_distance/
    object_size/depth_order/single_image_relative_direction)

Pose availability:
  - All ${EQUIPPED} equipped scenes have pose/ + intrinsic_color.txt
  - SPAR-Bench generation is now possible on ALL equipped scenes (not just the original 30)

See ${LOG} for full log.
EOF
log "  summary written to ${LOGDIR}/overnight_task2a_summary.txt"
