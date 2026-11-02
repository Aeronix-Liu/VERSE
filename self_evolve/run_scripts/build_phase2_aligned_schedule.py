#!/usr/bin/env python3
"""Build Phase-2 schedule aligned with VSI-Bench, SPAR-Bench, and MAZE.

Distribution (200 entries, first 100 used with max_steps=100):
  VSI-Bench  (scene mode, 7 types):       70 entries (35%)
  SPAR single (single_image, 6 types):    35 entries (17.5%)
  SPAR _mv    (image_pair, 10 types):      45 entries (22.5%)
  MAZE       (maze, 1 type):              40 entries (20%)
  Non-bench  (single_image, 3 types):     10 entries (5%)

Output: easy_r1/dataset/full_selfevolve_schedule_v2.json
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
EASY_R1 = os.path.join(HERE, "..", "easy_r1")
OUT_PATH = os.path.join(EASY_R1, "dataset", "full_selfevolve_schedule_v2.json")

random.seed(42)

# --- VSI-Bench (scene mode) ---
vsi_tasks = {
    "object_counting": 9,
    "absolute_distance": 13,
    "object_size": 15,
    "room_size": 4,
    "relative_distance": 11,
    "relative_direction_hard": 15,
    "route_planning": 3,
}

# --- SPAR single-image ---
spar_single_tasks = {
    "spar_depth_prediction_oc": 6,
    "spar_depth_prediction_oo": 6,
    "spar_distance_prediction_oc": 6,
    "spar_distance_prediction_oo": 6,
    "spar_distance_infer_center_oo": 5,
    "spar_obj_spatial_relation_oo": 6,
}

# --- SPAR image-pair (_mv variants only; position_matching/view_change/camera_motion
#     removed — they need bbox coords / frame_ids that the Proposer can't generate) ---
spar_mv_tasks = {
    "spar_depth_prediction_oc_mv": 6,
    "spar_depth_prediction_oo_mv": 6,
    "spar_distance_prediction_oc_mv": 6,
    "spar_distance_prediction_oo_mv": 5,
    "spar_distance_infer_center_oo_mv": 5,
    "spar_obj_spatial_relation_oc_mv": 5,
    "spar_obj_spatial_relation_oo_mv": 5,
}

# --- MAZE ---
maze_tasks = {
    "maze_navigation": 40,
}

# --- Non-benchmark (for generalization, ~5%) ---
non_bench_tasks = {
    "single_image_relative_direction": 4,
    "distance_cam_obj": 3,
    "depth_order_obj_obj": 3,
}

# --- Build schedule ---
schedule = []
for task, count in {**vsi_tasks, **spar_single_tasks, **spar_mv_tasks,
                    **maze_tasks, **non_bench_tasks}.items():
    schedule.extend([task] * count)

random.shuffle(schedule)

# Verify counts
from collections import Counter
counts = Counter(schedule)
print(f"Schedule: {len(schedule)} entries")
for t, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {n}")

# Modality breakdown
print("\nModality breakdown (first 100):")
first_100 = Counter()
for t in schedule[:100]:
    if t == "maze_navigation":
        first_100["maze"] += 1
    elif t.startswith("spar_") and ("_mv" in t or t in ("spar_position_matching", "spar_view_change_infer", "spar_camera_motion_infer")):
        first_100["image_pair"] += 1
    elif t.startswith("spar_"):
        first_100["single_image"] += 1
    elif t in ("single_image_relative_direction", "distance_cam_obj", "depth_order_obj_obj"):
        first_100["single_image (non-bench)"] += 1
    else:
        first_100["scene"] += 1
for m, c in sorted(first_100.items(), key=lambda x: -x[1]):
    print(f"  {m}: {c}")

with open(OUT_PATH, "w") as f:
    json.dump(schedule, f, indent=2)
print(f"\n[done] wrote {OUT_PATH}")
