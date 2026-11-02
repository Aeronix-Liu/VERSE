#!/usr/bin/env python3
"""Generate the deterministic schedule AND the modality-aligned ordered dataset
for the full self-evolve training.

Why an ordered dataset?
  The round-robin scheduler pins one task type per step (so every question in a
  step shares the same type). To honour that without wasting sampled contexts,
  we pre-order the dataset so that each consecutive block of `batch_size`
  contexts all share the modality required by that step's scheduled task. With
  `shuffle: false` in the config, the dataloader reads sequentially and every
  step naturally receives a modality-homogeneous batch.

Outputs:
  easy_r1/dataset/full_selfevolve_schedule.json   (list[str], len=steps)
  easy_r1/dataset/full_selfevolve_train.jsonl      (steps*batch_size entries)

Compatible task types (our contexts are 1 maze image or 1 scannet frame):
  maze  -> "maze_navigation"
  vsi   -> single_image VSI tasks (1 frame)
  spar  -> single-frame SPAR tasks (1 frame)
"""
import json
import os
import random
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO / "ScanNetData_local" / "train"
EXISTING_SELFEVOLVE = REPO / "easy_r1" / "dataset" / "selfevolve_train.jsonl"
SCHEDULE_OUT = REPO / "easy_r1" / "dataset" / "full_selfevolve_schedule.json"
DATA_OUT = REPO / "easy_r1" / "dataset" / "full_selfevolve_train.jsonl"

MAZE_TASK = "maze_navigation"
VSI_TYPES = [
    "single_image_relative_direction",
    "distance_cam_obj",
    "depth_order_obj_obj",
]
SPAR_TYPES = [
    "spar_depth_prediction_oc",
    "spar_depth_prediction_oo",
    "spar_distance_prediction_oc",
    "spar_distance_prediction_oo",
    "spar_distance_infer_center_oo",
    "spar_obj_spatial_relation_oo",
]


def build_schedule(steps: int, seed: int = 42) -> list[str]:
    per_family = steps // 3
    remainder = steps - per_family * 3
    schedule: list[str] = []
    families = [("maze", [MAZE_TASK]), ("vsi", VSI_TYPES), ("spar", SPAR_TYPES)]
    for i, (_name, types) in enumerate(families):
        n = per_family + (1 if i < remainder else 0)
        for j in range(n):
            schedule.append(types[j % len(types)])
    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


def load_maze_contexts() -> list[dict]:
    entries = []
    if EXISTING_SELFEVOLVE.exists():
        with open(EXISTING_SELFEVOLVE) as f:
            for line in f:
                row = json.loads(line)
                imgs = row.get("images", [])
                if imgs and "maze" in str(imgs[0]):
                    entries.append(row)
    return entries


def load_scannet_contexts() -> list[dict]:
    entries = []
    for scene_dir in sorted(TRAIN_ROOT.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene"):
            continue
        fp_dir = scene_dir / "frame_processed"
        if not fp_dir.is_dir():
            continue
        frames = sorted(f for f in os.listdir(fp_dir) if f.endswith(".jpg"))
        if not frames:
            continue
        mid = len(frames) // 2
        img_path = str(fp_dir / frames[mid])
        entries.append({"problem": None, "answer": None, "images": [img_path]})
    return entries


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    schedule = build_schedule(args.steps, args.seed)
    SCHEDULE_OUT.write_text(json.dumps(schedule, indent=2))
    dist = Counter()
    for t in schedule:
        if t == MAZE_TASK:
            dist["maze"] += 1
        elif t.startswith("spar_"):
            dist["spar"] += 1
        else:
            dist["vsi"] += 1
    print(f"Schedule: {len(schedule)} steps  maze={dist['maze']} vsi={dist['vsi']} spar={dist['spar']}")
    print(f"  written to {SCHEDULE_OUT}")

    maze_ctx = load_maze_contexts()
    scannet_ctx = load_scannet_contexts()
    print(f"maze contexts: {len(maze_ctx)}  scannet contexts: {len(scannet_ctx)}")
    if not maze_ctx or not scannet_ctx:
        raise RuntimeError("Empty context pool; check data files.")

    # Round-robin index per modality.
    maze_idx = 0
    scannet_idx = 0
    ordered: list[dict] = []
    for task_type in schedule:
        if task_type == MAZE_TASK:
            pool = maze_ctx
            counter_ref = "maze"
        else:
            pool = scannet_ctx
            counter_ref = "scannet"
        for _ in range(args.batch_size):
            if counter_ref == "maze":
                ordered.append(maze_ctx[maze_idx % len(maze_ctx)])
                maze_idx += 1
            else:
                ordered.append(scannet_ctx[scannet_idx % len(scannet_ctx)])
                scannet_idx += 1

    with open(DATA_OUT, "w") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Ordered dataset: {len(ordered)} entries  written to {DATA_OUT}")
    print("Set `data.shuffle: false` in the config to use the ordered alignment.")


if __name__ == "__main__":
    main()
