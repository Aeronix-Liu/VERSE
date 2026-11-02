#!/usr/bin/env python3
"""Build the expanded self-evolve training data: maze + all scannet scenes with frames.

Output: easy_r1/dataset/full_selfevolve_train.jsonl
  - 300 maze contexts (from existing maze pool)
  - ~259 scannet single-frame contexts (1 per scene with frames)

Each entry: {"problem": null, "answer": null, "images": [path]}
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO / "ScanNetData_local" / "train"
EXISTING_SELFEVOLVE = REPO / "easy_r1" / "dataset" / "selfevolve_train.jsonl"
OUT = REPO / "easy_r1" / "dataset" / "full_selfevolve_train.jsonl"


def main():
    entries = []

    # 1. Copy maze contexts from existing selfevolve_train.jsonl
    if EXISTING_SELFEVOLVE.exists():
        with open(EXISTING_SELFEVOLVE) as f:
            for line in f:
                row = json.loads(line)
                imgs = row.get("images", [])
                if imgs and "maze" in str(imgs[0]):
                    entries.append(row)
    maze_count = len(entries)
    print(f"maze contexts: {maze_count}")

    # 2. Build scannet contexts: 1 per scene with frames (pick middle frame)
    scannet_count = 0
    for scene_dir in sorted(TRAIN_ROOT.iterdir()):
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scene"):
            continue
        fp_dir = scene_dir / "frame_processed"
        if not fp_dir.is_dir():
            continue
        frames = sorted(f for f in os.listdir(fp_dir) if f.endswith(".jpg"))
        if not frames:
            continue
        # Pick a frame from the middle (more interesting viewpoint than first)
        mid = len(frames) // 2
        img_path = str(fp_dir / frames[mid])
        entries.append({"problem": None, "answer": None, "images": [img_path]})
        scannet_count += 1

    print(f"scannet contexts: {scannet_count}")
    print(f"total: {len(entries)}")

    with open(OUT, "w") as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
