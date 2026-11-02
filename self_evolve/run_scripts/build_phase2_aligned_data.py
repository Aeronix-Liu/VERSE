#!/usr/bin/env python3
"""Build Phase-2 training data aligned with VSI-Bench, SPAR-Bench, and MAZE.

Creates three modality buckets from the 332 equipped ScanNet scenes:
  1. scene  (8 frames/scene)  → VSI-Bench scene-level tasks
  2. image_pair (2 frames)    → SPAR-Bench _mv tasks
  3. single_image (1 frame)  → SPAR-Bench single-image tasks (kept from existing)

Plus existing maze entries (unchanged).

Output: easy_r1/dataset/full_selfevolve_train_v2.jsonl
"""
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EASY_R1 = os.path.join(HERE, "..", "easy_r1")
SCANNET_ROOT = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/ScanNetData_local/train"
OLD_TRAIN = os.path.join(EASY_R1, "dataset", "full_selfevolve_train.jsonl")
OUT_PATH = os.path.join(EASY_R1, "dataset", "full_selfevolve_train_v2.jsonl")

N_SCENE_FRAMES = 4  # frames per scene entry (user: "不用16帧那么多"; 4 + tool-call images <= 32)

random.seed(42)


def get_scannet_scenes():
    """Return dict: scene_id -> sorted list of frame paths."""
    scenes = {}
    for scene_id in sorted(os.listdir(SCANNET_ROOT)):
        frame_dir = os.path.join(SCANNET_ROOT, scene_id, "frame_processed")
        if not os.path.isdir(frame_dir):
            continue
        frames = sorted(
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.endswith(".jpg")
        )
        if frames:
            scenes[scene_id] = frames
    return scenes


def sample_frames(frames, n, rng):
    """Uniformly sample n frames from a sorted list."""
    if len(frames) <= n:
        return list(frames)
    step = len(frames) / n
    indices = [int(i * step) for i in range(n)]
    return [frames[i] for i in indices]


def main():
    scenes = get_scannet_scenes()
    print(f"[setup] ScanNet scenes with frames: {len(scenes)}")

    # --- 1. Scene entries (8 frames each) ---
    scene_rows = []
    for scene_id, frames in sorted(scenes.items()):
        sampled = sample_frames(frames, N_SCENE_FRAMES, random)
        scene_rows.append({"problem": None, "answer": None, "images": sampled})
    print(f"[scene] {len(scene_rows)} scene entries ({N_SCENE_FRAMES} frames each)")

    # --- 2. Image-pair entries (2 frames each) ---
    pair_rows = []
    for scene_id, frames in sorted(scenes.items()):
        if len(frames) < 2:
            continue
        # Pick first and last third for maximum viewpoint diversity
        idx1 = 0
        idx2 = len(frames) // 2
        pair_rows.append({"problem": None, "answer": None, "images": [frames[idx1], frames[idx2]]})
    print(f"[pair]  {len(pair_rows)} image_pair entries (2 frames each)")

    # --- 3. Single-image entries (keep from existing data) ---
    old_rows = [json.loads(l) for l in open(OLD_TRAIN)]
    single_scannet = [r for r in old_rows if "scene" in str(r.get("images", [""])[0])]
    maze_rows = [r for r in old_rows if "maze" in str(r.get("images", [""])[0]).lower()]
    print(f"[keep]  {len(single_scannet)} single_image ScanNet + {len(maze_rows)} maze (from existing)")

    # --- 4. Combine ---
    all_rows = scene_rows + pair_rows + single_scannet + maze_rows
    print(f"\n[total] {len(all_rows)} rows")
    from collections import Counter
    modality_counts = Counter()
    for r in all_rows:
        imgs = r.get("images", [])
        n = len(imgs) if isinstance(imgs, list) else 1
        first = str(imgs[0]) if isinstance(imgs, list) and imgs else str(imgs)
        if "maze" in first.lower():
            modality_counts["maze"] += 1
        elif n >= 3:
            modality_counts["scene"] += 1
        elif n == 2:
            modality_counts["image_pair"] += 1
        else:
            modality_counts["single_image"] += 1
    for m, c in sorted(modality_counts.items()):
        print(f"  {m}: {c}")

    with open(OUT_PATH, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[done] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
