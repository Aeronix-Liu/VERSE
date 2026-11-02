#!/usr/bin/env python3
"""Build v4 training data (2026-08-11): no-SPAR run, scene = 4 frames, maze only.

Motivation: the best self-evolve checkpoint so far (vilasr_selfevolve_4gpu_step200)
was trained WITHOUT any SPAR tasks; adding SPAR in imitate-0808 did not improve
benchmark results. This build ships a clean data pool that matches the new
no-SPAR schedule (VSI-Bench scene tasks + MAZE only) with 4-frame scenes
(rolled back from 8 to keep rollout latency down; user request).

Composition:
  - scene:  332 scenes x 3 distinct 4-frame subsets (coverage-greedy)  ~= 990 rows
  - maze:   full 750-maze pool                                          750 rows
  - image_pair and single_image rows: DROPPED (no schedule entries use them)

Output: easy_r1/dataset/full_selfevolve_train_v4.jsonl
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
EASY_R1 = os.path.join(REPO, "easy_r1")
SHARED_REPO = "/mnt/shared/MA4Tool/all_users_shared/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve"
SCANNET_TRAIN = os.path.join(REPO, "ScanNetData_local", "train")
SCANNET_META = os.path.join(REPO, "ScanNetData_local", "metadata")
MAZE_POOL = os.path.join(REPO, "dataset", "maze_pool")
OUT_PATH = os.path.join(EASY_R1, "dataset", "full_selfevolve_train_v4.jsonl")

N_SCENE_FRAMES = 4      # roll back from 8 -> 4 (user request, 2026-08-11)
N_SCENE_SUBSETS = 3     # distinct 4-frame subsets per scene


def usable_frames(scene_id):
    frame_dir = os.path.join(SCANNET_TRAIN, scene_id, "frame_processed")
    meta_dir = os.path.join(SCANNET_META, scene_id, "frame_processed")
    if not os.path.isdir(frame_dir) or not os.path.isdir(meta_dir):
        return []
    metas = {f[:-5] for f in os.listdir(meta_dir) if f.endswith(".json")}
    out = []
    for f in sorted(os.listdir(frame_dir)):
        if f.endswith(".jpg") and f[:-4] in metas:
            out.append(f"{SHARED_REPO}/ScanNetData_local/train/{scene_id}/frame_processed/{f}")
    return out


def frame_visible_objects(scene_id, frame_path):
    frame_name = os.path.basename(frame_path)[:-4]
    meta_path = os.path.join(SCANNET_META, scene_id, "frame_processed", frame_name + ".json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    ids = set()
    for obj in meta.get("objects", []) or []:
        oid = obj.get("object_id")
        if oid is not None:
            ids.add(oid)
    return ids


def coverage_greedy_subset(frames, visible_sets, k, first_idx):
    """Pick k frames greedily maximizing distinct visible-object coverage.

    The first frame is forced to `first_idx` so different starts yield different
    subsets. Ties broken by preferring temporally-spread viewpoints.
    """
    n = len(frames)
    if n <= k:
        return list(range(n))
    picked = [first_idx]
    covered = set(visible_sets[first_idx])
    while len(picked) < k:
        best_i, best_gain, best_spread = None, -1, -1
        for i in range(n):
            if i in picked:
                continue
            gain = len(visible_sets[i] - covered)
            spread = min(abs(i - p) for p in picked)
            if gain > best_gain or (gain == best_gain and spread > best_spread):
                best_i, best_gain, best_spread = i, gain, spread
        picked.append(best_i)
        covered |= visible_sets[best_i]
    return sorted(picked)


def main():
    scenes = sorted(
        s for s in os.listdir(SCANNET_TRAIN)
        if os.path.isdir(os.path.join(SCANNET_TRAIN, s, "frame_processed"))
    )
    print(f"[setup] {len(scenes)} ScanNet scenes")

    scene_rows = []
    for sid in scenes:
        frames = usable_frames(sid)
        n = len(frames)
        if n < N_SCENE_FRAMES:
            # keep every scene row at exactly N_SCENE_FRAMES so the modality-
            # aware dataloader classifies all rows as `scene` (len(images) >= 3).
            continue

        visible_sets = [frame_visible_objects(sid, fp) for fp in frames]
        start_order = sorted(range(n), key=lambda i: -len(visible_sets[i]))
        seen_subsets = set()
        for k in range(min(N_SCENE_SUBSETS, n)):
            idx = tuple(coverage_greedy_subset(frames, visible_sets, N_SCENE_FRAMES, start_order[k]))
            if idx in seen_subsets:
                continue
            seen_subsets.add(idx)
            scene_rows.append({"problem": None, "answer": None,
                               "images": [frames[i] for i in idx]})

    maze_rows = []
    for f in sorted(os.listdir(MAZE_POOL)):
        if f.endswith(".png"):
            maze_rows.append({"problem": None, "answer": None,
                              "images": [f"{SHARED_REPO}/dataset/maze_pool/{f}"]})

    all_rows = scene_rows + maze_rows
    counts = Counter()
    for r in all_rows:
        imgs = r["images"]
        if "maze" in imgs[0].lower():
            counts["maze"] += 1
        elif len(imgs) >= 3:
            counts["scene"] += 1
        elif len(imgs) == 2:
            counts["image_pair"] += 1
        else:
            counts["single_image"] += 1
    print(f"[total] {len(all_rows)} rows -> {len(all_rows) // 16} steps/epoch @ batch 16")
    for m, c in sorted(counts.items()):
        print(f"  {m}: {c}")

    missing = 0
    for r in all_rows:
        for p in r["images"]:
            if not os.path.exists(p):
                missing += 1
    if missing:
        raise SystemExit(f"{missing} referenced images missing!")

    with open(OUT_PATH, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
