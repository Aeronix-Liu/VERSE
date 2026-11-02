#!/usr/bin/env python3
"""Build v3 training data: ~4800 rows so one epoch ≈ 300 steps @ batch 16 (2026-08-07).

Expansion strategy (user request: more 3D contexts, epoch 100 -> ~300 steps):
  - scene:        332 scenes x 3 distinct 4-frame subsets   (~990 rows, was 329)
  - image_pair:   332 scenes x 3 distinct frame pairs        (~990 rows, was 335)
  - single_image: up to 7 uniformly-spread frames per scene (~2050 rows, was 532)
  - maze:         full 750-maze pool (extend_maze_pool.py)   (750 rows, was 268)

All frames verified to have per-frame metadata JSON (5260/5260 do). ScanNet++
was ruled out earlier (no instance annotations -> DGE cannot verify);
downloading new ScanNet scenes is an overnight pipeline, left as follow-up.

Output: easy_r1/dataset/full_selfevolve_train_v3.jsonl
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
EASY_R1 = os.path.join(REPO, "easy_r1")
# Use the /mnt/shared alias prefix, matching v2 row paths.
SHARED_REPO = "/mnt/shared/MA4Tool/all_users_shared/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve"
SCANNET_TRAIN = os.path.join(REPO, "ScanNetData_local", "train")
SCANNET_META = os.path.join(REPO, "ScanNetData_local", "metadata")
MAZE_POOL = os.path.join(REPO, "dataset", "maze_pool")
OUT_PATH = os.path.join(EASY_R1, "dataset", "full_selfevolve_train_v3.jsonl")

N_SCENE_FRAMES = 8      # frames per scene entry (2026-08-07: 4->8 to narrow the
                        # train/eval gap; VSI-Bench eval shows 16 frames. 8 + ~7
                        # tool-drawn images stays well under limit_images=32)
N_SCENE_SUBSETS = 3     # distinct frame subsets per scene
N_PAIRS = 3             # image pairs per scene
N_SINGLE = 7            # max single-image frames per scene


def usable_frames(scene_id):
    """Sorted frame paths (shared prefix) that have per-frame metadata."""
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
    """Set of visible object ids from the per-frame metadata JSON."""
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
    """Pick k frame indices greedily maximizing covered object ids.

    The first frame is forced to `first_idx` so different starts give distinct
    subsets. Ties are broken by temporal distance from already-picked frames
    (prefer spread-out viewpoints). Returns temporally sorted indices.
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


def spread_indices(n, k, offset_frac=0.0):
    """k roughly-uniform indices into range(n), phase-shifted by offset_frac."""
    if n <= k:
        return list(range(n))
    step = n / k
    return sorted({min(n - 1, int((i + offset_frac) * step)) for i in range(k)})


def main():
    scenes = sorted(
        s for s in os.listdir(SCANNET_TRAIN)
        if os.path.isdir(os.path.join(SCANNET_TRAIN, s, "frame_processed"))
    )
    print(f"[setup] {len(scenes)} ScanNet scenes")

    scene_rows, pair_rows, single_rows = [], [], []
    for sid in scenes:
        frames = usable_frames(sid)
        n = len(frames)
        if n == 0:
            continue

        # --- scene entries: coverage-greedy 8-frame subsets, distinct starts ---
        visible_sets = [frame_visible_objects(sid, fp) for fp in frames]
        # Rank frames by individual visible-object count; use the top ranks as
        # forced distinct starting frames for the greedy subsets.
        start_order = sorted(range(n), key=lambda i: -len(visible_sets[i]))
        seen_subsets = set()
        for k in range(min(N_SCENE_SUBSETS, n)):
            idx = tuple(coverage_greedy_subset(frames, visible_sets, N_SCENE_FRAMES, start_order[k]))
            if idx in seen_subsets:
                continue
            seen_subsets.add(idx)
            scene_rows.append({"problem": None, "answer": None,
                               "images": [frames[i] for i in idx]})

        # --- image pairs: distinct (i, j) combos with wide viewpoint gaps ---
        if n >= 2:
            pair_cands = [(0, n // 2), (n // 4, min(n - 1, 3 * n // 4)), (n // 2, n - 1)]
            seen_pairs = set()
            for i, j in pair_cands:
                if i >= j or (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))
                pair_rows.append({"problem": None, "answer": None,
                                  "images": [frames[i], frames[j]]})
                if len(seen_pairs) >= N_PAIRS:
                    break

        # --- single images: uniform spread ---
        for i in spread_indices(n, N_SINGLE):
            single_rows.append({"problem": None, "answer": None, "images": [frames[i]]})

    # --- maze: full pool ---
    maze_rows = []
    for f in sorted(os.listdir(MAZE_POOL)):
        if f.endswith(".png"):
            maze_rows.append({"problem": None, "answer": None,
                              "images": [f"{SHARED_REPO}/dataset/maze_pool/{f}"]})

    all_rows = scene_rows + pair_rows + single_rows + maze_rows
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

    # sanity: every referenced image must exist
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
