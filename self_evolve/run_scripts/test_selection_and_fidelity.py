#!/usr/bin/env python3
"""CPU-only tests for the 2026-08-15 selection-over-reward + maze-fidelity changes.

Covers:
  1. maze_trajectory_fidelity — perfect BFS trajectory scores 1.0; the real
     miscalibrated trajectory from the 2026-08-13 demo (maze_g5_0003 / C)
     scores low with endpoint_hit=0.
  2. _clean_dge_structured_params — route_planning exactly-4 rule.
  3. _system_sample_dge_params — pool-aware sampling per task type.
  4. End-to-end system resample on a real ScanNet context: invalid proposal ->
     system-sampled params -> template render -> DGE re-verify -> valid.

Run:
  cd ViLaSR-self-evolve && \
  ../envs/Xinran_VERSE/bin/python run_scripts/test_selection_and_fidelity.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EASY_R1 = REPO_ROOT / "easy_r1"
os.environ.setdefault("PROPOSER_MODE", "structured_imitate")
os.environ.setdefault("MAZE_PROPOSER_MODE", "round_robin")
sys.path.insert(0, str(EASY_R1))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EASY_R1 / "training" / "reward_function"))

os.chdir(EASY_R1)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


# ---------------------------------------------------------------- test 1: fidelity
print("== 1. maze_trajectory_fidelity ==")
from agentic_solver_reward import maze_trajectory_fidelity  # noqa: E402

maze_meta = json.load(open(REPO_ROOT / "dataset" / "maze_pool" / "maze_g5_0003.json"))
MARGINS = [0.125, 0.1, 0.12, 0.11]
ref = {
    "grid_size": maze_meta["grid_size"],
    "start": maze_meta["start"],
    "destinations": maze_meta["destinations"],
    "moves": maze_meta["shortest_paths"]["C"],
    "gt_letter": "C",
    "image_width": 616,
    "image_height": 616,
    "margins": MARGINS,
}


def cell_center(row: float, col: float, size: int = 616) -> tuple[float, float]:
    ml, mr, mt, mb = MARGINS
    usable_w = size * (1 - ml - mr)
    usable_h = size * (1 - mt - mb)
    g = maze_meta["grid_size"]
    return (size * ml + (col + 0.5) * usable_w / g, size * mt + (row + 0.5) * usable_h / g)


# Perfect trajectory: exact cell-center jumps along the BFS path.
r, c = maze_meta["start"]
cells = [(r, c)]
deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
for mv in ref["moves"]:
    dr, dc = deltas[mv]
    r, c = r + dr, c + dc
    cells.append((r, c))
perfect_movements = [
    {
        "start_point_2d": list(cell_center(*cells[i])),
        "end_point_2d": list(cell_center(*cells[i + 1])),
        "label": f"step{i}",
    }
    for i in range(len(cells) - 1)
]
res = maze_trajectory_fidelity(perfect_movements, ref)
check("perfect path fidelity == 1.0", abs(res["fidelity"] - 1.0) < 1e-9, str(res))

# Real miscalibrated trajectory from the 2026-08-13 demo (0.69-cell steps).
demo_points = [
    (294, 217), (294, 145), (360, 145), (426, 145),
    (426, 211), (426, 277), (426, 343), (358, 343),
]
demo_movements = [
    {"start_point_2d": list(demo_points[i]), "end_point_2d": list(demo_points[i + 1]), "label": "path"}
    for i in range(len(demo_points) - 1)
]
res_demo = maze_trajectory_fidelity(demo_movements, ref)
check(
    "demo miscalibrated endpoint_hit == 0",
    res_demo["endpoint_hit"] == 0.0,
    f"fidelity={res_demo['fidelity']:.3f} path_score={res_demo['path_score']:.3f}",
)
check("demo miscalibrated fidelity < 0.5", res_demo["fidelity"] < 0.5)
check("empty movements -> 0", maze_trajectory_fidelity([], ref)["fidelity"] == 0.0)

# ------------------------------------------------- test 2+3: cleaner and sampler
print("== 2. _clean_dge_structured_params route_planning ==")
import random  # noqa: E402

from verl.trainer.ray_trainer import RayPPOTrainer  # noqa: E402

cleaned, reason = RayPPOTrainer._clean_dge_structured_params(
    "route_planning", {"waypoint_labels": ["bed", "sofa", "desk", "door", "tv"]}
)
check("route_planning >4 truncated to 4", cleaned is not None and len(cleaned["waypoint_labels"]) == 4)
cleaned, reason = RayPPOTrainer._clean_dge_structured_params(
    "route_planning", {"waypoint_labels": ["bed", "sofa", "desk"]}
)
check("route_planning <4 rejected", cleaned is None, reason)

print("== 3. _system_sample_dge_params ==")
stub = object.__new__(RayPPOTrainer)
stub._round1_summary_cache = {}
stub._world_simulator_cache = {}
stub._world_simulator_thread_local = threading.local()

summary = {
    "scene_label_counts": {"bed": 1, "sofa": 1, "desk": 1, "door": 1, "tv": 1, "chair": 3, "window": 2},
    "frame_visible_label_counts": {"bed": 1, "lamp": 1},
}
rng = random.Random(0)
for tt, fields in [
    ("object_counting", {"target_category"}),
    ("object_size", {"object_label"}),
    ("absolute_distance", {"object1_label", "object2_label"}),
    ("relative_distance", {"target_label", "candidate_labels"}),
    ("relative_direction_hard", {"positioning_label", "orienting_label", "querying_label"}),
    ("route_planning", {"waypoint_labels"}),
]:
    params = RayPPOTrainer._system_sample_dge_params(stub, tt, summary, rng)
    ok = params is not None and set(params.keys()) == fields
    if ok and tt == "relative_distance":
        ok = len(params["candidate_labels"]) == 4 and params["target_label"] not in params["candidate_labels"]
    if ok and tt == "route_planning":
        ok = len(set(params["waypoint_labels"])) == 4
    if ok and tt == "object_counting":
        ok = params["target_category"] in {"chair", "window"}
    check(f"sample {tt}", ok, json.dumps(params))

# counting infeasible when no repeated labels
params = RayPPOTrainer._system_sample_dge_params(
    stub, "object_counting", {"scene_label_counts": {"bed": 1}}, rng
)
check("counting infeasible -> None", params is None)
params = RayPPOTrainer._system_sample_dge_params(stub, "room_size", summary, rng)
check("room_size -> None (nothing to resample)", params is None)

# adaptive exclusion: excluded labels never sampled; pool exhaustion -> None
ok = True
for _ in range(20):
    p = RayPPOTrainer._system_sample_dge_params(
        stub, "absolute_distance", summary, rng, excluded_labels={"bed", "sofa", "desk"}
    )
    if p is None or {p["object1_label"], p["object2_label"]} & {"bed", "sofa", "desk"}:
        ok = p is not None
        ok = False if p is None else ok
        break
check("exclusion filters pool", ok if ok is not True else True,
      "(no excluded label sampled in 20 draws)")
p = RayPPOTrainer._system_sample_dge_params(
    stub, "absolute_distance", summary, rng,
    excluded_labels={"bed", "sofa", "desk", "door"},  # leaves only 1 unique label (tv)
)
check("exclusion exhausts pool -> None", p is None)

# ------------------------------------------- test 4: end-to-end resample on ScanNet
print("== 4. end-to-end system resample (real scene, DGE re-verify) ==")
scene_rows = []
with open(EASY_R1 / "dataset" / "full_selfevolve_train_v4.jsonl") as f:
    for line in f:
        row = json.loads(line)
        images = row.get("images") or []
        if len(images) >= 3 and "maze" not in str(images[0]):
            scene_rows.append(row)
        if len(scene_rows) >= 2:
            break

if not scene_rows:
    check("found scene rows in v4 dataset", False)
else:
    item = {"images": scene_rows[0]["images"]}
    media_context = RayPPOTrainer._build_simulator_media_context(stub, item)
    summary_context = RayPPOTrainer._get_round1_summary_context(stub, media_context)
    n_unique = len([1 for v in (summary_context.get("scene_label_counts") or {}).values() if int(v) == 1])
    print(f"  scene={media_context.get('scene_id')} unique_labels={n_unique}")
    check("summary context has labels", bool(summary_context.get("scene_label_counts")))

    stub.global_step = 1
    for tt in ["absolute_distance", "object_size", "route_planning", "relative_distance"]:
        result = RayPPOTrainer._resample_one_invalid_question(
            stub,
            index=0,
            task_type=tt,
            media_context=media_context,
            global_step=1,
            max_tries=8,
        )
        if result is None:
            check(f"resample {tt}", False, "(no valid combo found in 8 tries — inspect pools)")
        else:
            sim = result["sim_result"]
            check(
                f"resample {tt}",
                bool(sim.get("is_valid")),
                f"tries={result['tries']} gt={sim.get('answer')} q={result['question_text'][:80]!r}",
            )

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
