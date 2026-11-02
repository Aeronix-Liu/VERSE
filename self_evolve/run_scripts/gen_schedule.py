#!/usr/bin/env python3
"""Generate a deterministic round-robin task type schedule for self-evolve.

Produces a JSON list of task type strings (one per step) where:
  - maze / VSI / SPAR families have roughly equal total weight
  - every step gets exactly one type (all contexts in that step share it)
  - the schedule is deterministic (no adaptive collapse)

Usage:
    python gen_schedule.py --steps 200 --out schedule.json
"""
import argparse
import json
import random
from pathlib import Path

MAZE_TYPES = ["maze"]

VSI_TYPES = [
    "object_counting",
    "relative_distance",
    "relative_direction_hard",
    "absolute_distance",
    "object_size",
    "room_size",
    "route_planning",
]

SPAR_TYPES = [
    "spar_depth_prediction_oc",
    "spar_depth_prediction_oo",
    "spar_depth_prediction_oc_mv",
    "spar_depth_prediction_oo_mv",
    "spar_distance_prediction_oc",
    "spar_distance_prediction_oo",
    "spar_distance_prediction_oc_mv",
    "spar_distance_prediction_oo_mv",
    "spar_distance_infer_center_oo",
    "spar_distance_infer_center_oo_mv",
    "spar_obj_spatial_relation_oo",
    "spar_obj_spatial_relation_oc_mv",
    "spar_obj_spatial_relation_oo_mv",
    "spar_position_matching",
    "spar_view_change_infer",
    "spar_camera_motion_infer",
]


def build_schedule(steps: int, seed: int = 42) -> list[str]:
    """Build a schedule with ~equal family weight, each type proportional."""
    families = {
        "maze": MAZE_TYPES,
        "vsi": VSI_TYPES,
        "spar": SPAR_TYPES,
    }
    # Each family gets steps/3 slots; within a family, types are round-robin.
    per_family = steps // len(families)
    remainder = steps - per_family * len(families)

    schedule = []
    for i, (fname, types) in enumerate(families.items()):
        n = per_family + (1 if i < remainder else 0)
        for j in range(n):
            schedule.append(types[j % len(types)])

    rng = random.Random(seed)
    rng.shuffle(schedule)
    return schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", default="schedule.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    schedule = build_schedule(args.steps, args.seed)

    # Print distribution
    from collections import Counter
    dist = Counter()
    for t in schedule:
        if t.startswith("spar_"):
            dist["spar"] += 1
        elif t == "maze":
            dist["maze"] += 1
        else:
            dist["vsi"] += 1
    print(f"Schedule: {len(schedule)} steps")
    print(f"  maze: {dist['maze']}, vsi: {dist['vsi']}, spar: {dist['spar']}")
    print(f"  type counts: {dict(Counter(schedule))}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(schedule, indent=2))
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
