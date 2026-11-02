#!/usr/bin/env python3
"""Build v3 schedule (2026-08-11): 200 steps, VSI-Bench + MAZE only. No SPAR.

Motivation (user): the best self-evolve checkpoint so far
(vilasr_selfevolve_4gpu_step200) was trained WITHOUT any SPAR data, and adding
SPAR in imitate-0808 did not translate to benchmark gains. So we drop SPAR
entirely and focus training on the 7 VSI-Bench scene categories + MAZE.

VSI-Bench overall is the arithmetic mean over the (7 available) scene
categories, so we give each VSI task equal weight. MAZE gets ~25% share
(50/200), matching the v2 schedule ratio.

Distribution (200 entries):
  MAZE:                50    (25.0%)
  VSI object_counting: 22    (VSI overall = mean over 7)
  VSI absolute_distance:22
  VSI object_size:     21
  VSI room_size:       21
  VSI relative_distance:22
  VSI relative_direction_hard:22
  VSI route_planning:  20
                       ---
                      TOTAL 200

Output: easy_r1/dataset/full_selfevolve_schedule_v3.json
"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
EASY_R1 = os.path.join(HERE, "..", "easy_r1")
OUT_PATH = os.path.join(EASY_R1, "dataset", "full_selfevolve_schedule_v3.json")

random.seed(2026_0811)

vsi_tasks = {
    "object_counting": 22,
    "absolute_distance": 22,
    "object_size": 21,
    "room_size": 21,
    "relative_distance": 22,
    "relative_direction_hard": 22,
    "route_planning": 20,
}

maze_tasks = {"maze_navigation": 50}

schedule = []
for task, count in {**vsi_tasks, **maze_tasks}.items():
    schedule.extend([task] * count)

assert len(schedule) == 200, f"schedule should be 200 entries, got {len(schedule)}"
random.shuffle(schedule)

counts = Counter(schedule)
print(f"Schedule: {len(schedule)} entries")
for t, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {t}: {n}")

modality = Counter()
for t in schedule:
    modality["maze" if t == "maze_navigation" else "scene"] += 1
print("\nModality breakdown:")
for m, c in sorted(modality.items(), key=lambda x: -x[1]):
    print(f"  {m}: {c}  ({c/len(schedule):.1%})")

with open(OUT_PATH, "w") as f:
    json.dump(schedule, f, indent=2)
print(f"\n[done] wrote {OUT_PATH}")
