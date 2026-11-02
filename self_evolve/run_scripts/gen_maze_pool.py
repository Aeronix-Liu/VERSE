#!/usr/bin/env python3
"""Generate the synthetic maze pool (images + deterministic metadata + contexts jsonl).

Usage:
    python run_scripts/gen_maze_pool.py [--num-per-grid 100] [--grid-sizes 4,5,6]

Outputs:
    dataset/maze_pool/maze_gN_XXXX.png / .json
    dataset/maze_contexts.jsonl   ({"problem": null, "answer": null, "images": [abs png]})
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.maze_engine import MazeEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-per-grid", type=int, default=100)
    parser.add_argument("--grid-sizes", default="4,5,6")
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    pool_dir = REPO / "dataset" / "maze_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    contexts_path = REPO / "dataset" / "maze_contexts.jsonl"

    rng = random.Random(args.seed)
    records = []
    for grid_size in [int(g) for g in args.grid_sizes.split(",")]:
        for i in range(args.num_per_grid):
            maze_id = f"maze_g{grid_size}_{i:04d}"
            image, meta = MazeEngine.generate(maze_id, grid_size, rng)
            image_path = pool_dir / f"{maze_id}.png"
            meta_path = pool_dir / f"{maze_id}.json"
            image.save(image_path)
            meta["image"] = image_path.name
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            records.append({"problem": None, "answer": None, "images": [str(image_path)]})

    with contexts_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated {len(records)} mazes -> {pool_dir}")
    print(f"Contexts jsonl -> {contexts_path}")

    # Self-check: every maze's shortest path to each destination must validate.
    ok = bad = 0
    for record in records[:50]:
        image_path = record["images"][0]
        meta = MazeEngine.load_metadata_for_image(image_path)
        for letter, path in meta["shortest_paths"].items():
            if not path:
                continue
            question = (
                "Task: Maze Navigation Simulation\n"
                "Determine the final destination (A, B, C or D) from the starting point (green point) "
                "following the action sequence.\n"
                "Full Action Sequence: " + " ".join(f"Go {m}." for m in path)
            )
            res = MazeEngine.validate_and_answer(question_text=question, image_path=image_path)
            if res["is_valid"] and res["answer"] == letter:
                ok += 1
            else:
                bad += 1
                print(f"SELF-CHECK FAIL {meta['maze_id']} {letter}: {res}")
    print(f"Self-check: {ok} ok, {bad} bad")


if __name__ == "__main__":
    main()
