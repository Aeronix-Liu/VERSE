#!/usr/bin/env python3
"""Extend the maze pool WITHOUT touching existing mazes (2026-08-07).

The original pool (gen_maze_pool.py, seed 20260724) generated ids
maze_g{4,5,6}_{0000..0099}. Re-running that script with a larger count would
shift the rng stream and OVERWRITE existing g5/g6 images with different
content, corrupting rows in older training jsonls. This script instead
generates NEW mazes with a fresh seed and ids starting at 0100 per grid,
saving alongside the originals.

Usage:
    python run_scripts/extend_maze_pool.py [--num-per-grid 150] [--grid-sizes 4,5,6]
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
    parser.add_argument("--num-per-grid", type=int, default=150)
    parser.add_argument("--grid-sizes", default="4,5,6")
    parser.add_argument("--start-index", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    pool_dir = REPO / "dataset" / "maze_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    records = []
    for grid_size in [int(g) for g in args.grid_sizes.split(",")]:
        for i in range(args.start_index, args.start_index + args.num_per_grid):
            maze_id = f"maze_g{grid_size}_{i:04d}"
            image_path = pool_dir / f"{maze_id}.png"
            if image_path.exists():
                raise SystemExit(f"refusing to overwrite existing {image_path}")
            image, meta = MazeEngine.generate(maze_id, grid_size, rng)
            meta_path = pool_dir / f"{maze_id}.json"
            image.save(image_path)
            meta["image"] = image_path.name
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            records.append(str(image_path))

    print(f"Generated {len(records)} new mazes -> {pool_dir}")

    # Self-check: every new maze's verified shortest path must validate.
    ok = bad = 0
    for image_path in records:
        meta_path = Path(image_path).with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        for dest, info in meta.get("destinations", {}).items():
            built = MazeEngine.build_question_from_letter(dest, image_path)
            if built is None:
                bad += 1
                continue
            res = MazeEngine.validate_and_answer(
                question_text=built["question_text"], image_path=image_path
            )
            if res.get("is_valid") and res.get("answer") == dest:
                ok += 1
            else:
                bad += 1
    print(f"Self-check: ok={ok} bad={bad}")
    if bad:
        raise SystemExit("self-check failed")


if __name__ == "__main__":
    main()
