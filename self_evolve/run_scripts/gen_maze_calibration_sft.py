#!/usr/bin/env python3
"""Generate calibrated maze Path-Tracer SFT data ("drawing repair" data).

Motivation (2026-08-13 demo + memory): the model draws ~0.69-cell steps and its
Path-Tracer trajectory does not land on the destination cell even when the
final letter is right — the drawing is decorative, not a faithful simulation.
One root cause is that the original ViLaSR cold-start/RRS maze data itself has
imprecise coordinates (e.g. 3 moves drawn as 240px on a 588px image ≈ 1.6
cells). This script generates SFT trajectories whose Path-Tracer coordinates
are EXACT cell-center jumps computed from the maze metadata, teaching the model
the "one move = one cell" calibration.

Trajectory design (2 assistant turns, mirroring the RRS sharegpt format):
  system  : ViLaSR system prompt (width/height = 616)
  user    : maze question (same text the RL Round-2 solver sees)
  asst 1  : <think> grid geometry + plan </think> Action: Path Tracer
            [one segment per move, chained exact cell centers, unique labels]
  user    : tool feedback image (all segments drawn with the SAME renderer the
            RL rollout uses: verl/utils/edit_image_video.plot_movement)
  asst 2  : <think> verify endpoint cell + letter </think> Action: Answer
            <answer> L </answer>

Every generated row is self-checked with maze_trajectory_fidelity == 1.0.

Output: sharegpt JSON + images under ViLaSR/ViLaSR-data/maze_calibration/,
and a "maze_calibration" entry registered in dataset_info.json.

Usage:
  cd ViLaSR-self-evolve && ../envs/Xinran_VERSE/bin/python \
      run_scripts/gen_maze_calibration_sft.py [--max-mazes N] [--letters ABCD]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EASY_R1 = REPO_ROOT / "easy_r1"
XINRAN_ROOT = REPO_ROOT.parent
VILASR_DATA = XINRAN_ROOT / "ViLaSR" / "ViLaSR-data"
sys.path.insert(0, str(EASY_R1))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EASY_R1 / "training" / "reward_function"))

from src.maze_engine import (  # noqa: E402
    MARGIN_BOTTOM_FRAC,
    MARGIN_LEFT_FRAC,
    MARGIN_RIGHT_FRAC,
    MARGIN_TOP_FRAC,
    MAZE_IMAGE_SIZE,
    MazeEngine,
)
from verl.utils.agentic import VILASR_SYSTEM_PROMPT  # noqa: E402
from verl.utils.edit_image_video import plot_movement  # noqa: E402
from agentic_solver_reward import maze_trajectory_fidelity  # noqa: E402

MOVE_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

THINK_PLAN_TEMPLATES = [
    (
        "The maze is a {g}x{g} grid. The drawable grid area spans from about "
        "({x_min}, {y_min}) to ({x_max}, {y_max}), so one cell is about {cw} px wide "
        "and {ch} px tall. The start (green dot) is at row {r0}, column {c0}, whose "
        "center is ({x0}, {y0}). I will trace the action sequence exactly one cell per "
        "move, jumping from cell center to cell center: each 'Go up/down' changes y by "
        "about {ch} px and each 'Go left/right' changes x by about {cw} px. I will draw "
        "every move as its own segment so I can verify the path cell by cell."
    ),
    (
        "This is a {g}x{g} maze, so each grid cell is roughly {cw} px by {ch} px inside "
        "the border. The green start dot sits at row {r0}, column {c0} with pixel center "
        "({x0}, {y0}). One action moves exactly one cell, i.e. about {cw} px horizontally "
        "or {ch} px vertically. I will plot the full action sequence as chained segments "
        "between adjacent cell centers and then check where the path ends."
    ),
    (
        "First I calibrate the grid: {g} columns and {g} rows over the usable area gives a "
        "cell size of about {cw}x{ch} px. The start is the green dot at row {r0}, column "
        "{c0} (center ({x0}, {y0})). Every 'Go' action must move exactly one full cell, so "
        "each segment I draw spans one cell width or height, landing precisely on the next "
        "cell center. Let me trace the whole sequence."
    ),
]

THINK_VERIFY_TEMPLATES = [
    (
        "The traced path is drawn. Checking segment by segment: every segment spans exactly "
        "one cell, and the final endpoint is ({xt}, {yt}), the center of row {rt}, column "
        "{ct}. That cell contains the letter {letter}, so the action sequence ends at {letter}."
    ),
    (
        "Each drawn segment moves exactly one cell as required. The path terminates at pixel "
        "({xt}, {yt}), which is the center of the cell at row {rt}, column {ct} — the cell "
        "labeled {letter}. The visualization matches the action sequence, so the answer is {letter}."
    ),
    (
        "The visualization confirms the calibration: the endpoint lands on the cell center "
        "({xt}, {yt}) at row {rt}, column {ct}, where the red letter {letter} is drawn. No "
        "adjustment is needed; the destination is {letter}."
    ),
]


def cell_center(row: int, col: int, grid: int) -> tuple[int, int]:
    size = MAZE_IMAGE_SIZE
    ml = size * MARGIN_LEFT_FRAC
    mt = size * MARGIN_TOP_FRAC
    usable_w = size * (1 - MARGIN_LEFT_FRAC - MARGIN_RIGHT_FRAC)
    usable_h = size * (1 - MARGIN_TOP_FRAC - MARGIN_BOTTOM_FRAC)
    return (round(ml + (col + 0.5) * usable_w / grid), round(mt + (row + 0.5) * usable_h / grid))


def build_row(maze_meta: dict, image_path: Path, letter: str, out_dir: Path, rng: random.Random) -> dict | None:
    built = MazeEngine.build_question_from_letter(letter, str(image_path))
    if built is None:
        return None
    grid = int(maze_meta["grid_size"])
    moves = built["path"]
    size = MAZE_IMAGE_SIZE
    usable_w = size * (1 - MARGIN_LEFT_FRAC - MARGIN_RIGHT_FRAC)
    usable_h = size * (1 - MARGIN_TOP_FRAC - MARGIN_BOTTOM_FRAC)
    cw, ch = round(usable_w / grid), round(usable_h / grid)

    r, c = maze_meta["start"]
    cells = [(r, c)]
    for mv in moves:
        dr, dc = MOVE_DELTAS[mv]
        r, c = r + dr, c + dc
        cells.append((r, c))

    segments = []
    for k in range(len(moves)):
        sx, sy = cell_center(*cells[k], grid)
        ex, ey = cell_center(*cells[k + 1], grid)
        segments.append(
            {
                "index": 1,
                "start_point_2d": [sx, sy],
                "end_point_2d": [ex, ey],
                "label": f"Step {k + 1}: Go {moves[k]}.",
            }
        )

    # Self-check with the exact reward-side fidelity scorer.
    ref = {
        "grid_size": grid,
        "start": list(maze_meta["start"]),
        "destinations": maze_meta["destinations"],
        "moves": moves,
        "gt_letter": letter,
        "image_width": size,
        "image_height": size,
        "margins": [MARGIN_LEFT_FRAC, MARGIN_RIGHT_FRAC, MARGIN_TOP_FRAC, MARGIN_BOTTOM_FRAC],
    }
    fid = maze_trajectory_fidelity(segments, ref)
    if fid["fidelity"] < 1.0 - 1e-9:
        print(f"  [skip] {maze_meta['maze_id']}/{letter}: self-check fidelity={fid['fidelity']:.3f}")
        return None

    sample_dir = out_dir / "images" / f"{maze_meta['maze_id']}_{letter}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    img0_path = sample_dir / "image_0.png"
    shutil.copyfile(image_path, img0_path)
    # Render the tool feedback with the SAME renderer the RL rollout uses.
    img1 = Image.open(image_path).convert("RGB")
    plot_movement(img1, segments, size, size)
    img1_path = sample_dir / "image_1.png"
    img1.save(img1_path)

    x0, y0 = cell_center(*cells[0], grid)
    xt, yt = cell_center(*cells[-1], grid)
    x_min, y_min = round(size * MARGIN_LEFT_FRAC), round(size * MARGIN_TOP_FRAC)
    x_max, y_max = round(size * (1 - MARGIN_RIGHT_FRAC)), round(size * (1 - MARGIN_BOTTOM_FRAC))

    think_plan = rng.choice(THINK_PLAN_TEMPLATES).format(
        g=grid, cw=cw, ch=ch, r0=cells[0][0], c0=cells[0][1], x0=x0, y0=y0,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
    )
    think_verify = rng.choice(THINK_VERIFY_TEMPLATES).format(
        xt=xt, yt=yt, rt=cells[-1][0], ct=cells[-1][1], letter=letter,
    )

    seg_json = json.dumps(segments, indent=4, ensure_ascii=False)
    question_text = built["question_text"]
    user_question = (
        "<image>### Question:\n"
        f"{question_text}\n"
        "The index of the given image is 1.\n"
        "Answer with the option's letter from the given choices directly.\n\n"
        "Begin your reasoning. After each tool use, critically evaluate the visualization and adjust if needed:\n"
    )
    conversations = [
        {"role": "system", "content": "system\n" + VILASR_SYSTEM_PROMPT.format(width=size, height=size)},
        {"role": "user", "content": user_question},
        {
            "role": "assistant",
            "content": f"<think> {think_plan}\n</think> Action: Path Tracer\n```json\n{seg_json}\n```",
        },
        {
            "role": "user",
            "content": (
                f"<image>The index of the given image is 2 (width: {size}, height: {size}). "
                "Continue your reasoning. After each tool use, critically evaluate the visualization and adjust if needed:\n"
            ),
        },
        {
            "role": "assistant",
            "content": f"<think> {think_verify}\n</think> Action: Answer\n<answer> {letter} </answer>",
        },
    ]
    try:
        rel = out_dir.relative_to(VILASR_DATA)
    except ValueError:
        rel = Path(out_dir.name)
    return {
        "source": "maze_calibration",
        "split": "train",
        "question_id": f"{maze_meta['maze_id']}_{letter}",
        "images": [
            f"./{rel}/images/{maze_meta['maze_id']}_{letter}/image_0.png",
            f"./{rel}/images/{maze_meta['maze_id']}_{letter}/image_1.png",
        ],
        "question": question_text,
        "answer": letter,
        "question_type": "multiple choice",
        "conversations": conversations,
        "score": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", default=str(REPO_ROOT / "dataset" / "maze_pool"))
    parser.add_argument("--out-dir", default=str(VILASR_DATA / "maze_calibration"))
    parser.add_argument("--letters", default="ABCD")
    parser.add_argument("--max-mazes", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--register", action="store_true", help="add maze_calibration to dataset_info.json")
    args = parser.parse_args()

    pool_dir = Path(args.pool_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    maze_jsons = sorted(pool_dir.glob("maze_*.json"))
    if args.max_mazes > 0:
        maze_jsons = maze_jsons[: args.max_mazes]

    rows = []
    for meta_path in maze_jsons:
        image_path = meta_path.with_suffix(".png")
        if not image_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        for letter in args.letters:
            if not (meta.get("shortest_paths") or {}).get(letter):
                continue
            row = build_row(meta, image_path, letter, out_dir, rng)
            if row is not None:
                rows.append(row)

    out_json = out_dir / "maze_calibration.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"Wrote {len(rows)} samples -> {out_json}")

    if args.register:
        info_path = VILASR_DATA / "dataset_info.json"
        info = json.loads(info_path.read_text())
        info["maze_calibration"] = {
            "file_name": "maze_calibration/maze_calibration.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
        print(f"Registered `maze_calibration` in {info_path}")


if __name__ == "__main__":
    main()
