#!/usr/bin/env python3
"""CPU test for export_traj_bank_sft.py: synthesize one maze bank row in the
exact rollout serialization format, export it, and verify the sharegpt output
(turn structure, <image> marker count, replayed tool image).

Run:
  cd ViLaSR-self-evolve && ../envs/Xinran_VERSE/bin/python \
      run_scripts/test_traj_bank_export.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EASY_R1 = REPO_ROOT / "easy_r1"
sys.path.insert(0, str(EASY_R1))
sys.path.insert(0, str(REPO_ROOT))

from src.maze_engine import (  # noqa: E402
    MARGIN_BOTTOM_FRAC,
    MARGIN_LEFT_FRAC,
    MARGIN_RIGHT_FRAC,
    MARGIN_TOP_FRAC,
    MAZE_IMAGE_SIZE,
)
from verl.utils.agentic import (  # noqa: E402
    AGENTIC_INTERMEDIATE_PROMPT,
    AGENTIC_INTERMEDIATE_TEMPLATE,
    VILASR_SYSTEM_PROMPT,
    process_image_agentic,
)

MIN_PIXELS, MAX_PIXELS = 100352, 200704
PAD = "<|vision_start|><|image_pad|><|vision_end|>"

maze_meta = json.load(open(REPO_ROOT / "dataset" / "maze_pool" / "maze_g5_0003.json"))
maze_png = str(REPO_ROOT / "dataset" / "maze_pool" / "maze_g5_0003.png")
letter = "C"
moves = maze_meta["shortest_paths"][letter]
grid = maze_meta["grid_size"]

proc = process_image_agentic(maze_png, MIN_PIXELS, MAX_PIXELS)
W, H = proc.size
print(f"processed maze image: {W}x{H}")


def cell_center(row: int, col: int) -> tuple[int, int]:
    ml, mt = MARGIN_LEFT_FRAC, MARGIN_TOP_FRAC
    uw = 1 - MARGIN_LEFT_FRAC - MARGIN_RIGHT_FRAC
    uh = 1 - MARGIN_TOP_FRAC - MARGIN_BOTTOM_FRAC
    return (round((ml + (col + 0.5) * uw / grid) * W), round((mt + (row + 0.5) * uh / grid) * H))


deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
r, c = maze_meta["start"]
cells = [(r, c)]
for mv in moves:
    dr, dc = deltas[mv]
    r, c = r + dr, c + dc
    cells.append((r, c))
segments = [
    {
        "index": 1,
        "start_point_2d": list(cell_center(*cells[i])),
        "end_point_2d": list(cell_center(*cells[i + 1])),
        "label": f"Step {i + 1}: Go {moves[i]}.",
    }
    for i in range(len(moves))
]

question = (
    "Task: Maze Navigation Simulation\n"
    "Determine the final destination (A, B, C or D) from the starting point (green point) "
    "following the action sequence.\n"
    "The definitions of the actions are as below.\n"
    "* Go up/left/down/right: move one grid space in the absolute up/left/down/right direction.\n"
    f"Full Action Sequence: {' '.join(f'Go {m}.' for m in moves)}"
)
asst1 = (
    "<think> I trace the sequence one cell per move. </think> Action: Path Tracer\n"
    "```json\n" + json.dumps(segments, indent=4) + "\n```"
)
asst2 = f"<think> The path ends on the cell labeled {letter}. </think> Action: Answer\n<answer> {letter} </answer>"
mid_template = AGENTIC_INTERMEDIATE_TEMPLATE.format(
    pad=PAD, prompt=AGENTIC_INTERMEDIATE_PROMPT.format(current_image_idx=2, width=W, height=H)
)
sequence_text = (
    "<|im_start|>system\n" + VILASR_SYSTEM_PROMPT.format(width=W, height=H) + "<|im_end|>\n"
    "<|im_start|>user\n" + PAD + "\n### Question:\n" + question +
    "\nThe index of the given image is 1.\nAnswer with the option's letter from the given choices directly."
    "\n\nBegin your reasoning. After each tool use, critically evaluate the visualization and adjust if needed:\n<|im_end|>\n"
    "<|im_start|>assistant\n" + asst1 + mid_template + asst2 + "<|im_end|>"
)

bank_row = {
    "step": 1,
    "task_type": "maze_navigation",
    "question": question,
    "gt": letter,
    "gt_letter": "",
    "mc_options_json": "",
    "answer_style": "multiple choice",
    "scene_id": maze_meta["maze_id"],
    "images": [maze_png],
    "accuracy": 1.0,
    "format": 1.0,
    "maze_traj_fidelity": 1.0,
    "question_source": "proposer",
    "sequence_text": sequence_text,
    "response_text": asst1 + "\n" + asst2,
}

with tempfile.TemporaryDirectory() as tmp:
    bank_path = Path(tmp) / "trajectory_bank.jsonl"
    bank_path.write_text(json.dumps(bank_row) + "\n")
    out_dir = Path(tmp) / "export"
    env = dict(os.environ)
    rc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_scripts" / "export_traj_bank_sft.py"),
            "--bank", str(bank_path),
            "--out-dir", str(out_dir),
            "--dataset-name", "traj_bank_test",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    print(rc.stdout[-2000:])
    if rc.returncode != 0:
        print(rc.stderr[-3000:])
        sys.exit(1)
    rows = json.loads((out_dir / "traj_bank_test.json").read_text())
    assert len(rows) == 1, f"expected 1 exported row, got {len(rows)}"
    row = rows[0]
    roles = [t["role"] for t in row["conversations"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"], roles
    n_markers = sum(t["content"].count("<image>") for t in row["conversations"])
    assert n_markers == 2 == len(row["images"]), (n_markers, row["images"])
    for p in row["images"]:
        full = p if os.path.isabs(p) else str(out_dir / Path(p).relative_to(Path(p).parts[0]))
        # exported paths are absolute in this test (out_dir not under ViLaSR-data)
        assert os.path.exists(p if os.path.isabs(p) else full), p
    assert row["answer"] == letter
    assert "<answer> C </answer>" in row["conversations"][-1]["content"]
    # copy the replayed tool image out for visual inspection
    import shutil
    shutil.copyfile(row["images"][1], "/tmp/traj_bank_replayed_tool_image.png")
    print("replayed tool image -> /tmp/traj_bank_replayed_tool_image.png")

print("ALL EXPORT TESTS PASSED")
