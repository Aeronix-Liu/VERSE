"""Deterministic Maze Engine — a zero-noise verifier for maze-navigation questions.

Analogous to the DGE for ScanNet scenes: the maze image pool is generated
programmatically together with full structural metadata (walls / start /
labeled destinations), so any proposed action sequence can be simulated
EXACTLY and the ground-truth destination letter computed with no model
involvement.

Round-1 (Proposer): given the maze image + textual maze context, writes a
ViLaSR-style maze-navigation question containing a "Full Action Sequence:
Go up. Go left. ..." block.
Verifier (this engine): parses the action sequence, simulates it from the
start cell, rejects boundary/wall violations, and returns the destination
letter (A-D) as ground truth.
Round-2 (Solver): answers from the image alone (agentic think->draw->observe).
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
from collections import deque
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

MAZE_POOL_DIRNAME = "maze_pool"
MAZE_IMAGE_SIZE = 616  # 28 * 22 — divisible by 28 (Qwen-VL patch grid, no resize).
# Margins mirror ViLaSR's matplotlib-rendered mazes (see calculate_grid_centers).
MARGIN_LEFT_FRAC = 0.125
MARGIN_RIGHT_FRAC = 0.1
MARGIN_TOP_FRAC = 0.12
MARGIN_BOTTOM_FRAC = 0.11

MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MOVE_PATTERN = re.compile(r"\bgo\s+(up|down|left|right)\b", re.IGNORECASE)
# Question text that reveals the intended destination letter spoils the Solver sample.
ANSWER_LEAK_PATTERN = re.compile(
    r"\b(?:ends?\s+(?:up\s+)?at|reach(?:es)?|arriv\w+\s+at|answer\s+is|destination\s+is|final\s+destination\s+is)"
    r"\s+(?:the\s+)?(?:destination\s+|point\s+|letter\s+)?[ABCD]\b",
    re.IGNORECASE,
)
MIN_MOVES = 2
MAX_MOVES = 25

_METADATA_CACHE: dict[str, dict[str, Any]] = {}
_METADATA_CACHE_LOCK = threading.Lock()


def _passage_key(cell_a: tuple[int, int], cell_b: tuple[int, int]) -> tuple:
    return tuple(sorted((tuple(cell_a), tuple(cell_b))))


class MazeEngine:
    # ------------------------------------------------------------- pool lookup

    @staticmethod
    def is_maze_image_path(path: Any) -> bool:
        return MAZE_POOL_DIRNAME in str(path or "")

    @staticmethod
    def metadata_path_for_image(image_path: str) -> str:
        base, _ = os.path.splitext(str(image_path))
        return base + ".json"

    @classmethod
    def load_metadata_for_image(cls, image_path: str) -> dict[str, Any]:
        meta_path = cls.metadata_path_for_image(image_path)
        with _METADATA_CACHE_LOCK:
            cached = _METADATA_CACHE.get(meta_path)
        if cached is not None:
            return cached
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with _METADATA_CACHE_LOCK:
            _METADATA_CACHE[meta_path] = meta
        return meta

    # -------------------------------------------------------------- generation

    @staticmethod
    def _generate_perfect_maze(grid_size: int, rng: random.Random) -> set[tuple]:
        """Recursive-backtracker perfect maze; returns the set of open passages."""
        visited = [[False] * grid_size for _ in range(grid_size)]
        passages: set[tuple] = set()
        stack = [(rng.randrange(grid_size), rng.randrange(grid_size))]
        visited[stack[0][0]][stack[0][1]] = True
        while stack:
            r, c = stack[-1]
            neighbors = []
            for dr, dc in MOVE_DELTAS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size and not visited[nr][nc]:
                    neighbors.append((nr, nc))
            if not neighbors:
                stack.pop()
                continue
            nr, nc = rng.choice(neighbors)
            visited[nr][nc] = True
            passages.add(_passage_key((r, c), (nr, nc)))
            stack.append((nr, nc))
        return passages

    @staticmethod
    def _open_extra_passages(grid_size: int, passages: set[tuple], rng: random.Random, frac: float = 0.15) -> None:
        """Knock down a few extra walls so mazes are less corridor-like (ViLaSR style)."""
        candidates = []
        for r in range(grid_size):
            for c in range(grid_size):
                for dr, dc in ((0, 1), (1, 0)):
                    nr, nc = r + dr, c + dc
                    if nr < grid_size and nc < grid_size:
                        key = _passage_key((r, c), (nr, nc))
                        if key not in passages:
                            candidates.append(key)
        rng.shuffle(candidates)
        for key in candidates[: max(1, int(len(candidates) * frac))]:
            passages.add(key)

    @staticmethod
    def _bfs_path(
        grid_size: int,
        passages: set[tuple],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> Optional[list[str]]:
        prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
        seen = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if cell == goal:
                moves: list[str] = []
                cur = cell
                while cur != start:
                    parent, move = prev[cur]
                    moves.append(move)
                    cur = parent
                return moves[::-1]
            r, c = cell
            for move, (dr, dc) in MOVE_DELTAS.items():
                nxt = (r + dr, c + dc)
                if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size):
                    continue
                if nxt in seen or _passage_key(cell, nxt) not in passages:
                    continue
                seen.add(nxt)
                prev[nxt] = (cell, move)
                queue.append(nxt)
        return None

    @classmethod
    def generate(cls, maze_id: str, grid_size: int, rng: random.Random) -> tuple[Image.Image, dict[str, Any]]:
        """Generate one maze image + full structural metadata."""
        passages = cls._generate_perfect_maze(grid_size, rng)
        cls._open_extra_passages(grid_size, passages, rng)

        cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        rng.shuffle(cells)
        start = cells[0]
        # Prefer destinations at distance >= 2 from the start and from each other.
        destinations: dict[str, tuple[int, int]] = {}
        letters = ["A", "B", "C", "D"]
        for cell in cells[1:]:
            if len(destinations) == 4:
                break
            if abs(cell[0] - start[0]) + abs(cell[1] - start[1]) < 2:
                continue
            if any(abs(cell[0] - d[0]) + abs(cell[1] - d[1]) < 2 for d in destinations.values()):
                continue
            destinations[letters[len(destinations)]] = cell
        for cell in cells[1:]:
            if len(destinations) == 4:
                break
            if cell == start or cell in destinations.values():
                continue
            destinations[letters[len(destinations)]] = cell

        shortest_paths = {}
        for letter, goal in destinations.items():
            path = cls._bfs_path(grid_size, passages, start, tuple(goal))
            shortest_paths[letter] = path or []

        meta = {
            "type": "maze",
            "maze_id": maze_id,
            "grid_size": grid_size,
            "start": list(start),
            "destinations": {k: list(v) for k, v in destinations.items()},
            "passages": [[list(a), list(b)] for a, b in sorted(passages)],
            "shortest_paths": shortest_paths,
            "suggested_min_moves": MIN_MOVES,
            "suggested_max_moves": min(MAX_MOVES, grid_size * grid_size),
        }
        image = cls.render(meta)
        return image, meta

    # --------------------------------------------------------------- rendering

    @staticmethod
    def _load_font(size: int) -> ImageFont.FreeTypeFont:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        try:
            import matplotlib.font_manager as fm

            return ImageFont.truetype(fm.findfont("DejaVu Sans", fontext="ttf"), size)
        except Exception:
            return ImageFont.load_default(size)

    @classmethod
    def render(cls, meta: dict[str, Any]) -> Image.Image:
        grid_size = int(meta["grid_size"])
        size = MAZE_IMAGE_SIZE
        margin_left = int(size * MARGIN_LEFT_FRAC)
        margin_right = int(size * MARGIN_RIGHT_FRAC)
        margin_top = int(size * MARGIN_TOP_FRAC)
        margin_bottom = int(size * MARGIN_BOTTOM_FRAC)
        usable_w = size - margin_left - margin_right
        usable_h = size - margin_top - margin_bottom
        cell_w = usable_w / grid_size
        cell_h = usable_h / grid_size

        passages = {_passage_key(tuple(a), tuple(b)) for a, b in meta["passages"]}

        img = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)

        def cell_x(col: float) -> float:
            return margin_left + col * cell_w

        def cell_y(row: float) -> float:
            return margin_top + row * cell_h

        # Light grid lines.
        for i in range(1, grid_size):
            draw.line([(cell_x(i), cell_y(0)), (cell_x(i), cell_y(grid_size))], fill=(210, 210, 210), width=2)
            draw.line([(cell_x(0), cell_y(i)), (cell_x(grid_size), cell_y(i))], fill=(210, 210, 210), width=2)

        wall_w = max(8, int(size * 0.022))
        # Border walls.
        draw.rectangle(
            [cell_x(0) - wall_w // 2, cell_y(0) - wall_w // 2, cell_x(grid_size) + wall_w // 2, cell_y(grid_size) + wall_w // 2],
            outline="black",
            width=wall_w,
        )
        # Interior walls: draw where a passage between neighbors is CLOSED.
        for r in range(grid_size):
            for c in range(grid_size):
                if c + 1 < grid_size and _passage_key((r, c), (r, c + 1)) not in passages:
                    x = cell_x(c + 1)
                    draw.line([(x, cell_y(r) - wall_w / 2), (x, cell_y(r + 1) + wall_w / 2)], fill="black", width=wall_w)
                if r + 1 < grid_size and _passage_key((r, c), (r + 1, c)) not in passages:
                    y = cell_y(r + 1)
                    draw.line([(cell_x(c) - wall_w / 2, y), (cell_x(c + 1) + wall_w / 2, y)], fill="black", width=wall_w)

        # Start point (green dot).
        sr, sc = meta["start"]
        cx, cy = cell_x(sc + 0.5), cell_y(sr + 0.5)
        radius = min(cell_w, cell_h) * 0.18
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 128, 0))

        # Destination letters (red).
        font = cls._load_font(int(min(cell_w, cell_h) * 0.45))
        for letter, (r, c) in meta["destinations"].items():
            cx, cy = cell_x(c + 0.5), cell_y(r + 0.5)
            bbox = draw.textbbox((0, 0), letter, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), letter, fill=(220, 20, 20), font=font)

        return img

    # ------------------------------------------------------ prompt context text

    @staticmethod
    def build_prompt_context_text(meta: dict[str, Any]) -> str:
        """Grounded maze facts injected into the Round-1 Proposer prompt."""
        grid_size = int(meta["grid_size"])
        sr, sc = meta["start"]
        lines = [
            f"- Grid size: {grid_size}x{grid_size} (rows and columns are 0-indexed; row 0 is the TOP row, column 0 is the LEFT column).",
            f"- Start point (green dot): row {sr}, column {sc}.",
        ]
        for letter, (r, c) in meta["destinations"].items():
            lines.append(f"- Destination {letter} (red letter): row {r}, column {c}.")
        lines.append("- Verified shortest action sequences from the start (moving one grid cell per action, never crossing a wall):")
        for letter, path in meta.get("shortest_paths", {}).items():
            if path:
                seq = " ".join(f"Go {m}." for m in path)
                lines.append(f"  * To {letter} ({len(path)} moves): {seq}")
        return "\n".join(lines)

    # ------------------------------------------------------- pick-letter helper

    @classmethod
    def build_question_from_letter(cls, letter: str, image_path: Optional[str]) -> Optional[dict[str, Any]]:
        """Construct a valid maze-navigation question from a chosen destination letter.

        Used by the maze-only self-evolve variant where the Round-1 Questioner
        only picks A/B/C/D; the system looks up the verified shortest path for
        that destination and builds the full question text. The result is
        always valid by construction (the path comes from BFS on the maze graph).
        """
        if not image_path or not letter:
            return None
        letter = letter.strip().upper()
        if letter not in ("A", "B", "C", "D"):
            return None
        try:
            meta = cls.load_metadata_for_image(image_path)
        except Exception:
            return None
        path = (meta.get("shortest_paths") or {}).get(letter)
        if not path:
            return None
        seq = " ".join(f"Go {m}." for m in path)
        question_text = (
            "Task: Maze Navigation Simulation\n"
            "Determine the final destination (A, B, C or D) from the starting point (green point) "
            "following the action sequence.\n"
            "The definitions of the actions are as below.\n"
            "* Go up/left/down/right: move one grid space in the absolute up/left/down/right direction.\n"
            f"Full Action Sequence: {seq}"
        )
        return {
            "question_text": question_text,
            "answer": letter,
            "path": path,
            "num_moves": len(path),
        }

    # ---------------------------------------------------------------- verifier

    @classmethod
    def validate_and_answer(cls, question_text: str, image_path: Optional[str]) -> dict[str, Any]:
        """Simulate the proposed action sequence; return validity + GT letter."""

        def _invalid(error: str, error_code: str, failure_stage: str, parsed: Optional[dict] = None) -> dict[str, Any]:
            return {
                "task_type": "maze_navigation",
                "is_valid": False,
                "answer": "invalid",
                "error": error,
                "error_code": error_code,
                "failure_stage": failure_stage,
                "parsed_params": parsed or {},
                "validation_result": {"is_valid": False, "reason": error},
                "judge_reference": {
                    "invalid_reason": error,
                    "error_code": error_code,
                    "failure_stage": failure_stage,
                },
            }

        if not image_path:
            return _invalid("no maze image associated with this question", "maze_no_image", "input_contract")
        try:
            meta = cls.load_metadata_for_image(image_path)
        except Exception as exc:
            return _invalid(f"failed to load maze metadata: {exc}", "maze_metadata_missing", "input_contract")

        grid_size = int(meta["grid_size"])
        passages = {_passage_key(tuple(a), tuple(b)) for a, b in meta["passages"]}
        # Only extract moves from the "Full Action Sequence:" line, not from the
        # instruction line which contains "Go up/left/down/right" as a description.
        action_seq_text = question_text or ""
        seq_marker = "Full Action Sequence:"
        marker_idx = action_seq_text.find(seq_marker)
        if marker_idx >= 0:
            action_seq_text = action_seq_text[marker_idx + len(seq_marker):]
        moves = [m.lower() for m in MOVE_PATTERN.findall(action_seq_text)]
        parsed = {"moves": moves}

        if ANSWER_LEAK_PATTERN.search(str(question_text or "")):
            return _invalid(
                "the question text reveals the intended destination letter; the solver must derive it from the action sequence",
                "maze_answer_leak",
                "validation",
                parsed,
            )
        if not moves:
            return _invalid(
                "no action sequence found; the question must contain moves like 'Go up. Go left.'",
                "maze_no_action_sequence",
                "extraction",
                parsed,
            )
        if len(moves) < MIN_MOVES:
            return _invalid(
                f"action sequence too short ({len(moves)} moves; minimum {MIN_MOVES})",
                "maze_sequence_too_short",
                "validation",
                parsed,
            )
        if len(moves) > MAX_MOVES:
            return _invalid(
                f"action sequence too long ({len(moves)} moves; maximum {MAX_MOVES})",
                "maze_sequence_too_long",
                "validation",
                parsed,
            )

        r, c = meta["start"]
        for i, move in enumerate(moves):
            dr, dc = MOVE_DELTAS[move]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                return _invalid(
                    f"move {i + 1} ('Go {move}') leaves the maze boundary from cell (row {r}, col {c})",
                    "maze_out_of_bounds",
                    "simulation",
                    {**parsed, "failed_move_index": i, "cell_before_failure": [r, c]},
                )
            if _passage_key((r, c), (nr, nc)) not in passages:
                return _invalid(
                    f"move {i + 1} ('Go {move}') crosses a wall between cell (row {r}, col {c}) and cell (row {nr}, col {nc})",
                    "maze_wall_crossing",
                    "simulation",
                    {**parsed, "failed_move_index": i, "cell_before_failure": [r, c]},
                )
            r, c = nr, nc

        parsed["final_cell"] = [r, c]
        for letter, cell in meta["destinations"].items():
            if [r, c] == list(cell):
                return {
                    "task_type": "maze_navigation",
                    "is_valid": True,
                    "answer": letter,
                    "error": "",
                    "error_code": "",
                    "failure_stage": "ok",
                    "parsed_params": parsed,
                    "validation_result": {"is_valid": True},
                    "judge_reference": None,
                }
        return _invalid(
            f"the action sequence ends at cell (row {r}, col {c}), which is not one of the labeled destinations A-D",
            "maze_not_at_destination",
            "simulation",
            parsed,
        )
