# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Round-2 Solver reward for the AGENTIC (ViLaSR think->draw->observe) protocol.

Composition (mirrors VERSE's Round-2 reward, with the format term replaced
by the ViLaSR gcot multi-turn format check):

    valid question:   r = 0.1 * f_fmt_gcot + 0.9 * f_acc
    invalid question: r = 0.1 * f_fmt_gcot + 0.9 * f_explain   (LLM judge)

- f_fmt_gcot: every assistant turn must be `<think>...</think> Action: ...`,
  tool JSON blocks must parse as Object Mapper / Path Tracer calls, and only the
  final turn may (and must) contain `<answer>...</answer>`. Ported verbatim from
  ViLaSR verl/utils/reward_score/gcot.py.
- f_acc: multiple-choice letters are mapped back through `mc_options_json` and
  scored with VERSE's task-specific `answer_correctness_score` (numeric
  tasks use MRA); maze_navigation is exact letter match.
- A `reflection` metric (same-label bbox re-drawn with new coordinates) is
  logged like in ViLaSR but NOT added to the overall reward.
"""

import ast
import json
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, TypedDict

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from majority_correctness_reward import (  # noqa: E402
    _extract_last_answer_tag,
    _resolve_invalid_reason_judge_max_workers,
    answer_correctness_score,
    judge_invalid_reason_explanation,
)

REWARD_NAME = "agentic_solver"
REWARD_TYPE = "batch"
ROUND2_FORMAT_WEIGHT = 0.1


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


# --------------------------------------------------------------------------
# gcot multi-turn format check (ported from ViLaSR verl/utils/reward_score/gcot.py)
# --------------------------------------------------------------------------

def parse_bbox_and_movement(response: str):
    def parse_json(json_output):
        json_output_list = []
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                tmp = "\n".join(lines[i + 1 :])
                tmp = tmp.split("```")[0]
                json_output_list.append(tmp)
        return json_output_list

    def safe_eval(item):
        try:
            return ast.literal_eval(item)
        except (ValueError, SyntaxError):
            return None

    parsed_list = [safe_eval(item) for item in parse_json(response)]
    fail_parse_response_list = []
    bbox_list, movement_list, not_bbox_movement_list = [], [], []
    for item_list in parsed_list:
        if item_list is None:
            fail_parse_response_list.append(item_list)
            continue
        try:
            items = list(item_list)
        except TypeError:
            fail_parse_response_list.append(item_list)
            continue
        for item in items:
            if not isinstance(item, dict):
                not_bbox_movement_list.append(item)
            elif "bbox_2d" in item and "label" in item:
                bbox_list.append(item)
            elif "start_point_2d" in item and "end_point_2d" in item and "label" in item:
                movement_list.append(item)
            else:
                not_bbox_movement_list.append(item)
    return bbox_list, movement_list, fail_parse_response_list, not_bbox_movement_list


def parse_dialog(serialized_content: str) -> list[dict[str, Any]]:
    segments = re.split(r"<\|im_start\|>|<\|im_end\|>", str(serialized_content or ""))
    segments = [s for s in segments if s]

    conversations = []
    if segments and segments[0].startswith("system"):
        conversations.append({"role": "system", "content": segments[0].replace("system\n\n", "", 1)})
        segments = segments[1:]

    for segment in segments:
        if segment.startswith("user"):
            text = segment.replace("user\n", "", 1)
            text = text.replace("<|vision_start|><|image_pad|><|vision_end|>\n", "", 1)
            conversations.append({"role": "user", "content": text})
        elif segment.startswith("assistant"):
            conversations.append({"role": "assistant", "content": segment.replace("assistant\n", "", 1)})
    return conversations


def format_reward(conversations: list[dict[str, Any]]) -> tuple[float, float]:
    """Return (format in {0,1}, reflection in {0,1}) over the parsed dialog."""
    format_flag = True
    total_content = ""
    assistant_indices = [i for i, turn in enumerate(conversations) if turn["role"] == "assistant"]
    if not assistant_indices:
        return 0.0, 0.0
    last_assistant_idx = assistant_indices[-1]
    for i in assistant_indices:
        content = conversations[i]["content"]
        pattern_action = r"<think>(.*?)</think>\s*Action:\s*(.*?)(?=\n|$)"
        match_action = re.fullmatch(pattern_action, content, re.DOTALL)
        if (
            match_action is None
            or (i == last_assistant_idx and ("<answer>" not in content or "</answer>" not in content))
            or (i != last_assistant_idx and ("<answer>" in content or "</answer>" in content))
        ):
            format_flag = False
            break
        total_content += content

    bbox_list = []
    if format_flag:
        bbox_list, movement_list, fail_parse_response_list, not_bbox_movement_list = parse_bbox_and_movement(
            total_content
        )
        if len(fail_parse_response_list) > 0 or len(not_bbox_movement_list) > 0:
            format_flag = False

    if not format_flag:
        return 0.0, 0.0

    # Reflection: same bbox label re-drawn with different coordinates.
    bbox_dict: dict[str, set] = {}
    reflection = 0.0
    for bbox in bbox_list:
        try:
            bbox_2d = tuple(map(float, bbox["bbox_2d"]))
            label = bbox["label"]
        except (TypeError, ValueError, KeyError):
            continue
        if label not in bbox_dict:
            bbox_dict[label] = {bbox_2d}
        elif bbox_2d not in bbox_dict[label]:
            reflection = 1.0
            bbox_dict[label].add(bbox_2d)
    return 1.0, reflection


# --------------------------------------------------------------------------
# Engine-verified maze trajectory fidelity (process reward)
# --------------------------------------------------------------------------
# The maze image geometry is fully deterministic (MazeEngine), so the Solver's
# Path-Tracer drawing can be checked against the true cell path implied by the
# question's action sequence. This turns "faithful drawing" into a verifiable,
# zero-LLM process signal (2026-08-13 demo showed the model draws ~0.69-cell
# steps and lands outside any labeled cell while still guessing the right
# letter — outcome reward alone cannot see that).
#
# Enable reward shaping with SPATIALEVO_MAZE_FIDELITY_WEIGHT=w (0<w<0.9):
#     maze valid:  r = 0.1*fmt + (0.9-w)*acc + w*fidelity
# With w=0 (default) fidelity is still computed and logged as a diagnostic.

_MAZE_MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _maze_expected_cells(start: list, moves: list[str]) -> list[tuple[int, int]]:
    cells = [(int(start[0]), int(start[1]))]
    r, c = cells[0]
    for move in moves:
        dr, dc = _MAZE_MOVE_DELTAS.get(str(move).lower(), (0, 0))
        r, c = r + dr, c + dc
        cells.append((r, c))
    return cells


def _maze_snap_point_to_cell(x: float, y: float, ref: dict) -> tuple[int, int]:
    """Snap an absolute pixel point (processed-image frame) to the nearest cell."""
    grid = int(ref["grid_size"])
    width = float(ref["image_width"])
    height = float(ref["image_height"])
    ml, mr, mt, mb = [float(v) for v in ref["margins"]]
    # Fractional position inside the usable grid area.
    fx = (x / width - ml) / max(1e-9, 1.0 - ml - mr)
    fy = (y / height - mt) / max(1e-9, 1.0 - mt - mb)
    col = int(min(grid - 1, max(0, math.floor(fx * grid))))
    row = int(min(grid - 1, max(0, math.floor(fy * grid))))
    return row, col


def _lcs_length(a: list, b: list) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


def maze_trajectory_fidelity(movement_list: list[dict], ref: dict) -> dict[str, float]:
    """Score how faithfully the drawn Path-Tracer trajectory tracks the GT cell path.

    Returns dict with:
      - path_score: LCS(drawn cell path, expected cell path) / len(expected)
      - endpoint_hit: 1.0 when the final drawn point snaps to the GT final cell
      - start_hit: 1.0 when the first drawn point snaps to the start cell
      - fidelity: 0.6*path_score + 0.4*endpoint_hit
      - num_segments: number of parsed Path-Tracer segments
    """
    out = {
        "fidelity": 0.0,
        "path_score": 0.0,
        "endpoint_hit": 0.0,
        "start_hit": 0.0,
        "num_segments": float(len(movement_list or [])),
    }
    try:
        expected_cells = _maze_expected_cells(ref["start"], ref["moves"])
    except (KeyError, TypeError, ValueError, IndexError):
        return out
    if not movement_list:
        return out

    points: list[tuple[float, float]] = []
    for seg_idx, seg in enumerate(movement_list):
        try:
            sx, sy = [float(v) for v in seg["start_point_2d"][:2]]
            ex, ey = [float(v) for v in seg["end_point_2d"][:2]]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if not points:
            points.append((sx, sy))
        points.append((ex, ey))
    if not points:
        return out

    drawn_cells: list[tuple[int, int]] = []
    for x, y in points:
        cell = _maze_snap_point_to_cell(x, y, ref)
        if not drawn_cells or drawn_cells[-1] != cell:
            drawn_cells.append(cell)

    lcs = _lcs_length(drawn_cells, expected_cells)
    path_score = float(lcs) / float(max(1, len(expected_cells)))
    endpoint_hit = 1.0 if drawn_cells[-1] == expected_cells[-1] else 0.0
    start_hit = 1.0 if drawn_cells[0] == expected_cells[0] else 0.0
    out.update(
        {
            "path_score": path_score,
            "endpoint_hit": endpoint_hit,
            "start_hit": start_hit,
            "fidelity": 0.6 * path_score + 0.4 * endpoint_hit,
        }
    )
    return out


def _maze_fidelity_weight() -> float:
    try:
        weight = float(os.environ.get("SPATIALEVO_MAZE_FIDELITY_WEIGHT", "0") or 0.0)
    except ValueError:
        return 0.0
    return min(max(weight, 0.0), 0.9 - ROUND2_FORMAT_WEIGHT)


# --------------------------------------------------------------------------
# Engine-verified VSI grounding fidelity (process reward)
# --------------------------------------------------------------------------
# ScanNet per-frame metadata carries ground-truth 2D bboxes for every labeled
# object, so the Solver's Object-Mapper boxes can be checked against the real
# geometry of the objects the question references: did the model actually
# localize the queried objects, or draw a box somewhere and guess? This is the
# VSI analogue of the maze Path-Tracer fidelity — a zero-LLM, deterministic
# process signal aimed at the "draw-then-guess" decoupling (VSI is otherwise
# outcome-only). The trainer scales GT boxes into the processed coordinate frame
# the Solver draws in; all scene frames share one source size, so GT instances
# from every frame live in one comparable frame and we match a drawn box to the
# GT of the same label by max IoU across frames (index-agnostic, robust to the
# model's image-index convention).
#
# Enable reward shaping with SPATIALEVO_VSI_GROUNDING_WEIGHT=w (0<w<0.9):
#     scene valid: r = 0.1*fmt + (0.9-w)*acc + w*grounding
# With w=0 (default) grounding is still computed and logged as a diagnostic.

# Scene task types whose questions reference concrete, groundable objects.
VSI_GROUNDING_TASKS = {
    "object_counting",
    "absolute_distance",
    "object_size",
    "relative_distance",
    "relative_direction_hard",
    "route_planning",
}
_VSI_GROUNDING_IOU_HIT = 0.5


def _bbox_iou(a, b) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    except (TypeError, ValueError, IndexError):
        return 0.0
    if ax1 > ax2:
        ax1, ax2 = ax2, ax1
    if ay1 > ay2:
        ay1, ay2 = ay2, ay1
    if bx1 > bx2:
        bx1, bx2 = bx2, bx1
    if by1 > by2:
        by1, by2 = by2, by1
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def vsi_grounding_fidelity(bbox_list: list, ref: dict) -> dict[str, float]:
    """Score how well the Solver's Object-Mapper boxes localize the queried objects.

    ref = {
      "relevant_labels": [str, ...],           # objects the question is about
      "gt_by_label": {label_lower: [[x1,y1,x2,y2], ...]},  # GT in processed coords,
                                                # union over all frames
    }

    Returns dict with:
      - grounding: mean over relevant labels of the best drawn-vs-GT IoU (0 for a
        relevant label the model never drew / never matched)
      - grounding_hit_rate: fraction of relevant labels with best IoU >= 0.5
      - coverage: fraction of relevant labels the model drew a box for at all
      - mean_matched_iou: mean best IoU over relevant labels that were drawn
      - num_relevant / num_drawn_bbox
    """
    out = {
        "grounding": 0.0,
        "grounding_hit_rate": 0.0,
        "coverage": 0.0,
        "mean_matched_iou": 0.0,
        "num_relevant": 0.0,
        "num_drawn_bbox": float(len(bbox_list or [])),
    }
    relevant = [str(l).strip().lower() for l in (ref.get("relevant_labels") or []) if str(l).strip()]
    gt_by_label = ref.get("gt_by_label") or {}
    if not relevant:
        return out
    out["num_relevant"] = float(len(relevant))

    drawn_by_label: dict[str, list] = defaultdict(list)
    for b in (bbox_list or []):
        if not isinstance(b, dict):
            continue
        lbl = str(b.get("label", "")).strip().lower()
        box = b.get("bbox_2d")
        if lbl and isinstance(box, (list, tuple)) and len(box) >= 4:
            drawn_by_label[lbl].append(box)

    def _match(store: dict, label: str) -> list:
        if label in store:
            return list(store[label])
        hits: list = []
        for k, v in store.items():
            if label == k or label in k or k in label:
                hits.extend(v)
        return hits

    label_ious: list[float] = []
    drawn_flags: list[float] = []
    for label in relevant:
        gts = _match(gt_by_label, label)
        drawn = _match(drawn_by_label, label)
        drawn_flags.append(1.0 if drawn else 0.0)
        best = 0.0
        for db in drawn:
            for gt in gts:
                iou = _bbox_iou(db, gt)
                if iou > best:
                    best = iou
        label_ious.append(best)

    n = float(len(relevant))
    out["grounding"] = sum(label_ious) / n
    out["grounding_hit_rate"] = sum(1.0 for v in label_ious if v >= _VSI_GROUNDING_IOU_HIT) / n
    out["coverage"] = sum(drawn_flags) / n
    matched = [iou for iou, drew in zip(label_ious, drawn_flags) if drew > 0]
    out["mean_matched_iou"] = (sum(matched) / len(matched)) if matched else 0.0
    return out


def _vsi_grounding_weight() -> float:
    try:
        weight = float(os.environ.get("SPATIALEVO_VSI_GROUNDING_WEIGHT", "0") or 0.0)
    except ValueError:
        return 0.0
    return min(max(weight, 0.0), 0.9 - ROUND2_FORMAT_WEIGHT)


# --------------------------------------------------------------------------
# Accuracy scoring
# --------------------------------------------------------------------------

def _letter_to_option_label(answer_text: str, mc_options_json: str) -> tuple[Optional[str], Optional[str]]:
    """Return (letter, mapped_label) when the answer looks like an option letter."""
    text = str(answer_text or "").strip()
    match = re.match(r"^\(?([A-Ha-h])[\).:]?\s*$", text) or re.match(r"^\(?([A-Ha-h])[\).:]\s+", text)
    if not match:
        return None, None
    letter = match.group(1).upper()
    try:
        options = json.loads(mc_options_json) if mc_options_json else None
    except Exception:
        options = None
    if isinstance(options, list):
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(options):
            return letter, str(options[idx])
    return letter, None


def _accuracy_for_valid(
    answer_text: str,
    gt: str,
    task_type: str,
    *,
    mc_options_json: str = "",
    gt_letter: str = "",
    answer_style: str = "",
) -> float:
    answer_text = str(answer_text or "").strip()
    gt_letter = str(gt_letter or "").strip().upper()

    if task_type == "maze_navigation":
        letter, _ = _letter_to_option_label(answer_text, mc_options_json)
        candidate = letter or answer_text.strip().upper()[:1]
        return 1.0 if candidate and candidate == str(gt or "").strip().upper() else 0.0

    if answer_style == "multiple choice" and gt_letter:
        letter, mapped_label = _letter_to_option_label(answer_text, mc_options_json)
        if letter is not None:
            if letter == gt_letter:
                return 1.0
            # Letter mismatched; fall through with the mapped label so partial
            # direction-set matching can still award partial credit.
            if mapped_label is not None:
                return answer_correctness_score(mapped_label, gt, task_type)
            return 0.0
        # Model answered with the raw label instead of the letter.
        return answer_correctness_score(answer_text, gt, task_type)

    return answer_correctness_score(answer_text, gt, task_type)


# --------------------------------------------------------------------------
# Batch entry point
# --------------------------------------------------------------------------

def compute_score(reward_inputs: list[dict[str, Any]], **kwargs) -> list[RewardScore]:
    judge_url = kwargs.get("invalid_reason_judge_url", kwargs.get("service_url", kwargs.get("url", "https://example.com/v1")))
    judge_timeout = kwargs.get("invalid_reason_judge_timeout", kwargs.get("timeout", 30))
    judge_model = kwargs.get("invalid_reason_judge_model", kwargs.get("model", "gpt-oss-120b-ldm"))
    judge_api_key = kwargs.get("invalid_reason_judge_api_key", kwargs.get("api_key", ""))
    judge_max_workers = _resolve_invalid_reason_judge_max_workers(kwargs)

    sample_details = []
    invalid_jobs = []

    for idx, reward_input in enumerate(reward_inputs):
        uid = reward_input.get("uid", f"unknown_{idx}")
        gt = reward_input.get("gt", "")
        q_type = str(reward_input.get("question_type", "") or "")
        is_valid = bool(reward_input.get("is_valid", True))
        # Prefer the rollout-provided raw texts; fall back to the manager decode.
        sequence_text = str(reward_input.get("agentic_sequence_text", "") or "")
        response_text = str(reward_input.get("agentic_response_text", "") or reward_input.get("response", "") or "")

        if sequence_text:
            conversations = parse_dialog(sequence_text)
            fmt, reflection = format_reward(conversations)
        else:
            conversations = []
            fmt, reflection = 0.0, 0.0

        answer_text = _extract_last_answer_tag(response_text) or ""
        detail = {
            "uid": uid,
            "is_valid": is_valid,
            "format_score": float(fmt),
            "reflection": float(reflection),
            "r_acc": 0.0,
            "invalid_reason_score": 0.0,
            "content": answer_text,
            "maze_fidelity": None,
            "vsi_grounding": None,
        }

        # Engine-verified maze process signal (always computed for diagnostics;
        # only shaped into the reward when SPATIALEVO_MAZE_FIDELITY_WEIGHT > 0).
        if q_type == "maze_navigation" and is_valid:
            ref_json = str(reward_input.get("maze_fidelity_ref_json", "") or "")
            if ref_json:
                try:
                    ref = json.loads(ref_json)
                except (TypeError, ValueError):
                    ref = None
                if isinstance(ref, dict) and ref.get("grid_size"):
                    assistant_text = "".join(
                        turn["content"] for turn in conversations if turn["role"] == "assistant"
                    )
                    _, movement_list, _, _ = parse_bbox_and_movement(assistant_text)
                    detail["maze_fidelity"] = maze_trajectory_fidelity(movement_list, ref)

        # Engine-verified VSI grounding process signal (always computed for
        # diagnostics; only shaped into the reward when the weight env > 0).
        if q_type in VSI_GROUNDING_TASKS and is_valid:
            gref_json = str(reward_input.get("vsi_grounding_ref_json", "") or "")
            if gref_json:
                try:
                    gref = json.loads(gref_json)
                except (TypeError, ValueError):
                    gref = None
                if isinstance(gref, dict) and gref.get("relevant_labels"):
                    assistant_text = "".join(
                        turn["content"] for turn in conversations if turn["role"] == "assistant"
                    )
                    bbox_list, _, _, _ = parse_bbox_and_movement(assistant_text)
                    detail["vsi_grounding"] = vsi_grounding_fidelity(bbox_list, gref)

        if is_valid:
            detail["r_acc"] = float(
                _accuracy_for_valid(
                    answer_text,
                    gt,
                    q_type,
                    mc_options_json=str(reward_input.get("mc_options_json", "") or ""),
                    gt_letter=str(reward_input.get("gt_letter", "") or ""),
                    answer_style=str(reward_input.get("answer_style", "") or ""),
                )
            )
            sample_details.append(detail)
            continue

        # Invalid question: judge the model's explanation of the invalidity.
        explanation = answer_text or response_text
        detail["content"] = explanation
        detail["judge_payload"] = {
            "task_type": q_type,
            "question": str(reward_input.get("question", "") or ""),
            "explanation": explanation,
            "task_reference_text": reward_input.get("round2_task_reference", ""),
            "sim_error_code": reward_input.get("sim_error_code", ""),
            "sim_failure_stage": reward_input.get("sim_failure_stage", ""),
            "sim_error": reward_input.get("sim_error", ""),
            "sim_judge_reference_json": reward_input.get("sim_judge_reference_json", ""),
            "sim_parsed_params_json": reward_input.get("sim_parsed_params_json", ""),
            "sim_validation_result_json": reward_input.get("sim_validation_result_json", ""),
            "sim_result_json": reward_input.get("sim_result_json", ""),
        }
        sample_details.append(detail)
        if explanation:
            invalid_jobs.append(len(sample_details) - 1)

    if invalid_jobs:
        def _run_judge(idx: int) -> tuple[int, float]:
            score = judge_invalid_reason_explanation(
                **sample_details[idx]["judge_payload"],
                ak=judge_api_key,
                model=judge_model,
                url=judge_url,
                timeout=judge_timeout,
            )
            return idx, float(score)

        workers = min(judge_max_workers, len(invalid_jobs))
        if workers > 1 and len(invalid_jobs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_judge, idx): idx for idx in invalid_jobs}
                for future in as_completed(futures):
                    fidx = futures[future]
                    try:
                        idx, score = future.result()
                        sample_details[idx]["invalid_reason_score"] = score
                    except Exception as exc:
                        print(f"Invalid-reason judge failed (idx={fidx}): {exc}")
        else:
            for idx in invalid_jobs:
                try:
                    _, score = _run_judge(idx)
                    sample_details[idx]["invalid_reason_score"] = score
                except Exception as exc:
                    print(f"Invalid-reason judge failed (idx={idx}): {exc}")

    scores: list[RewardScore] = []
    fidelity_weight = _maze_fidelity_weight()
    grounding_weight = _vsi_grounding_weight()
    for detail in sample_details:
        fmt = float(detail["format_score"])
        r_acc = float(detail["r_acc"])
        content_reward = r_acc if detail["is_valid"] else float(detail["invalid_reason_score"])
        overall = (ROUND2_FORMAT_WEIGHT * fmt) + ((1.0 - ROUND2_FORMAT_WEIGHT) * content_reward)
        maze_fid = detail.get("maze_fidelity")
        vsi_g = detail.get("vsi_grounding")
        if isinstance(maze_fid, dict) and fidelity_weight > 0.0 and detail["is_valid"]:
            overall = (
                (ROUND2_FORMAT_WEIGHT * fmt)
                + ((1.0 - ROUND2_FORMAT_WEIGHT - fidelity_weight) * content_reward)
                + (fidelity_weight * float(maze_fid["fidelity"]))
            )
        elif isinstance(vsi_g, dict) and grounding_weight > 0.0 and detail["is_valid"]:
            overall = (
                (ROUND2_FORMAT_WEIGHT * fmt)
                + ((1.0 - ROUND2_FORMAT_WEIGHT - grounding_weight) * content_reward)
                + (grounding_weight * float(vsi_g["grounding"]))
            )
        score_item = {
            "overall": overall,
            "format": fmt,
            "accuracy": r_acc,
            "answer_correctness": r_acc,
            "reflection": float(detail["reflection"]),
            "invalid_reason_score": float(detail["invalid_reason_score"]),
            "valid_case": 1.0 if detail["is_valid"] else 0.0,
            "invalid_case": 0.0 if detail["is_valid"] else 1.0,
        }
        if isinstance(maze_fid, dict):
            score_item["maze_traj_fidelity"] = float(maze_fid["fidelity"])
            score_item["maze_traj_path_score"] = float(maze_fid["path_score"])
            score_item["maze_traj_endpoint_hit"] = float(maze_fid["endpoint_hit"])
            score_item["maze_traj_start_hit"] = float(maze_fid["start_hit"])
            score_item["maze_traj_num_segments"] = float(maze_fid["num_segments"])
        if isinstance(vsi_g, dict):
            score_item["vsi_grounding"] = float(vsi_g["grounding"])
            score_item["vsi_grounding_hit_rate"] = float(vsi_g["grounding_hit_rate"])
            score_item["vsi_grounding_coverage"] = float(vsi_g["coverage"])
            score_item["vsi_grounding_mean_iou"] = float(vsi_g["mean_matched_iou"])
        scores.append(score_item)

    # Diversity diagnostics across GRPO groups (same as majority_correctness).
    try:
        uid2contents = defaultdict(list)
        for d in sample_details:
            uid2contents[d["uid"]].append(d["content"])
        total_groups = len(uid2contents)
        uniq_ge2 = sum(1 for v in uid2contents.values() if len(set(v)) >= 2)
        if scores and total_groups > 0:
            scores[0].update(
                {
                    "diversity/num_groups": float(total_groups),
                    "diversity/uniq_ge2_ratio": float(uniq_ge2) / float(total_groups),
                }
            )
    except Exception:
        pass

    return scores
