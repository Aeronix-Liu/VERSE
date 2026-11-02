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

import base64
import json
import mimetypes
import numpy as np
import os
import re
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, TypedDict
import time
from openai import OpenAI

class RewardInput(TypedDict, total=False):
    """Reward Input."""
    response: str
    response_length: int
    ground_truth: str
    multi_modal_data: Optional[dict]
    uid: Optional[str]  # Group id for GRPO.
    round: Optional[int]  # 1 for question generation, 2 for answer generation.
    gt: str
    question_type: str
    is_valid: bool
    question: str
    round2_mode: str
    round2_task_reference: str
    sim_error: str
    sim_error_code: str
    sim_failure_stage: str
    sim_parsed_params_json: str
    sim_validation_result_json: str
    sim_judge_reference_json: str
    sim_result_json: str

class RewardScore(TypedDict):
    """Reward Score."""
    overall: float
    format: Optional[float]
    accuracy: Optional[float]

REWARD_NAME = "answer_correctness"
REWARD_TYPE = "batch"  # Batch-level reward.

TASK_ANSWER_HINTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "format_prompt", "task_answer_format_hints.json")
)
DEFAULT_SUPPORTED_TASK_TYPES = {
    "object_counting",
    "relative_distance",
    "relative_direction_hard",
    "object_size",
    "absolute_distance",
    "room_size",
    "single_image_relative_direction",
    "distance_cam_obj",
    "depth_order_obj_obj",
    "position_cam_cam",
    "elevation_cam_cam",
    "visibility_compare",
    "position_cam_obj",
    "position_cam_reg",
    "motion_camera",
    "attribute_measurement",
}

try:
    with open(TASK_ANSWER_HINTS_PATH, "r", encoding="utf-8") as f:
        TASK_ANSWER_HINTS = json.load(f).get("tasks") or {}
except Exception:
    TASK_ANSWER_HINTS = {}

SUPPORTED_TASK_TYPES = set(TASK_ANSWER_HINTS.keys()) or DEFAULT_SUPPORTED_TASK_TYPES

# Task groups defined by the answer-format hints.
DIRECTION_MATCH_TASK_TYPES = {
    "relative_direction_hard",
    "single_image_relative_direction",
    "position_cam_cam",
    "position_cam_obj",
    "position_cam_reg",
    "motion_camera",
}
COUNT_TASK_TYPES = {
    "object_counting",
}
METRIC_DISTANCE_TASK_TYPES = {
    "absolute_distance",
    "distance_cam_obj",
}
OBJECT_SIZE_TASK_TYPES = {
    "object_size",
}
ROOM_SIZE_TASK_TYPES = {
    "room_size",
}
ROUTE_PLANNING_TASK_TYPES = {
    "route_planning",
}
EXACT_MATCH_TASK_TYPES = {
    "relative_distance",
    "depth_order_obj_obj",
    "elevation_cam_cam",
    "visibility_compare",
    "attribute_measurement",
}
LEGACY_EXACT_MATCH_TASK_TYPES = {"appearance_order"}
INVALID_REASON_SCORE_LEVELS = (0.0, 0.3, 0.6, 1.0)
ROUND2_FORMAT_WEIGHT = 0.1

# --- Split-role PRM (process reward for the Solver's step-by-step reasoning) ---
# For VALID questions the legacy reward only scores the final answer (ORM). The
# PRM augments this with a scene-grounded, GT-aware, step-level reasoning score
# judged by the LOCAL frozen-Proposer vLLM server (NOT an external API). The
# per-step scores are aggregated (mean by default) and blended with answer
# correctness. Disabled by default so non-PRM runs are byte-for-byte unchanged.
PRM_STEP_SCORE_LEVELS = (0.0, 0.3, 0.6, 1.0)
PRM_DEFAULT_ANSWER_WEIGHT = 0.7
PRM_DEFAULT_REASONING_WEIGHT = 0.3
PRM_MAX_STEPS = 8
PRM_MIN_STEP_CHARS = 12

def clean_text(text, exclude_chars=['\n', '\r']):
    text = "" if text is None else str(text)

    for char in exclude_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

DIRECTION_ALIAS_MAP = {
    "behind": "back",
    "backward": "back",
    "backwards": "back",
    "rear": "back",
    "forward": "front",
    "forwards": "front",
    "ahead": "front",
    "above": "up",
    "below": "down",
}
DIRECTION_BASE_TOKENS = {"left", "right", "front", "back", "up", "down"}
EXACT_ARTICLES = ("the ", "a ", "an ")
VISIBILITY_EQUIVALENTS = {
    "image1": "image1",
    "image 1": "image1",
    "image one": "image1",
    "the first image": "image1",
    "first image": "image1",
    "image2": "image2",
    "image 2": "image2",
    "image two": "image2",
    "the second image": "image2",
    "second image": "image2",
}
EXACT_EQUIVALENTS = {
    "same level": "same_level",
    "same-level": "same_level",
    "samelevel": "same_level",
}
NUMBER_WORD_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
NUMBER_WORD_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
NUMBER_WORD_SCALES = {
    "hundred": 100,
    "thousand": 1000,
}
NUMBER_PATTERN = re.compile(
    r"(?<![a-zA-Z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![a-zA-Z])"
)

def _normalize_exact_answer(text: str, q_type: str = "") -> str:
    normalized = clean_text(text)
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if q_type == "visibility_compare":
        return VISIBILITY_EQUIVALENTS.get(normalized, normalized)

    normalized = EXACT_EQUIVALENTS.get(normalized, normalized)
    for article in EXACT_ARTICLES:
        if normalized.startswith(article):
            normalized = normalized[len(article):].strip()
            break
    return normalized

@lru_cache(maxsize=256)
def _split_compound_direction_token(token: str) -> tuple[str, ...]:
    normalized = str(token or "").strip().lower()
    if not normalized:
        return ()

    normalized = DIRECTION_ALIAS_MAP.get(normalized, normalized)
    if normalized in DIRECTION_BASE_TOKENS:
        return (normalized,)

    compact = normalized.replace("_", "").replace("-", "")
    if not compact:
        return ()

    candidates: list[tuple[str, ...]] = []

    def _dfs(remaining: str, parts: list[str]) -> None:
        if not remaining:
            candidates.append(tuple(parts))
            return

        for base in sorted(DIRECTION_BASE_TOKENS, key=len, reverse=True):
            if remaining.startswith(base):
                parts.append(base)
                _dfs(remaining[len(base):], parts)
                parts.pop()

    _dfs(compact, [])

    valid_candidates = [parts for parts in candidates if 1 <= len(parts) <= 3]
    if not valid_candidates:
        return ()

    valid_candidates.sort(key=lambda parts: (len(parts), parts))
    return valid_candidates[0]

def _normalize_direction_parts(text: str, q_type: str = "") -> tuple[str, ...]:
    raw = clean_text(text).replace("_", " ").replace("-", " ")
    tokens = re.findall(r"[a-zA-Z]+", raw)
    normalized_parts: list[str] = []

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token == "counter" and index + 1 < len(tokens) and tokens[index + 1] == "clockwise":
            token = "counterclockwise"
            index += 1
        elif token == "same" and index + 1 < len(tokens) and tokens[index + 1] == "level":
            token = "same_level"
            index += 1

        if q_type == "motion_camera":
            if token == "clockwise":
                token = "right"
            elif token == "counterclockwise":
                token = "left"

        parts = _split_compound_direction_token(token)
        if not parts:
            token = DIRECTION_ALIAS_MAP.get(token, token)
            if token in DIRECTION_BASE_TOKENS:
                normalized_parts.append(token)
        else:
            normalized_parts.extend(parts)

        index += 1

    return tuple(sorted(set(normalized_parts)))

def direction_reward(content, gt, q_type: str = ""):
    content_parts = _normalize_direction_parts(content, q_type=q_type)
    gt_parts = _normalize_direction_parts(gt, q_type=q_type)

    if not content_parts or not gt_parts:
        return 0.0

    return 1.0 if content_parts == gt_parts else 0.0

_ROUTE_TURN_PATTERN = re.compile(r"(left|right|back|around|behind|u[\s-]?turn)", re.IGNORECASE)

def _parse_turn_sequence(text: str) -> list[str]:
    """Parse an ordered sequence of route-planning turns from free text.

    Each turn is normalized to one of {left, right, back}. `turn around`,
    `turn behind`, and `u-turn` are treated as `back`. Order is preserved.
    """
    cleaned = clean_text(text)
    turns: list[str] = []
    for match in _ROUTE_TURN_PATTERN.finditer(cleaned):
        token = match.group(1).lower().replace(" ", "").replace("-", "")
        if token in ("back", "around", "behind", "uturn"):
            turns.append("back")
        elif token == "left":
            turns.append("left")
        elif token == "right":
            turns.append("right")
    return turns

def route_planning_reward(content, gt):
    """Per-turn partial credit: fraction of turns that match the GT sequence."""
    gt_turns = _parse_turn_sequence(gt)
    if not gt_turns:
        return 0.0
    pred_turns = _parse_turn_sequence(content)
    correct = sum(
        1
        for idx, gt_turn in enumerate(gt_turns)
        if idx < len(pred_turns) and pred_turns[idx] == gt_turn
    )
    return float(correct) / float(len(gt_turns))

ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
GENERIC_XML_TAG_PATTERN = re.compile(r"<\s*/?\s*([A-Za-z][A-Za-z0-9_-]*)[^>]*>")

def _extract_answer_tag_matches(text: str) -> list[str]:
    matches = ANSWER_TAG_PATTERN.findall(str(text or ""))
    return [str(match or "").strip() for match in matches]

def _extract_last_answer_tag(text: str) -> Optional[str]:
    matches = _extract_answer_tag_matches(text)
    if not matches:
        return None
    answer = str(matches[-1] or "").strip()
    return answer or None

def _strip_answer_tags(text: str) -> str:
    return ANSWER_TAG_PATTERN.sub(" ", str(text or ""))

def _parse_round2_response(response: Any) -> dict[str, Any]:
    raw_text = str(response or "").strip()
    answer_tag_matches = _extract_answer_tag_matches(raw_text)
    answer_tag_count = len(answer_tag_matches)
    has_exactly_one_answer_tag = answer_tag_count == 1
    has_multiple_answer_tags = answer_tag_count > 1
    has_missing_answer_tag = answer_tag_count == 0
    tagged_answer = answer_tag_matches[-1] if has_exactly_one_answer_tag else None
    reasoning_text = _strip_answer_tags(raw_text).strip()
    xml_tags = [str(tag or "").strip().lower() for tag in GENERIC_XML_TAG_PATTERN.findall(raw_text)]
    non_answer_xml_tags = [tag for tag in xml_tags if tag != "answer"]
    has_other_xml_tags = bool(non_answer_xml_tags)

    if has_multiple_answer_tags or has_other_xml_tags:
        format_score = -1.0
    elif has_missing_answer_tag:
        format_score = 0.0
    else:
        format_score = 1.0

    extracted_answer = tagged_answer if has_exactly_one_answer_tag else ""
    explanation_text = reasoning_text if reasoning_text else raw_text

    return {
        "raw_text": raw_text,
        "format_score": float(format_score),
        "answer_text": str(extracted_answer or "").strip(),
        "explanation_text": str(explanation_text or "").strip(),
        "answer_tag_count": int(answer_tag_count),
        "has_exactly_one_answer_tag": bool(has_exactly_one_answer_tag),
        "has_missing_answer_tag": bool(has_missing_answer_tag),
        "has_multiple_answer_tags": bool(has_multiple_answer_tags),
        "has_other_xml_tags": bool(has_other_xml_tags),
        "non_answer_xml_tags": list(non_answer_xml_tags),
    }

def single_image_relative_direction_reward(content, gt):
    content_parts = _normalize_direction_parts(content, q_type="single_image_relative_direction")
    gt_parts = _normalize_direction_parts(gt, q_type="single_image_relative_direction")

    if not content_parts or not gt_parts:
        return 0.0

    if content_parts == gt_parts:
        return 1.0

    content_set = set(content_parts)
    gt_set = set(gt_parts)

    if len(gt_set) == 3 and len(content_set) == 2 and content_set.issubset(gt_set):
        return 1.0

    if len(gt_set) == 2 and len(content_set) == 1 and content_set.issubset(gt_set):
        return 0.1

    return 0.0

def to_float(text):
    if text is None:
        return None

    normalized_text = str(text).strip()
    match = NUMBER_PATTERN.search(normalized_text.replace(",", ""))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None

    lowered = clean_text(text).replace("-", " ")
    tokens = [token for token in re.findall(r"[a-zA-Z]+", lowered) if token]
    if not tokens:
        return None

    total = 0
    current = 0
    consumed = 0
    for token in tokens:
        if token == "and":
            consumed += 1
            continue
        if token in NUMBER_WORD_UNITS:
            current += NUMBER_WORD_UNITS[token]
            consumed += 1
            continue
        if token in NUMBER_WORD_TENS:
            current += NUMBER_WORD_TENS[token]
            consumed += 1
            continue
        if token == "hundred":
            current = max(current, 1) * NUMBER_WORD_SCALES[token]
            consumed += 1
            continue
        if token == "thousand":
            total += max(current, 1) * NUMBER_WORD_SCALES[token]
            current = 0
            consumed += 1
            continue
        break

    if consumed > 0:
        return float(total + current)
    return None

def abs_dist_norm(pred, target):
    # Normalize absolute error by the target scale.
    return abs(pred - target) / max(target, 1e-9)

def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
    if pred is None or target is None:
        return 0.0

    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, round(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return float(accuracy.mean())

def numeric_error_stats(content, solution_answer):
    pred = to_float(clean_text(content))
    target = to_float(clean_text(solution_answer))
    if pred is None or target is None:
        return None, None, None, None
    abs_err = abs(pred - target)
    rel_err = abs_dist_norm(pred, target)
    return pred, target, abs_err, rel_err

def mca_reward(content, solution_answer):
    content = _normalize_exact_answer(content)
    solution_answer = _normalize_exact_answer(solution_answer)
    return 1.0 if content == solution_answer else 0.0

def count_reward(content, solution_answer):
    pred, target, abs_err, _ = numeric_error_stats(content, solution_answer)
    if pred is None or target is None or abs_err is None:
        return 0.0

    if abs_err <= 1e-6:
        return 1.0
    if abs_err <= 1.0:
        return 0.3
    if abs_err <= 2.0:
        return 0.1
    return 0.0

def metric_distance_reward(content, solution_answer):
    pred, target, _, _ = numeric_error_stats(content, solution_answer)
    if pred is None or target is None:
        return 0.0
    return mean_relative_accuracy(pred, target)

def object_size_reward(content, solution_answer):
    pred, target, _, _ = numeric_error_stats(content, solution_answer)
    if pred is None or target is None:
        return 0.0
    return mean_relative_accuracy(pred, target)

def room_size_reward(content, solution_answer):
    pred, target, _, _ = numeric_error_stats(content, solution_answer)
    if pred is None or target is None:
        return 0.0
    return mean_relative_accuracy(pred, target)

def answer_correctness_score(content, gt, q_type):
    norm_q_type = str(q_type or "").strip()
    gt_text = "" if gt is None else str(gt)

    if "invalid" in gt_text.lower():
        gt_text = ''

    if norm_q_type == "single_image_relative_direction":
        return single_image_relative_direction_reward(content, gt_text)

    if norm_q_type in DIRECTION_MATCH_TASK_TYPES:
        return direction_reward(content, gt_text, q_type=norm_q_type)

    if norm_q_type in COUNT_TASK_TYPES:
        return count_reward(content, gt_text)

    if norm_q_type in METRIC_DISTANCE_TASK_TYPES:
        return metric_distance_reward(content, gt_text)

    if norm_q_type in OBJECT_SIZE_TASK_TYPES:
        return object_size_reward(content, gt_text)

    if norm_q_type in ROOM_SIZE_TASK_TYPES:
        return room_size_reward(content, gt_text)

    if norm_q_type in ROUTE_PLANNING_TASK_TYPES:
        return route_planning_reward(content, gt_text)

    if norm_q_type in EXACT_MATCH_TASK_TYPES or norm_q_type in LEGACY_EXACT_MATCH_TASK_TYPES:
        return mca_reward(
            _normalize_exact_answer(content, q_type=norm_q_type),
            _normalize_exact_answer(gt_text, q_type=norm_q_type),
        )

    return mca_reward(
        _normalize_exact_answer(content, q_type=norm_q_type),
        _normalize_exact_answer(gt_text, q_type=norm_q_type),
    )

def _load_optional_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None

def _resolve_invalid_reason_judge_max_workers(kwargs: dict[str, Any]) -> int:
    configured = kwargs.get(
        "invalid_reason_judge_max_workers",
        os.environ.get("INVALID_REASON_JUDGE_MAX_WORKERS", 16),
    )
    try:
        workers = int(configured)
    except (TypeError, ValueError):
        workers = 16
    return max(1, workers)

def _format_task_reference_for_judge(task_reference: Any) -> str:
    text = str(task_reference or "").strip()
    return text if text else "(No task reference provided.)"

def _render_simulator_reference_for_judge(
    *,
    sim_error_code: Any,
    sim_failure_stage: Any,
    sim_error: Any,
    sim_judge_reference_json: Any,
    sim_parsed_params_json: Any,
    sim_validation_result_json: Any,
    sim_result_json: Any,
) -> str:
    judge_reference = _load_optional_json(sim_judge_reference_json)
    parsed_params = _load_optional_json(sim_parsed_params_json)
    validation_result = _load_optional_json(sim_validation_result_json)

    if isinstance(judge_reference, dict):
        payload = judge_reference
    else:
        payload = {
            "error_code": str(sim_error_code or ""),
            "failure_stage": str(sim_failure_stage or ""),
            "error": str(sim_error or ""),
            "parsed_params": parsed_params or {},
            "validation_result": validation_result or {},
        }

    final_invalid_reason = payload.get("final_invalid_reason") or payload.get("error") or str(sim_error or "")
    error_code = payload.get("error_code") or str(sim_error_code or "")
    failure_stage = payload.get("failure_stage") or str(sim_failure_stage or "")
    expected_fields = payload.get("expected_extraction_fields", [])
    resolved_fields = payload.get("resolved_extraction_fields", [])
    unresolved_fields = payload.get("unresolved_extraction_fields", [])
    payload_parsed_params = payload.get("parsed_params", parsed_params or {})
    validation_issues = payload.get("validation_issues", [])
    validation_suggestions = payload.get("validation_suggestions", [])

    lines = [
        "These simulator signals are authoritative diagnostics from the deterministic validation pipeline.",
        "Judge the explanation by whether it matches the simulator's actual failure reason.",
        "",
        "[Final Verdict]",
        f"- Final invalid reason: {final_invalid_reason}",
        f"- Error code: {error_code}",
        f"- Failure stage: {failure_stage}",
        "",
        "[Grounding Summary]",
        f"- Expected extraction fields: {json.dumps(expected_fields, ensure_ascii=False)}",
        f"- Unresolved extraction fields: {json.dumps(unresolved_fields, ensure_ascii=False)}",
        f"- Resolved extraction fields: {json.dumps(resolved_fields, ensure_ascii=False)}",
        f"- Parsed params: {json.dumps(payload_parsed_params, ensure_ascii=False, sort_keys=True)}",
    ]
    if validation_issues:
        lines.extend([
            "",
            "[Validation Issues]",
            f"- Issues: {json.dumps(validation_issues, ensure_ascii=False)}",
        ])
        if validation_suggestions:
            lines.append(f"- Suggestions: {json.dumps(validation_suggestions, ensure_ascii=False)}")
    elif validation_suggestions:
        lines.extend([
            "",
            "[Validation Issues]",
            f"- Suggestions: {json.dumps(validation_suggestions, ensure_ascii=False)}",
        ])
    return "\n".join(lines)

def _snap_invalid_reason_score(value: float) -> float:
    return min(INVALID_REASON_SCORE_LEVELS, key=lambda candidate: abs(candidate - value))

def judge_invalid_reason_explanation(
    *,
    task_type: str,
    question: str,
    explanation: str,
    task_reference_text: str,
    sim_error_code: Any,
    sim_failure_stage: Any,
    sim_error: Any,
    sim_judge_reference_json: Any,
    sim_parsed_params_json: Any,
    sim_validation_result_json: Any,
    sim_result_json: Any,
    ak="",
    model="gpt-oss-120b-ldm",
    url="https://example.com/v1",
    timeout=30,
) -> float:
    explanation = str(explanation or "").strip()
    if not explanation:
        return 0.0

    simulator_reference = _render_simulator_reference_for_judge(
        sim_error_code=sim_error_code,
        sim_failure_stage=sim_failure_stage,
        sim_error=sim_error,
        sim_judge_reference_json=sim_judge_reference_json,
        sim_parsed_params_json=sim_parsed_params_json,
        sim_validation_result_json=sim_validation_result_json,
        sim_result_json=sim_result_json,
    )
    task_reference = _format_task_reference_for_judge(task_reference_text)

    judge_prompt = f"""You are judging whether a model correctly explained why a spatial reasoning question is invalid.

The simulator diagnostics below are authoritative deterministic ground truth. They come from the real validation pipeline, not from another language model.
Your job is to check whether the model explanation captures the simulator's actual failure reason.
Prefer the simulator's final conclusion and grounded extraction status over fluent but unsupported explanations.

### ASSIGNED TASK TYPE:
{task_type}

### ORIGINAL QUESTION:
{question}

### TASK REFERENCE:
{task_reference}

### SIMULATOR INVALID SIGNALS:
{simulator_reference}

### MODEL EXPLANATION:
{explanation}

### SCORING RUBRIC:
- 1.0: The explanation clearly identifies the simulator's main invalid reason and correctly grounds it in the task rules, grounded object pool, extraction failure, non-unique reference, or validation issue.
- 0.6: The explanation is mostly correct and captures the main failure, but misses an important supporting detail or states it too vaguely.
- 0.3: The explanation catches only part of the problem, or gives a generic reason that weakly overlaps with the simulator diagnostics.
- 0.0: The explanation is wrong, contradicts the simulator diagnostics, or fails to explain why the question is invalid.

Return ONLY one score from this set:
0
0.3
0.6
1.0
"""

    client = OpenAI(api_key=ak, base_url=url)
    messages = [{"role": "user", "content": judge_prompt}]
    max_retries = 5
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.0,
                timeout=timeout,
            )
            content = str(chat_completion.choices[0].message.content or "").strip()
            match = re.search(r"0\.3|0\.6|0(?:\.0+)?|1(?:\.0+)?", content)
            if match:
                return _snap_invalid_reason_score(float(match.group(0)))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Invalid-reason judge failed: {e}")

    return 0.0

def _prm_image_path_to_data_url(image_path: Any, max_pixels: int = 100352) -> Optional[str]:
    """Load an image, downscale it to ~`max_pixels` (matching the Solver's input
    resolution so token cost/latency stay bounded), and return a JPEG data URL.

    ScanNet color frames are ~1.25M px; without capping, a single judge call with
    8 frames would balloon to >12k image tokens. We mirror the training
    min_pixels=max_pixels=100352 budget for consistency with what the Solver saw.
    """
    path = str(image_path or "")
    if not path or not os.path.exists(path):
        return None
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            cur = w * h
            if cur > max_pixels and cur > 0:
                scale = (max_pixels / float(cur)) ** 0.5
                new_w = max(28, int(round(w * scale)))
                new_h = max(28, int(round(h * scale)))
                im = im.resize((new_w, new_h), Image.BICUBIC)
            import io

            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
    except Exception:
        # Fallback: raw bytes (server-side processor will handle resizing).
        try:
            mime_type, _ = mimetypes.guess_type(path)
            with open(path, "rb") as image_file:
                payload = base64.b64encode(image_file.read()).decode("ascii")
            return f"data:{mime_type or 'image/jpeg'};base64,{payload}"
        except Exception:
            return None


_PRM_STEP_MARKER_RE = re.compile(
    r"(?im)^\s*(?:step\s*\d+\s*[:.\)-]|\d+\s*[:.\)]|[-*]\s+|first[,]|second[,]|third[,]|then[,]|next[,]|finally[,])"
)
_PRM_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _segment_reasoning_steps(text: str, max_steps: int = PRM_MAX_STEPS) -> list[str]:
    """Deterministically split the Solver's reasoning into discrete steps.

    Strategy: prefer explicit line / step-marker boundaries; fall back to
    sentence splitting when the reasoning is a single block. Tiny fragments are
    merged forward, and the result is capped at `max_steps` (extra trailing
    segments are merged into the last kept step) so the judge sees a bounded,
    numbered list.
    """
    raw = str(text or "").strip()
    if not raw:
        return []

    # 1) Split on newlines first (most reasoning is line-structured).
    parts = [seg.strip() for seg in re.split(r"[\r\n]+", raw) if seg.strip()]

    # 2) If it is essentially one block, split on sentence boundaries.
    if len(parts) <= 1:
        parts = [seg.strip() for seg in _PRM_SENTENCE_RE.split(raw) if seg.strip()]

    if not parts:
        return []

    # 3) Merge fragments that are too short to be a meaningful step.
    merged: list[str] = []
    for seg in parts:
        if merged and len(seg) < PRM_MIN_STEP_CHARS:
            merged[-1] = f"{merged[-1]} {seg}".strip()
        else:
            merged.append(seg)
    if not merged:
        merged = [raw]

    # 4) Cap the number of steps; fold overflow into the final step.
    if len(merged) > max_steps:
        head = merged[: max_steps - 1]
        tail = " ".join(merged[max_steps - 1 :]).strip()
        merged = head + [tail]
    return merged


def _snap_prm_step_score(value: float) -> float:
    return min(PRM_STEP_SCORE_LEVELS, key=lambda candidate: abs(candidate - float(value)))


def _parse_prm_step_scores(text: str, num_steps: int) -> list[float]:
    """Parse `num_steps` step scores from the judge output.

    Accepts both `Step k: <score>` lines and a bare JSON array. Returns a list
    aligned to `num_steps` (truncated/padded with the parsed mean) or an empty
    list when nothing parseable is found (caller treats that as a judge miss).
    """
    content = str(text or "").strip()
    if not content:
        return []

    # Order matters: decimal forms must precede bare "0"/"1" so "0.3"/"0.6" are
    # not split into "0" + ".3".
    score_token = r"(?:0\.3|0\.6|1\.0+|0\.0+|\.3|\.6|1|0)"

    # Preferred: explicit "Step k: score" lines.
    per_line: dict[int, float] = {}
    for match in re.finditer(rf"(?im)step\s*(\d+)\s*[:.\)-]?\s*({score_token})", content):
        idx = int(match.group(1))
        per_line[idx] = _snap_prm_step_score(float(match.group(2)))
    if per_line:
        ordered = [per_line[k] for k in sorted(per_line)]
    else:
        # Fallback: a JSON array or any run of score tokens.
        nums = re.findall(score_token, content)
        ordered = [_snap_prm_step_score(float(n)) for n in nums]

    if not ordered:
        return []

    if len(ordered) >= num_steps:
        return ordered[:num_steps]
    pad_value = float(np.mean(ordered))
    return ordered + [_snap_prm_step_score(pad_value)] * (num_steps - len(ordered))


def judge_valid_reasoning_steps(
    *,
    task_type: str,
    question: str,
    gt: str,
    answer_text: str,
    reasoning_text: str,
    image_paths: list[str],
    answer_correctness: float,
    ak: str = "",
    model: str = "",
    url: str = "",
    timeout: float = 60.0,
    max_retries: int = 4,
    max_images: int = 8,
) -> tuple[float, list[float]]:
    """Scene-grounded, GT-aware, step-level PRM judge for a VALID answer.

    Returns `(mean_step_score, per_step_scores)`. On any failure the score
    falls back to `answer_correctness` (neutral: blending then leaves the
    reward equal to the pure-ORM value, so a judge outage never zeroes signal).
    """
    steps = _segment_reasoning_steps(reasoning_text)
    if not steps:
        # No reasoning to grade -> neutral (reward == ORM for this sample).
        return float(answer_correctness), []

    numbered = "\n".join(f"Step {i + 1}: {seg}" for i, seg in enumerate(steps))
    num_steps = len(steps)

    judge_prompt = f"""You are grading the step-by-step reasoning a model used to answer a spatial-reasoning question about the scene shown in the images.

You are given the GROUND-TRUTH answer. Judge each reasoning step on whether it is (a) consistent with what is visible in the scene images, and (b) a correct, helpful step toward the ground-truth answer. A fluent step that is visually unsupported or that pushes away from the ground truth must score low, even if the final answer happens to be right.

### TASK TYPE:
{task_type}

### QUESTION:
{question}

### GROUND-TRUTH ANSWER:
{gt}

### MODEL FINAL ANSWER:
{answer_text}

### MODEL REASONING STEPS:
{numbered}

### PER-STEP SCORING RUBRIC:
- 1.0: Correct, scene-grounded, and clearly advances toward the ground-truth answer.
- 0.6: Mostly correct/grounded but with a minor gap, vagueness, or weak support.
- 0.3: Partially relevant but contains a notable error or is poorly grounded.
- 0.0: Wrong, hallucinated/ungrounded, or contradicts the scene or ground truth.

Return EXACTLY one line per step in this format, and nothing else:
Step 1: <score>
Step 2: <score>
...
Each <score> must be one of: 0, 0.3, 0.6, 1.0
"""

    content: list[dict[str, Any]] = [{"type": "text", "text": judge_prompt}]
    for image_path in (image_paths or [])[: max(1, int(max_images))]:
        data_url = _prm_image_path_to_data_url(image_path)
        if data_url:
            content.append({"type": "image_url", "image_url": {"url": data_url}})

    client = OpenAI(api_key=ak or "EMPTY", base_url=url)
    messages = [{"role": "user", "content": content}]
    for attempt in range(max(1, max_retries)):
        try:
            completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.0,
                max_tokens=256,
                timeout=timeout,
            )
            raw = str(completion.choices[0].message.content or "").strip()
            step_scores = _parse_prm_step_scores(raw, num_steps)
            if step_scores:
                return float(np.mean(step_scores)), step_scores
        except Exception as exc:
            if attempt + 1 >= max(1, max_retries):
                print(f"Valid-reasoning PRM judge failed: {exc}")
            else:
                time.sleep(min(30.0, 2.0 * (attempt + 1)))

    # Parse/judge miss -> neutral fallback.
    return float(answer_correctness), []


def _prm_enabled(kwargs: dict[str, Any]) -> bool:
    value = kwargs.get("prm_enable", os.environ.get("SPATIALEVO_PRM_ENABLE", "0"))
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def compute_score(
    reward_inputs: list[RewardInput],
    **kwargs
) -> list[RewardScore]:
    """Compute Score."""
    judge_url = kwargs.get("invalid_reason_judge_url", kwargs.get("service_url", kwargs.get("url", "https://example.com/v1")))
    judge_timeout = kwargs.get("invalid_reason_judge_timeout", kwargs.get("timeout", 30))
    judge_model = kwargs.get("invalid_reason_judge_model", kwargs.get("model", "gpt-oss-120b-ldm"))
    judge_api_key = kwargs.get("invalid_reason_judge_api_key", kwargs.get("api_key", ""))
    judge_max_workers = _resolve_invalid_reason_judge_max_workers(kwargs)

    # --- Split-role PRM config (scene-grounded step-level reasoning judge) ---
    prm_enable = _prm_enabled(kwargs)
    prm_judge_url = str(kwargs.get("prm_judge_url", os.environ.get("SPATIALEVO_PRM_JUDGE_URL", "")) or "").strip()
    prm_judge_model = str(kwargs.get("prm_judge_model", os.environ.get("SPATIALEVO_PRM_JUDGE_MODEL", "")) or "").strip()
    prm_judge_api_key = str(kwargs.get("prm_judge_api_key", os.environ.get("SPATIALEVO_PRM_JUDGE_API_KEY", "EMPTY")) or "EMPTY")
    prm_judge_timeout = float(kwargs.get("prm_judge_timeout", os.environ.get("SPATIALEVO_PRM_JUDGE_TIMEOUT", 60)))
    prm_max_images = int(kwargs.get("prm_max_images", os.environ.get("SPATIALEVO_PRM_MAX_IMAGES", 8)))
    prm_answer_weight = float(kwargs.get("prm_answer_weight", os.environ.get("SPATIALEVO_PRM_ANSWER_WEIGHT", PRM_DEFAULT_ANSWER_WEIGHT)))
    prm_reasoning_weight = float(kwargs.get("prm_reasoning_weight", os.environ.get("SPATIALEVO_PRM_REASONING_WEIGHT", PRM_DEFAULT_REASONING_WEIGHT)))
    try:
        prm_max_workers = int(kwargs.get("prm_max_workers", os.environ.get("SPATIALEVO_PRM_MAX_WORKERS", 8)))
    except (TypeError, ValueError):
        prm_max_workers = 8
    prm_max_workers = max(1, prm_max_workers)
    # PRM requires a local judge endpoint; silently degrade to ORM if unset.
    prm_active = prm_enable and bool(prm_judge_url and prm_judge_model)

    sample_details = []
    invalid_jobs = []
    prm_jobs = []

    for idx, reward_input in enumerate(reward_inputs):
        response = reward_input.get("response", "")
        uid = reward_input.get("uid", f"unknown_{idx}")
        gt = reward_input.get("gt", "")
        q_type = reward_input.get("question_type", "")
        is_valid = bool(reward_input.get("is_valid", True))
        round2_mode = str(reward_input.get("round2_mode", "") or "")
        parsed_response = _parse_round2_response(response)
        response_format_score = float(parsed_response["format_score"])
        has_multiple_answer_tags = bool(parsed_response.get("has_multiple_answer_tags", False))
        has_missing_answer_tag = bool(parsed_response.get("has_missing_answer_tag", False))
        has_other_xml_tags = bool(parsed_response.get("has_other_xml_tags", False))
        valid_answer_hard_format_fail = bool(
            is_valid and (has_multiple_answer_tags or has_other_xml_tags)
        )
        if not is_valid:
            response_format_score = -1.0 if has_other_xml_tags else 0.0

        if is_valid:
            content = parsed_response["answer_text"]
            r_acc = answer_correctness_score(content, gt, q_type)
            mm_data = reward_input.get("multi_modal_data")
            image_paths: list[str] = []
            if isinstance(mm_data, dict):
                raw_images = mm_data.get("images")
                if isinstance(raw_images, (list, tuple)):
                    image_paths = [str(p) for p in raw_images]
                elif raw_images:
                    image_paths = [str(raw_images)]
            sample_details.append({
                "uid": uid,
                "is_valid": True,
                "content": content,
                "r_acc": float(r_acc),
                "format_score": response_format_score,
                "invalid_reason_score": 0.0,
                "round2_mode": round2_mode or "answer",
                "has_multiple_answer_tags": has_multiple_answer_tags,
                "has_missing_answer_tag": has_missing_answer_tag,
                "has_other_xml_tags": has_other_xml_tags,
                "valid_answer_hard_format_fail": valid_answer_hard_format_fail,
                # PRM: neutral defaults (reasoning == answer correctness -> blend is a no-op).
                "prm_reasoning_score": float(r_acc),
                "prm_step_scores": [],
                "prm_applied": 0.0,
                "prm_payload": {
                    "task_type": str(q_type or ""),
                    "question": str(reward_input.get("question", "") or ""),
                    "gt": str(gt or ""),
                    "answer_text": str(content or ""),
                    "reasoning_text": str(parsed_response.get("explanation_text", "") or ""),
                    "image_paths": image_paths,
                    "answer_correctness": float(r_acc),
                },
            })
            # Only worth judging when PRM is active, the answer is well-formed, and
            # there is actual reasoning text to grade.
            if (
                prm_active
                and not valid_answer_hard_format_fail
                and sample_details[-1]["prm_payload"]["reasoning_text"].strip()
            ):
                prm_jobs.append(len(sample_details) - 1)
            continue

        explanation = parsed_response["raw_text"]
        sample_details.append({
            "uid": uid,
            "is_valid": False,
            "content": explanation,
            "r_acc": 0.0,
            "format_score": response_format_score,
            "invalid_reason_score": 0.0,
            "round2_mode": round2_mode or "invalid_explanation",
            "has_multiple_answer_tags": has_multiple_answer_tags,
            "has_missing_answer_tag": has_missing_answer_tag,
            "has_other_xml_tags": has_other_xml_tags,
            "valid_answer_hard_format_fail": False,
            "judge_payload": {
                "task_type": str(q_type or ""),
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
        })
        if explanation and not has_multiple_answer_tags:
            invalid_jobs.append(len(sample_details) - 1)

    if invalid_jobs:
        if judge_max_workers > 1 and len(invalid_jobs) > 1:
            max_workers = min(judge_max_workers, len(invalid_jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        judge_invalid_reason_explanation,
                        **sample_details[idx]["judge_payload"],
                        ak=judge_api_key,
                        model=judge_model,
                        url=judge_url,
                        timeout=judge_timeout,
                    ): idx
                    for idx in invalid_jobs
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        sample_details[idx]["invalid_reason_score"] = float(future.result())
                    except Exception as exc:
                        print(f"Invalid-reason reward parallel call failed (idx={idx}): {exc}")
                        sample_details[idx]["invalid_reason_score"] = 0.0
        else:
            for idx in invalid_jobs:
                try:
                    sample_details[idx]["invalid_reason_score"] = float(
                        judge_invalid_reason_explanation(
                            **sample_details[idx]["judge_payload"],
                            ak=judge_api_key,
                            model=judge_model,
                            url=judge_url,
                            timeout=judge_timeout,
                        )
                    )
                except Exception as exc:
                    print(f"Invalid-reason reward failed (idx={idx}): {exc}")
                    sample_details[idx]["invalid_reason_score"] = 0.0

    # --- Split-role PRM: scene-grounded step-level reasoning judge (valid only) ---
    if prm_active and prm_jobs:
        def _run_prm(idx: int) -> tuple[int, float, list[float]]:
            payload = sample_details[idx]["prm_payload"]
            mean_score, step_scores = judge_valid_reasoning_steps(
                task_type=payload["task_type"],
                question=payload["question"],
                gt=payload["gt"],
                answer_text=payload["answer_text"],
                reasoning_text=payload["reasoning_text"],
                image_paths=payload["image_paths"],
                answer_correctness=payload["answer_correctness"],
                ak=prm_judge_api_key,
                model=prm_judge_model,
                url=prm_judge_url,
                timeout=prm_judge_timeout,
                max_images=prm_max_images,
            )
            return idx, mean_score, step_scores

        workers = min(prm_max_workers, len(prm_jobs))
        if workers > 1 and len(prm_jobs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_prm, idx): idx for idx in prm_jobs}
                for future in as_completed(futures):
                    fidx = futures[future]
                    try:
                        idx, mean_score, step_scores = future.result()
                        sample_details[idx]["prm_reasoning_score"] = float(mean_score)
                        sample_details[idx]["prm_step_scores"] = list(step_scores)
                        sample_details[idx]["prm_applied"] = 1.0
                    except Exception as exc:
                        print(f"Valid-reasoning PRM parallel call failed (idx={fidx}): {exc}")
        else:
            for idx in prm_jobs:
                try:
                    _, mean_score, step_scores = _run_prm(idx)
                    sample_details[idx]["prm_reasoning_score"] = float(mean_score)
                    sample_details[idx]["prm_step_scores"] = list(step_scores)
                    sample_details[idx]["prm_applied"] = 1.0
                except Exception as exc:
                    print(f"Valid-reasoning PRM failed (idx={idx}): {exc}")

    scores = []
    for detail in sample_details:
        prm_reasoning_score = float(detail.get("prm_reasoning_score", detail.get("r_acc", 0.0)))
        prm_step_scores = list(detail.get("prm_step_scores", []) or [])
        prm_applied = float(detail.get("prm_applied", 0.0))

        if bool(detail.get("valid_answer_hard_format_fail", False)):
            scores.append({
                "overall": -1.0,
                "format": -1.0,
                "accuracy": 0.0,
                "answer_correctness": 0.0,
                "invalid_reason_score": 0.0,
                "valid_case": 1.0 if detail["is_valid"] else 0.0,
                "invalid_case": 0.0 if detail["is_valid"] else 1.0,
                "format_hard_penalty": 1.0,
                "prm_reasoning_score": 0.0,
                "prm_num_steps": 0.0,
                "prm_applied": 0.0,
                "prm_step_scores": [],
            })
            continue

        r_acc = float(detail["r_acc"])
        format_score = float(detail.get("format_score", 0.0))
        invalid_reason_score = float(detail["invalid_reason_score"])
        if detail["is_valid"]:
            # PRM blend: content = w_ans*answer_correctness + w_reason*reasoning.
            # When PRM is inactive / not applied, prm_reasoning_score == r_acc, so
            # the blend collapses back to the original ORM value.
            if prm_active and prm_applied >= 1.0:
                content_reward = (prm_answer_weight * r_acc) + (prm_reasoning_weight * prm_reasoning_score)
            else:
                content_reward = r_acc
        else:
            content_reward = invalid_reason_score
        overall_reward = (ROUND2_FORMAT_WEIGHT * format_score) + ((1.0 - ROUND2_FORMAT_WEIGHT) * content_reward)
        scores.append({
            "overall": overall_reward,
            "format": format_score,
            "accuracy": r_acc,
            "answer_correctness": r_acc,
            "invalid_reason_score": invalid_reason_score,
            "valid_case": 1.0 if detail["is_valid"] else 0.0,
            "invalid_case": 0.0 if detail["is_valid"] else 1.0,
            "format_hard_penalty": 0.0,
            "prm_reasoning_score": prm_reasoning_score if detail["is_valid"] else 0.0,
            "prm_num_steps": float(len(prm_step_scores)),
            "prm_applied": prm_applied if detail["is_valid"] else 0.0,
            "prm_step_scores": prm_step_scores if detail["is_valid"] else [],
        })

    try:
        uid2contents = defaultdict(list)
        for d in sample_details:
            uid2contents[d["uid"]].append(d["content"])
        total_groups = len(uid2contents)
        uniq_ge2 = sum(1 for v in uid2contents.values() if len(set(v)) >= 2)
        uniq_ge3 = sum(1 for v in uid2contents.values() if len(set(v)) >= 3)
        diversity_metrics = {
            "diversity/num_groups": float(total_groups),
            "diversity/uniq_ge2_ratio": (float(uniq_ge2) / float(total_groups)) if total_groups > 0 else 0.0,
            "diversity/uniq_ge3_ratio": (float(uniq_ge3) / float(total_groups)) if total_groups > 0 else 0.0,
        }
        # The trainer reduces these extra metrics across workers.
        if scores:
            scores[0].update(diversity_metrics)
    except Exception:
        pass
    
    return scores
