
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
import itertools
import hashlib
import math
import mimetypes
import os
import random
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Optional, Type
from openai import OpenAI
import numpy as np
import ray
import torch
from jinja2 import Template
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
import re

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.dataset import process_image
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulator.task_support import NON_ANCHOR_LABELS
from src.simulator.world_simulator import WorldSimulator
from src.simulator.structured_question_renderer import (
    TASK_CONFIG as DGE_TASK_CONFIG,
    render_question as render_dge_structured_question,
)
from src.maze_engine import MazeEngine

from ..utils.agentic import (
    VILASR_PROMPT_TEMPLATE,
    VILASR_SYSTEM_PROMPT,
    VILASR_TYPE_TEMPLATE,
    process_image_agentic,
)

DATASET_SPLIT_DIR_NAMES = {"train", "val", "validation", "test"}
# Deterministic maze verifier task (not a DGE task; validated by MazeEngine).
MAZE_TASK_TYPE = "maze_navigation"
# Scene tasks whose questions reference concrete objects we can ground against
# per-frame GT bboxes (VSI grounding process reward). Mirrors VSI_GROUNDING_TASKS
# in agentic_solver_reward.py.
VSI_GROUNDING_TASK_TYPES = {
    "object_counting",
    "absolute_distance",
    "object_size",
    "relative_distance",
    "relative_direction_hard",
    "route_planning",
}
PROMPT_NON_ANCHOR_LABELS = {str(label).strip().lower() for label in NON_ANCHOR_LABELS}
TASK_ANSWER_HINTS_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "task_answer_format_hints.json"
)
ROUND2_ANSWER_PROMPT_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "answer_question_round2.jinja"
)
ROUND2_INVALID_EXPLANATION_PROMPT_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "answer_invalid_question_round2.jinja"
)
ROUND1_SCENE_TASK_PROMPT_TEMPLATE_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_scene_task_conditioned.jinja"
)
ROUND1_SINGLE_IMAGE_TASK_PROMPT_TEMPLATE_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_single_image_task_conditioned.jinja"
)
ROUND1_IMAGE_PAIR_TASK_PROMPT_TEMPLATE_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_image_pair_task_conditioned.jinja"
)
ROUND1_TASK_PROMPT_SPECS_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_task_conditioned_specs.json"
)
ROUND1_MAZE_TASK_PROMPT_TEMPLATE_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_maze_task.jinja"
)
ROUND1_MAZE_PICK_TASK_PROMPT_TEMPLATE_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_maze_pick.jinja"
)
# "action_sequence" (default) = Questioner writes the full action sequence.
# "pick_letter" = Questioner only picks A/B/C/D; system fills the verified path.
# "round_robin" = system assigns A/B/C/D in round-robin per context (max diversity,
#                no mode collapse); Questioner output is ignored for letter selection.
MAZE_PROPOSER_MODE = os.environ.get("MAZE_PROPOSER_MODE", "action_sequence").strip().lower()

# Phase-2 structured_imitate proposer mode for DGE scene/image tasks.
# When "structured_imitate", the Proposer outputs <question_params>{...}</question_params>
# (object-label picks only; task_type is pinned by the round-robin schedule), and the
# system renders the final GRPO-style question text from those params via
# structured_question_renderer. r_Q = 0.1*fmt + 0.9*valid (no curriculum reward —
# deliberately removed because curriculum rewards historically caused collapse).
# Maze tasks are unaffected (they keep MAZE_PROPOSER_MODE).
PROPOSER_MODE = os.environ.get("PROPOSER_MODE", "free_text").strip().lower()
ROUND1_DGE_STRUCTURED_PROMPT_TEMPLATE_PATH = os.path.join(
    project_root, "easy_r1", "training", "format_prompt", "question_generate_dge_structured.jinja"
)

ROUND1_TASKS_BY_MODALITY = {
    "scene": [
        "object_counting",
        "absolute_distance",
        "object_size",
        "room_size",
        "relative_distance",
        "relative_direction_hard",
        "route_planning",
    ],
    "single_image": [
        "single_image_relative_direction",
        "distance_cam_obj",
        "depth_order_obj_obj",
    ],
    "image_pair": [
        "position_cam_cam",
        "elevation_cam_cam",
        "visibility_compare",
        "position_cam_obj",
        "position_cam_reg",
        "motion_camera",
        "attribute_measurement",
    ],
}

# VERSE note: reverse mapping task_type -> required prompt_mode for the round-robin
# scheduler. SPAR single-frame tasks live on "single_image"; the _mv variants and
# position_matching / view_change / camera_motion need two frames ("image_pair").
def _build_task_type_to_prompt_mode() -> dict[str, str]:
    mapping: dict[str, str] = {MAZE_TASK_TYPE: "maze"}
    for mode, types in ROUND1_TASKS_BY_MODALITY.items():
        for t in types:
            mapping[t] = mode
    spar_single = {
        "spar_depth_prediction_oc", "spar_depth_prediction_oo",
        "spar_distance_prediction_oc", "spar_distance_prediction_oo",
        "spar_distance_infer_center_oo", "spar_obj_spatial_relation_oo",
    }
    spar_pair = {
        "spar_depth_prediction_oc_mv", "spar_depth_prediction_oo_mv",
        "spar_distance_prediction_oc_mv", "spar_distance_prediction_oo_mv",
        "spar_distance_infer_center_oo_mv",
        "spar_obj_spatial_relation_oc_mv", "spar_obj_spatial_relation_oo_mv",
        "spar_position_matching", "spar_view_change_infer",
        "spar_camera_motion_infer",
    }
    for t in spar_single:
        mapping[t] = "single_image"
    for t in spar_pair:
        mapping[t] = "image_pair"
    return mapping


TASK_TYPE_TO_PROMPT_MODE = _build_task_type_to_prompt_mode()
ROUND1_TASK_SAMPLING_PRIOR_ACC = 0.35
ROUND1_TASK_SAMPLING_PRIOR_COUNT = 2.0
ROUND1_TASK_SAMPLING_MIN_WEIGHT = 0.05
ROUND1_TASK_SCHEDULER_ACC_TARGETS = {
    "absolute_distance": 0.50,
    "distance_cam_obj": 0.50,
    "room_size": 0.50,
}

def _load_required_text_file(path: str, label: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to load required {label} from `{path}`: {exc}") from exc

    if not text:
        raise RuntimeError(f"Required {label} file is empty: `{path}`")
    return text

def _load_required_json_file(path: str, label: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load required {label} from `{path}`: {exc}") from exc

TASK_ANSWER_HINTS = _load_required_json_file(
    TASK_ANSWER_HINTS_PATH,
    "task answer hints",
).get("tasks") or {}

TASK_ANSWER_HINT_TASKS = set(TASK_ANSWER_HINTS.keys())
TASK_ANSWER_FORMAT_SUFFIXES = {
    task_name: str(task_config.get("format_prompt_suffix", "") or "")
    for task_name, task_config in TASK_ANSWER_HINTS.items()
}
ROUND2_DIRECTION_EXAMPLE_COUNT = 4
ROUND2_DYNAMIC_DIRECTION_HINT_TASKS = {
    "relative_direction_hard",
    "single_image_relative_direction",
    "position_cam_cam",
    "position_cam_obj",
    "position_cam_reg",
    "motion_camera",
}
ROUND1_GENERIC_DIRECTION_SUFFIXES = {
    "relative_direction_hard": " A concise answer is usually a short direction phrase.",
    "single_image_relative_direction": " A concise answer is usually a short relative direction phrase.",
    "position_cam_cam": " A concise answer is usually a short direction phrase.",
    "position_cam_obj": " A concise answer is usually a short direction phrase.",
    "position_cam_reg": " A concise answer is usually a short direction phrase.",
    "motion_camera": " A concise answer is usually a short camera-motion label.",
}
ROUND2_DIRECTION_LABEL_POOLS = {
    "relative_direction_hard": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
    ],
    "position_cam_cam": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
    ],
    "position_cam_obj": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
    ],
    "position_cam_reg": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
    ],
    "motion_camera": [
        "left", "right", "front", "back",
        "up", "down",
        "left-front", "right-front", "left-back", "right-back",
        "left-up", "right-up", "front-up", "back-up",
        "left-down", "right-down", "front-down", "back-down",
    ],
    "single_image_relative_direction": [
        "left", "right", "front", "back",
        "front-left", "front-right", "back-left", "back-right",
        "up", "down",
        "up-front", "up-back", "up-left", "up-right",
        "down-front", "down-back", "down-left", "down-right",
    ],
}
DYNAMIC_HINT_STRIP_PATTERNS = (
    re.compile(r"\s*A concise answer is usually one of:\s*[^.]+\.?\s*$", re.IGNORECASE),
    re.compile(r"\s*Answer with exactly one of:\s*[^.]+\.?\s*$", re.IGNORECASE),
)
ROUND_DYNAMIC_FIXED_CHOICE_POOLS = {
    "visibility_compare": ["image1", "image2", "same", "neither"],
    "elevation_cam_cam": ["higher", "lower"],
}

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

def _image_path_to_data_url(image_path: Any, max_pixels: int = 100352) -> str:
    """Encode an image as a data URL, downscaled to ~`max_pixels`.

    The frozen-Proposer local vLLM server is fed via this helper. ScanNet color
    frames are ~1.25M px; we cap them to the training min/max_pixels budget
    (100352) so the Proposer sees the same resolution the in-loop actor would and
    the request stays well within the server's max_model_len.
    """
    path = str(image_path or "")
    try:
        from PIL import Image
        import io

        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            cur = w * h
            if cur > max_pixels and cur > 0:
                scale = (max_pixels / float(cur)) ** 0.5
                im = im.resize((max(28, int(round(w * scale))), max(28, int(round(h * scale)))), Image.BICUBIC)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
    except Exception:
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "image/jpeg"
        with open(path, "rb") as image_file:
            payload = base64.b64encode(image_file.read()).decode("ascii")
        return f"data:{mime_type};base64,{payload}"

def _openai_message_text(completion: Any) -> str:
    try:
        content = completion.choices[0].message.content
    except Exception:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict):
                pieces.append(str(item.get("text", "") or ""))
            else:
                pieces.append(str(getattr(item, "text", "") or ""))
        return "\n".join(piece for piece in pieces if piece).strip()
    return str(content or "").strip()

def _normalize_suffix_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())

def _strip_known_answer_suffix(question: str) -> str:
    base = str(question or "").rstrip()
    candidate_suffixes = set(TASK_ANSWER_FORMAT_SUFFIXES.values()) | set(ROUND1_GENERIC_DIRECTION_SUFFIXES.values())
    normalized_base = _normalize_suffix_spacing(base)
    for suffix in sorted(candidate_suffixes, key=len, reverse=True):
        normalized_suffix = _normalize_suffix_spacing(suffix)
        if not normalized_suffix:
            continue
        if normalized_base.endswith(normalized_suffix):
            suffix_index = base.rfind(suffix.strip())
            if suffix_index >= 0:
                base = base[:suffix_index].rstrip()
                break
    for pattern in DYNAMIC_HINT_STRIP_PATTERNS:
        base = pattern.sub("", base).rstrip()
    return base

def _normalize_direction_label_for_hint(label: Any) -> str:
    text = str(label or "").strip().lower()
    text = text.replace("_", "-")
    text = re.sub(r"\s+", "-", text)
    return text

def _build_round2_direction_hint(question: str, task_type: str, gt: str) -> str:
    if task_type not in ROUND2_DYNAMIC_DIRECTION_HINT_TASKS:
        return ""

    normalized_gt = _normalize_direction_label_for_hint(gt)
    if not normalized_gt or normalized_gt == "invalid":
        return ""

    label_pool = list(ROUND2_DIRECTION_LABEL_POOLS.get(task_type, []))
    if normalized_gt not in label_pool:
        label_pool.append(normalized_gt)

    distractors = [label for label in label_pool if label != normalized_gt]
    seed_source = f"{task_type}||{question}||{normalized_gt}"
    seed = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(distractors)

    examples = [normalized_gt] + distractors[: max(0, ROUND2_DIRECTION_EXAMPLE_COUNT - 1)]
    rng.shuffle(examples)

    if task_type == "motion_camera":
        prefix = " A concise answer is usually a short camera-motion label, for example: "
    elif task_type == "single_image_relative_direction":
        prefix = " A concise answer is usually a short relative direction phrase, for example: "
    else:
        prefix = " A concise answer is usually a short direction phrase, for example: "
    return prefix + ", ".join(examples[:ROUND2_DIRECTION_EXAMPLE_COUNT]) + "."

def _seeded_shuffle_labels(task_type: str, question: str, labels: list[str]) -> list[str]:
    seed_source = f"{task_type}||{question}||{'|'.join(labels)}"
    seed = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return shuffled

def _load_optional_json_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None

def _normalize_signature_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"null", "none"}:
        return ""
    text = re.sub(r"\bimage\s+one\b", "image1", text)
    text = re.sub(r"\bimage\s+two\b", "image2", text)
    text = re.sub(r"\bimg\s*1\b", "image1", text)
    text = re.sub(r"\bimg\s*2\b", "image2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _normalize_image_ref_token(value: Any) -> str:
    text = _normalize_signature_text(value)
    if text in {"image1", "1", "one"}:
        return "image1"
    if text in {"image2", "2", "two"}:
        return "image2"
    return text

def _extract_ordered_image_refs(question: Any) -> list[str]:
    text = str(question or "")
    matches = re.finditer(r"\bimage(?:\s+|_)?(1|2|one|two)\b", text, flags=re.IGNORECASE)
    refs: list[str] = []
    for match in matches:
        ref = _normalize_image_ref_token(match.group(1))
        if ref:
            refs.append(ref)
    return refs

def _extract_depth_order_answer_candidates_from_sim(
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> tuple[str, str] | None:
    key_pairs = (
        ("object1_label", "object2_label"),
        ("object_a", "object_b"),
    )

    def _extract_from_mapping(mapping: Any) -> tuple[str, str] | None:
        if not isinstance(mapping, dict):
            return None

        for left_key, right_key in key_pairs:
            left = str(mapping.get(left_key, "") or "").strip()
            right = str(mapping.get(right_key, "") or "").strip()
            if left and right and left.lower() != "null" and right.lower() != "null":
                return left, right

        for nested_key in ("parsed_params", "context", "result", "validation_result"):
            nested = mapping.get(nested_key)
            found = _extract_from_mapping(nested)
            if found is not None:
                return found
        return None

    for payload in (sim_parsed_params, sim_validation_result, sim_result):
        found = _extract_from_mapping(_load_optional_json_payload(payload))
        if found is not None:
            return found
    return None

def _extract_relative_distance_candidates_from_sim(
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> Optional[list[str]]:
    """Extract the sanitized candidate_labels list for relative_distance."""

    def _extract_from_mapping(mapping: Any) -> Optional[list[str]]:
        if not isinstance(mapping, dict):
            return None
        cands = mapping.get("candidate_labels")
        if isinstance(cands, list):
            vals = [
                str(c).strip()
                for c in cands
                if str(c).strip() and str(c).strip().lower() != "null"
            ]
            if len(vals) >= 2:
                return vals
        for nested_key in ("parsed_params", "context", "result", "validation_result"):
            found = _extract_from_mapping(mapping.get(nested_key))
            if found is not None:
                return found
        return None

    for payload in (sim_parsed_params, sim_validation_result, sim_result):
        found = _extract_from_mapping(_load_optional_json_payload(payload))
        if found is not None:
            return found
    return None


def _extract_relative_direction_labels_from_sim(
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> Optional[tuple[str, str, str]]:
    """Extract (positioning, orienting, querying) labels for relative_direction_hard."""

    def _extract_from_mapping(mapping: Any) -> Optional[tuple[str, str, str]]:
        if not isinstance(mapping, dict):
            return None
        pos = str(mapping.get("positioning_label", "") or "").strip()
        ori = str(mapping.get("orienting_label", "") or "").strip()
        qry = str(mapping.get("querying_label", "") or "").strip()
        if (
            pos and ori and qry
            and pos.lower() != "null" and ori.lower() != "null" and qry.lower() != "null"
        ):
            return pos, ori, qry
        for nested_key in ("parsed_params", "context", "result", "validation_result"):
            found = _extract_from_mapping(mapping.get(nested_key))
            if found is not None:
                return found
        return None

    for payload in (sim_parsed_params, sim_validation_result, sim_result):
        found = _extract_from_mapping(_load_optional_json_payload(payload))
        if found is not None:
            return found
    return None


# VERSE note: VSI-Bench object_rel_direction question texts, verbatim.
_VSI_REL_DIRECTION_HARD_TEXT = (
    "If I am standing by the {pos} and facing the {ori}, is the {qry} to my "
    "front-left, front-right, back-left, or back-right?\n"
    "The directions refer to the quadrants of a Cartesian plane "
    "(if I am standing at the origin and facing along the positive y-axis)."
)
_VSI_REL_DIRECTION_MEDIUM_TEXT = (
    "If I am standing by the {pos} and facing the {ori}, is the {qry} to my "
    "left, right, or back?\n"
    "An object is to my back if I would have to turn at least 135 degrees in order to face it."
)
_VSI_REL_DIRECTION_QUADRANTS = ("front-left", "front-right", "back-left", "back-right")
_VSI_REL_DIRECTION_MEDIUM_SET = ("left", "right", "back")


def _build_relative_direction_bench_question(
    gt: str,
    *,
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> Optional[tuple[str, list[str], str]]:
    """Render VSI-Bench-style relative-direction question + options from DGE GT.

    The DGE computes an 8-bin direction. The 4 diagonal bins lie strictly inside
    the corresponding Cartesian quadrant, so they map 1:1 onto the bench "hard"
    variant; left/right/back map 1:1 onto the bench "medium" variant (their
    45-degree wedges are unambiguous under the 135-degree back rule). GT="front"
    has no bench-style equivalent, so the caller keeps the generic 8-way form.

    Returns (question_text, options, gt_letter) or None.
    """
    gt_norm = _normalize_direction_label_for_hint(gt)
    labels = _extract_relative_direction_labels_from_sim(
        sim_parsed_params=sim_parsed_params,
        sim_validation_result=sim_validation_result,
        sim_result=sim_result,
    )
    if labels is None:
        return None
    pos, ori, qry = labels

    if gt_norm in _VSI_REL_DIRECTION_QUADRANTS:
        text = _VSI_REL_DIRECTION_HARD_TEXT.format(pos=pos, ori=ori, qry=qry)
        pool = list(_VSI_REL_DIRECTION_QUADRANTS)
    elif gt_norm in _VSI_REL_DIRECTION_MEDIUM_SET:
        text = _VSI_REL_DIRECTION_MEDIUM_TEXT.format(pos=pos, ori=ori, qry=qry)
        pool = list(_VSI_REL_DIRECTION_MEDIUM_SET)
    else:
        return None

    options = _seeded_shuffle_labels("relative_direction_hard", text, pool)
    gt_letter = chr(ord("A") + options.index(gt_norm))
    return text, options, gt_letter


# ---------------------------------------------------------------------------
# VERSE note: SPAR-Bench-aligned Round-2 rendering.
#
# SPAR-Bench questions reference colored visual markers drawn on the image
# ("towel (red point)", "books (red bbox)") and carry fixed instruction
# sentences. The Round-1 text (used only for DGE validation + proposer reward)
# stays simple; at Round-2 we rebuild the question in bench style, and the
# trainer draws the matching markers onto the Solver's images.
#
# Text conventions mirror what the model sees at EVAL time (infer.py strips
# "Answer using a single number and nothing else." and "Your answer can only
# include one of options A, B, C or D.", then appends the ViLaSR type suffix;
# the "A, B." variant for distance_infer_center is NOT stripped).
# ---------------------------------------------------------------------------

_SPAR_CALC_NOTE = "Calculate or judge based on the 3D center points of these objects."
_SPAR_MV_VIEW_NOTE = "We align the first image with the observer's perspective, making it the main viewpoint."
_SPAR_DEPTH_MV_NOTE = "The depth is calculated based on the image where the markers corresponding to these objects are located."

_SPAR_RELATION_OPTIONS_DESC_OO = (
    "The options describe the spatial relationship between two objects in terms of "
    "left-right (left, right, or empty if indistinguishable), above-below (above, below, "
    "or empty if indistinguishable), and farther-closer (farther, closer, or empty if indistinguishable)."
)
_SPAR_RELATION_OPTIONS_DESC_OC = (
    "The options describe the spatial relationship between object and observer in terms of "
    "left-right (left, right, or empty if indistinguishable), above-below (above, below, "
    "or empty if indistinguishable), and front-behind (front, behind, or empty if indistinguishable)."
)

_SPAR_NUMERICAL_TYPES = {
    "spar_depth_prediction_oc", "spar_depth_prediction_oc_mv",
    "spar_depth_prediction_oo", "spar_depth_prediction_oo_mv",
    "spar_distance_prediction_oc", "spar_distance_prediction_oc_mv",
    "spar_distance_prediction_oo", "spar_distance_prediction_oo_mv",
}
_SPAR_RELATION_TYPES = {
    "spar_obj_spatial_relation_oo", "spar_obj_spatial_relation_oo_mv",
    "spar_obj_spatial_relation_oc_mv",
}
_SPAR_INFER_TYPES = {"spar_distance_infer_center_oo", "spar_distance_infer_center_oo_mv"}
SPAR_ROUND2_TYPES = _SPAR_NUMERICAL_TYPES | _SPAR_RELATION_TYPES | _SPAR_INFER_TYPES


def _spar_payload_get(payloads: list[Any], key: str) -> Any:
    """Fetch `key` from the first payload mapping that has a non-empty value."""
    for payload in payloads:
        obj = _load_optional_json_payload(payload)
        if isinstance(obj, dict):
            v = obj.get(key)
            if v not in (None, "", [], {}):
                return v
            for nested_key in ("parsed_params", "tool_result_detail", "context", "result"):
                nested = obj.get(nested_key)
                if isinstance(nested, dict):
                    nv = nested.get(key)
                    if nv not in (None, "", [], {}):
                        return nv
    return None


def _spar_relation_distractors(gt_triple_str: str, axes3: tuple[str, str], seed_src: str) -> list[str]:
    """3 distinct distractor triples (bench format 'x, y, z', blanks kept)."""
    combos = []
    for lr in ("left", "right", ""):
        for ab in ("above", "below", ""):
            for dz in (axes3[0], axes3[1], ""):
                if not (lr or ab or dz):
                    continue
                s = ", ".join([lr, ab, dz])
                if s != gt_triple_str:
                    combos.append(s)
    seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(combos)
    return combos[:3]


def _build_spar_bench_round2(
    task_type: str,
    gt: str,
    *,
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> Optional[dict]:
    """Bench-style Round-2 question + options + marker spec for SPAR tasks.

    Returns dict(question, answer_style, options, gt_letter, markers) or None
    (caller falls back to the generic path). markers = [(label, color, kind)].
    """
    if task_type not in SPAR_ROUND2_TYPES:
        return None
    payloads = [sim_parsed_params, sim_validation_result, sim_result]
    gt_str = str(gt or "").strip()
    if not gt_str or gt_str.lower() == "invalid":
        return None
    is_mv = task_type.endswith("_mv")

    if task_type in _SPAR_NUMERICAL_TYPES:
        labels = _spar_payload_get(payloads, "object_labels")
        if isinstance(labels, list):
            labels = [str(x).strip() for x in labels if str(x).strip()]
        if task_type.startswith("spar_distance_prediction_oc"):
            if not labels:
                return None
            a = labels[0]
            text = (
                f"What is the distance between the camera and the {a} (red point)? "
                f"{_SPAR_CALC_NOTE} The unit is meter."
            )
            if is_mv:
                text += f" {_SPAR_MV_VIEW_NOTE}"
            markers = [(a, "red", "point")]
        elif task_type.startswith("spar_distance_prediction_oo"):
            if not labels or len(labels) < 2:
                return None
            a, b = labels[0], labels[1]
            where = "in the images" if is_mv else "in the image"
            text = (
                f"What is the distance between the {a} (red point) and the {b} (blue point) {where}? "
                f"{_SPAR_CALC_NOTE} The unit is meter."
            )
            if is_mv:
                text += f" {_SPAR_MV_VIEW_NOTE}"
            markers = [(a, "red", "point"), (b, "blue", "point")]
        elif task_type.startswith("spar_depth_prediction_oc"):
            if not labels or len(labels) < 2:
                return None
            a, b = labels[0], labels[1]
            anchor_depth = _spar_payload_get(payloads, "anchor_depth")
            if anchor_depth is None:
                return None
            text = (
                f"The camera coordinates show the {a} (red point) at {anchor_depth} meters depth. "
                f"What is the depth of the {b} (blue point)? "
                f"{_SPAR_CALC_NOTE} The unit is meter."
            )
            markers = [(a, "red", "point"), (b, "blue", "point")]
        else:  # spar_depth_prediction_oo(_mv)
            if not labels or len(labels) < 2:
                return None
            a, b = labels[0], labels[1]
            text = (
                f"What is the depth difference (absolute value) between the {a} (green point) "
                f"and the {b} (blue point) in meters? {_SPAR_CALC_NOTE}"
            )
            if is_mv:
                text += f" {_SPAR_DEPTH_MV_NOTE}"
            markers = [(a, "green", "point"), (b, "blue", "point")]
        final_question = text + VILASR_TYPE_TEMPLATE["numerical"]
        return {"question": final_question, "answer_style": "numerical",
                "options": None, "gt_letter": "", "markers": markers}

    if task_type in _SPAR_RELATION_TYPES:
        target = _spar_payload_get(payloads, "target_label")
        ref = _spar_payload_get(payloads, "ref_label")
        if not target:
            return None
        target = str(target).strip()
        ref = str(ref or "").strip()
        if task_type == "spar_obj_spatial_relation_oc_mv":
            text = (
                f"How is object {target} (bbox) situated with respect to the observer's main viewpoint? "
                f"{_SPAR_CALC_NOTE} "
                "The observer's perspective is captured by setting the first image as the primary view.\n"
                f"{_SPAR_RELATION_OPTIONS_DESC_OC}\n"
                "Pick the right response from the available choices."
            )
            axes3 = ("front", "behind")
            markers = [(target, "red", "bbox")]
        else:
            if not ref:
                return None
            if task_type.endswith("_oo_mv"):
                lead = (
                    f"In the two images, explain the location of the {target} (red bbox) with respect to "
                    f"the {ref} (blue bbox), from the observer's perspective. {_SPAR_CALC_NOTE} "
                    "To illustrate the observer's perspective, we designate the first image as the main focus."
                )
            else:
                lead = (
                    f"In the provided image, explain the location of the {target} (red bbox) with respect to "
                    f"the {ref} (blue bbox), considering the observer's viewpoint. {_SPAR_CALC_NOTE}"
                )
            text = (
                f"{lead}\n{_SPAR_RELATION_OPTIONS_DESC_OO}\n"
                "Please select the correct option from the choices provided."
            )
            axes3 = ("farther", "closer")
            markers = [(target, "red", "bbox"), (ref, "blue", "bbox")]
        # GT is the bench triple string ("right, , farther"); build 4 options.
        distractors = _spar_relation_distractors(gt_str, axes3, f"{task_type}||{target}||{ref}||{gt_str}")
        options = [gt_str] + distractors
        seed = int(hashlib.md5(f"opt||{task_type}||{target}||{gt_str}".encode("utf-8")).hexdigest()[:8], 16)
        random.Random(seed).shuffle(options)
        gt_letter = chr(ord("A") + options.index(gt_str))
        option_lines = [f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(options)]
        final_question = text + "\n" + "\n".join(option_lines) + VILASR_TYPE_TEMPLATE["multiple choice"]
        return {"question": final_question, "answer_style": "multiple choice",
                "options": options, "gt_letter": gt_letter, "markers": markers}

    # distance_infer_center_oo(_mv)
    anchor = _spar_payload_get(payloads, "anchor_label")
    opt_a = _spar_payload_get(payloads, "option_a_label")
    opt_b = _spar_payload_get(payloads, "option_b_label")
    if not (anchor and opt_a and opt_b):
        return None
    anchor, opt_a, opt_b = str(anchor).strip(), str(opt_a).strip(), str(opt_b).strip()
    gt_letter = gt_str.upper()[:1]
    if gt_letter not in ("A", "B"):
        return None
    options = [f"{opt_a} (green point)", f"{opt_b} (blue point)"]
    text = (
        f"From the given positions, which is farther to {anchor} (red point): "
        f"{opt_a} (green point) or {opt_b} (blue point)? {_SPAR_CALC_NOTE} "
        "Select the appropriate response from the given choices.\n"
        f"A. {options[0]}\nB. {options[1]}\n"
        "Your answer can only include one of options A, B."
    )
    final_question = text + VILASR_TYPE_TEMPLATE["multiple choice"]
    markers = [(anchor, "red", "point"), (opt_a, "green", "point"), (opt_b, "blue", "point")]
    return {"question": final_question, "answer_style": "multiple choice",
            "options": options, "gt_letter": gt_letter, "markers": markers}


def _build_dynamic_choice_hint(
    question: str,
    task_type: str,
    *,
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> str:
    if task_type == "depth_order_obj_obj":
        candidates = _extract_depth_order_answer_candidates_from_sim(
            sim_parsed_params=sim_parsed_params,
            sim_validation_result=sim_validation_result,
            sim_result=sim_result,
        )
        if not candidates:
            return ""
        labels = _seeded_shuffle_labels(task_type, question, [candidates[0], candidates[1], "same"])
        return " Answer with exactly one of: " + ", ".join(labels) + "."

    fixed_choices = ROUND_DYNAMIC_FIXED_CHOICE_POOLS.get(task_type)
    if not fixed_choices:
        return ""
    labels = _seeded_shuffle_labels(task_type, question, fixed_choices)
    return " A concise answer is usually one of: " + ", ".join(labels) + "."

def _build_question_with_answer_hint(
    question: str,
    task_type: str,
    gt: str | None = None,
    *,
    round_stage: int,
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> str:
    base_question = _strip_known_answer_suffix(question)
    choice_suffix = _build_dynamic_choice_hint(
        base_question,
        task_type,
        sim_parsed_params=sim_parsed_params,
        sim_validation_result=sim_validation_result,
        sim_result=sim_result,
    )
    if round_stage == 1:
        suffix = choice_suffix or ROUND1_GENERIC_DIRECTION_SUFFIXES.get(
            task_type,
            TASK_ANSWER_FORMAT_SUFFIXES.get(task_type, ""),
        )
    else:
        suffix = _build_round2_direction_hint(base_question, task_type, str(gt or ""))
        if not suffix:
            suffix = choice_suffix or ROUND1_GENERIC_DIRECTION_SUFFIXES.get(
                task_type,
                TASK_ANSWER_FORMAT_SUFFIXES.get(task_type, ""),
            )

    if suffix and not _normalize_suffix_spacing(base_question).endswith(_normalize_suffix_spacing(suffix)):
        return base_question + suffix
    return base_question

# ---------------------------------------------------------------------------
# ViLaSR-style Round-2 question construction (multiple-choice options blocks +
# numerical answer suffixes), matching the ViLaSR GRPO training-set format.
# ---------------------------------------------------------------------------
NUMERICAL_ANSWER_TASK_TYPES = {
    "object_counting",
    "absolute_distance",
    "object_size",
    "room_size",
    "distance_cam_obj",
}
MC_OPTION_COUNT = 4

def _build_round2_mc_options(
    question: str,
    task_type: str,
    gt: str,
    *,
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> Optional[tuple[list[str], str]]:
    """Build a ViLaSR-style A)/B)/C)/D) options list around the DGE ground truth.

    Returns (option_labels, gt_letter) or None when no option pool applies.
    option_labels are the raw label strings in display order (letters assigned
    by position: A, B, C, ...).
    """
    if task_type == MAZE_TASK_TYPE:
        gt_letter = str(gt or "").strip().upper()
        if gt_letter not in {"A", "B", "C", "D"}:
            return None
        return ["A", "B", "C", "D"], gt_letter

    if task_type == "relative_distance":
        # VERSE note: VSI-Bench object_rel_distance lists the candidate
        # objects as lettered options in the same order they appear inside the
        # question text — no shuffle, no distractor pool.
        candidates = _extract_relative_distance_candidates_from_sim(
            sim_parsed_params=sim_parsed_params,
            sim_validation_result=sim_validation_result,
            sim_result=sim_result,
        )
        gt_norm = str(gt or "").strip().lower()
        if candidates and gt_norm:
            normalized = [str(c).strip() for c in candidates if str(c).strip()]
            lowered = [c.lower() for c in normalized]
            if gt_norm in lowered:
                gt_letter = chr(ord("A") + lowered.index(gt_norm))
                return normalized, gt_letter
        return None

    gt_label = _normalize_direction_label_for_hint(gt) if task_type in ROUND2_DIRECTION_LABEL_POOLS else str(gt or "").strip().lower()
    if not gt_label or gt_label == "invalid":
        return None

    pool: list[str] = []
    if task_type in ROUND2_DIRECTION_LABEL_POOLS:
        pool = list(ROUND2_DIRECTION_LABEL_POOLS[task_type])
    elif task_type in ROUND_DYNAMIC_FIXED_CHOICE_POOLS:
        pool = list(ROUND_DYNAMIC_FIXED_CHOICE_POOLS[task_type])
    elif task_type == "depth_order_obj_obj":
        candidates = _extract_depth_order_answer_candidates_from_sim(
            sim_parsed_params=sim_parsed_params,
            sim_validation_result=sim_validation_result,
            sim_result=sim_result,
        )
        if not candidates:
            return None
        pool = [str(candidates[0]).strip().lower(), str(candidates[1]).strip().lower(), "same"]
    else:
        return None

    normalized_pool = []
    seen = set()
    for label in pool:
        norm = str(label or "").strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            normalized_pool.append(norm)
    if gt_label not in normalized_pool:
        normalized_pool.append(gt_label)

    distractors = [label for label in normalized_pool if label != gt_label]
    seed_source = f"mc||{task_type}||{question}||{gt_label}"
    seed = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(distractors)

    options = [gt_label] + distractors[: max(1, MC_OPTION_COUNT - 1)]
    rng.shuffle(options)
    gt_letter = chr(ord("A") + options.index(gt_label))
    return options, gt_letter

def _build_round2_agentic_question(
    question: str,
    task_type: str,
    gt: str,
    *,
    sim_parsed_params: Any = None,
    sim_validation_result: Any = None,
    sim_result: Any = None,
) -> tuple[str, str, Optional[list[str]], str]:
    """ViLaSR-style Round-2 question: options block for choice tasks, numerical
    suffix for metric tasks, legacy hint fallback otherwise.

    Returns (final_question, answer_style, options or None, gt_letter).
    answer_style in {"multiple choice", "numerical", "other"}.
    """
    base_question = _strip_known_answer_suffix(question)

    # VERSE note: relative_direction_hard — re-render the question in
    # the exact VSI-Bench "hard" (4 quadrants + Cartesian note) or "medium"
    # (left/right/back + 135-degree note) form, depending on the DGE GT.
    # GT="front" (no bench equivalent) falls through to the generic 8-way form.
    if task_type == "relative_direction_hard":
        bench_style = _build_relative_direction_bench_question(
            gt,
            sim_parsed_params=sim_parsed_params,
            sim_validation_result=sim_validation_result,
            sim_result=sim_result,
        )
        if bench_style is not None:
            bench_text, options, gt_letter = bench_style
            option_lines = [f"{chr(ord('A') + i)}. {label}" for i, label in enumerate(options)]
            final_question = bench_text + "\n" + "\n".join(option_lines) + VILASR_TYPE_TEMPLATE["multiple choice"]
            return final_question, "multiple choice", options, gt_letter

    mc = _build_round2_mc_options(
        base_question,
        task_type,
        gt,
        sim_parsed_params=sim_parsed_params,
        sim_validation_result=sim_validation_result,
        sim_result=sim_result,
    )
    if mc is not None:
        options, gt_letter = mc
        if task_type == MAZE_TASK_TYPE:
            option_lines = [f"{chr(ord('A') + i)})" for i in range(len(options))]
        else:
            # VERSE note: "A. label" marker, matching the benchmark
            # option format rendered at eval time.
            option_lines = [f"{chr(ord('A') + i)}. {label}" for i, label in enumerate(options)]
        final_question = base_question + "\n" + "\n".join(option_lines) + VILASR_TYPE_TEMPLATE["multiple choice"]
        return final_question, "multiple choice", options, gt_letter

    if task_type in NUMERICAL_ANSWER_TASK_TYPES:
        final_question = base_question + VILASR_TYPE_TEMPLATE["numerical"]
        return final_question, "numerical", None, ""

    final_question = _build_question_with_answer_hint(
        question,
        task_type,
        gt=gt,
        round_stage=2,
        sim_parsed_params=sim_parsed_params,
        sim_validation_result=sim_validation_result,
        sim_result=sim_result,
    )
    return final_question, "other", None, ""

ROUND2_ANSWER_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND2_ANSWER_PROMPT_PATH,
    "Round2 answer prompt",
)
ROUND2_INVALID_EXPLANATION_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND2_INVALID_EXPLANATION_PROMPT_PATH,
    "Round2 invalid explanation prompt",
)
ROUND1_SCENE_TASK_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND1_SCENE_TASK_PROMPT_TEMPLATE_PATH,
    "Round1 scene task prompt",
)
ROUND1_SINGLE_IMAGE_TASK_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND1_SINGLE_IMAGE_TASK_PROMPT_TEMPLATE_PATH,
    "Round1 single-image task prompt",
)
ROUND1_IMAGE_PAIR_TASK_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND1_IMAGE_PAIR_TASK_PROMPT_TEMPLATE_PATH,
    "Round1 image-pair task prompt",
)
ROUND1_MAZE_TASK_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND1_MAZE_TASK_PROMPT_TEMPLATE_PATH,
    "Round1 maze task prompt",
)
ROUND1_MAZE_PICK_TASK_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND1_MAZE_PICK_TASK_PROMPT_TEMPLATE_PATH,
    "Round1 maze pick-letter task prompt",
)
ROUND1_DGE_STRUCTURED_PROMPT_TEMPLATE = _load_required_text_file(
    ROUND1_DGE_STRUCTURED_PROMPT_TEMPLATE_PATH,
    "Round1 DGE structured-imitate task prompt",
)
ROUND1_TASK_PROMPT_SPECS = _load_required_json_file(
    ROUND1_TASK_PROMPT_SPECS_PATH,
    "Round1 task prompt specs",
).get("tasks") or {}

class Role(IntEnum):

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto() # Actor + Rollout
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()  # Actor + Rollout + RefPolicy

@dataclass
class ResourcePoolManager:

    resource_pool_spec: dict[str, list[int]]  # Pool name -> GPU counts per node.
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,  # GPU count per node.
                use_gpu=True,  # Allocate GPU resources.
                max_colocate_count=1,  # One worker group per pool slot.
                name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Available GPU count {gpus_available} is smaller than required GPU count {gpus_required}.")

def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]  # Token-level reward scores.
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]  # Valid response tokens.

    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # Token-level KL over valid response tokens.

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics

def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],  # Token-level rewards.
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,  # GAE lambda.
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data

class RayPPOTrainer:

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self._world_simulator_cache: dict[str, WorldSimulator] = {}
        self._world_simulator_thread_local = threading.local()
        self._round1_summary_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._round1_task_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "acc_sum": 0.0, "scheduler_acc_sum": 0.0}
        )
        self._round1_task_rng = np.random.default_rng(int(getattr(config.data, "seed", 1) or 1))
        self._split_role_frozen_proposer = _env_flag("SPATIALEVO_SPLIT_ROLE_FROZEN_PROPOSER", False)
        self._proposer_openai_model = os.environ.get("SPATIALEVO_PROPOSER_OPENAI_MODEL", "").strip()
        self._proposer_openai_base_url = os.environ.get("SPATIALEVO_PROPOSER_OPENAI_BASE_URL", "").strip()
        self._proposer_openai_api_key = os.environ.get("SPATIALEVO_PROPOSER_OPENAI_API_KEY", "").strip()
        self._proposer_openai_temperature = float(os.environ.get("SPATIALEVO_PROPOSER_TEMPERATURE", "1.0"))
        self._proposer_openai_top_p = float(os.environ.get("SPATIALEVO_PROPOSER_TOP_P", "1.0"))
        self._proposer_openai_timeout = float(os.environ.get("SPATIALEVO_PROPOSER_TIMEOUT", "120"))
        self._proposer_openai_max_retries = int(os.environ.get("SPATIALEVO_PROPOSER_MAX_RETRIES", "6"))
        self._proposer_openai_max_workers = int(os.environ.get("SPATIALEVO_PROPOSER_MAX_WORKERS", "4"))
        self._proposer_openai_client: Optional[OpenAI] = None
        # SFT-on-correct-rollouts (STaR/RFT) mode: replace the R2 GRPO update with an
        # SFT update on the solver's own correct rollouts. Correct = is_valid AND
        # reward accuracy >= threshold (exact for discrete, MRA>=thr for numeric).
        self._sft_on_correct = _env_flag("SPATIALEVO_SFT_ON_CORRECT", False)
        self._sft_correct_threshold = float(os.environ.get("SPATIALEVO_SFT_CORRECT_THRESHOLD", "0.8"))
        self._sft_include_invalid = _env_flag("SPATIALEVO_SFT_INCLUDE_INVALID", False)
        # Collect-only mode: run the full proposer->DGE->solver rollout pipeline but SKIP the
        # solver update, so the solver stays fixed (base 8B) and we just accumulate its correct
        # rollouts (logged to round2_valid_qa.jsonl with SAMPLE_LOG_FREQ=1) for a later offline SFT.
        self._sft_collect_only = _env_flag("SPATIALEVO_SFT_COLLECT_ONLY", False)
        # VERSE note:selection-over-reward proposer design:
        # - SPATIALEVO_ROUND1_SKIP_UPDATE: keep sampling Round-1 from the live policy but skip
        #   the Round-1 actor update entirely (the structured_imitate proposer only ever gets a
        #   saturated 0.1*format gradient — the update is pure overhead once format converges).
        # - SPATIALEVO_QUESTION_RESAMPLE: after DGE validation, replace each invalid proposal
        #   with a system-sampled valid question (labels drawn from the same scene pools shown
        #   in the Round-1 prompt, rendered through the same templates, re-verified by DGE).
        #   Validity is enforced by construction/selection instead of by reward, so the
        #   expensive Round-2 agentic rollouts are spent on real answering signal.
        self._round1_skip_update = _env_flag("SPATIALEVO_ROUND1_SKIP_UPDATE", False)
        self._question_resample = _env_flag("SPATIALEVO_QUESTION_RESAMPLE", False)
        self._question_resample_tries = int(os.environ.get("SPATIALEVO_QUESTION_RESAMPLE_TRIES", "8"))
        # Ablation knob: comma-separated task types excluded from system resample.
        self._question_resample_blacklist = {
            t.strip()
            for t in os.environ.get("SPATIALEVO_RESAMPLE_TASK_BLACKLIST", "").split(",")
            if t.strip()
        }
        # VERSE note:verified trajectory bank (cheap train-time-search):
        # persist every valid + correct (+ maze-fidelity-passing) solver rollout each step so
        # they can later be distilled via rejection-sampling SFT. JSONL under checkpoints dir.
        self._traj_bank_enabled = _env_flag("SPATIALEVO_TRAJ_BANK", False)
        self._traj_bank_acc_threshold = float(os.environ.get("SPATIALEVO_TRAJ_BANK_ACC_THRESHOLD", "0.8"))
        self._traj_bank_fidelity_threshold = float(os.environ.get("SPATIALEVO_TRAJ_BANK_FIDELITY_THRESHOLD", "0.6"))
        if self._split_role_frozen_proposer and not (
            self._proposer_openai_model and self._proposer_openai_base_url and self._proposer_openai_api_key
        ):
            raise ValueError(
                "SPATIALEVO_SPLIT_ROLE_FROZEN_PROPOSER=1 requires "
                "SPATIALEVO_PROPOSER_OPENAI_MODEL, SPATIALEVO_PROPOSER_OPENAI_BASE_URL, "
                "and SPATIALEVO_PROPOSER_OPENAI_API_KEY."
            )
        for task_names in ROUND1_TASKS_BY_MODALITY.values():
            for task_name in task_names:
                _ = self._round1_task_stats[task_name]

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping  # Role -> worker class map.
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, so KL metrics will not be logged. Set `kl_coef=0` instead if you still want KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by the actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by the actor experience micro-batch size."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by the critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by the critic experience micro-batch size."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO require `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    @staticmethod
    def _extract_media_paths_from_item(item: Any) -> list[str]:
        if not isinstance(item, dict):
            return []

        for key in ("images", "videos"):
            raw_paths = item.get(key)
            if not isinstance(raw_paths, list):
                continue
            return [path for path in raw_paths if isinstance(path, str) and path.strip()]
        return []

    @staticmethod
    def _infer_scene_id_from_paths(paths: list[str]) -> str:
        # Infer the dataset scene id from media paths.
        for raw_path in paths:
            path = Path(raw_path)
            candidates = [path.stem] + [parent.name for parent in path.parents]
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.lower().startswith("scene"):
                    return candidate
        return "unknown"

    @staticmethod
    def _infer_scene_root_and_path(scene_id: str, paths: list[str]) -> tuple[Optional[str], Optional[str]]:
        # Infer the normalized scene directory.
        if not isinstance(scene_id, str) or not scene_id.strip():
            return None, None

        for raw_path in paths:
            path = Path(raw_path)
            for current in [path.parent] + list(path.parents):
                if current.name == scene_id:
                    scene_root = str(current.parent)
                    scene_data_path = str(current)
                    return scene_root, scene_data_path
        return None, None

    @staticmethod
    def _infer_metadata_dir_from_paths(paths: list[str]) -> Optional[str]:
        # Infer the metadata directory.
        for raw_path in paths:
            path = Path(raw_path)
            for current in [path.parent] + list(path.parents):
                if current.name == "metadata":
                    return str(current)
                if current.name.lower() in DATASET_SPLIT_DIR_NAMES:
                    candidate = current.parent / "metadata"
                    if candidate.exists():
                        return str(candidate)
        return None

    def _build_simulator_media_context(self, item: Any) -> dict[str, Any]:
        # Build simulator context from media paths.
        image_paths = self._extract_media_paths_from_item(item)
        scene_id = self._infer_scene_id_from_paths(image_paths)
        scene_root, scene_data_path = self._infer_scene_root_and_path(scene_id, image_paths)
        metadata_dir = self._infer_metadata_dir_from_paths(image_paths)

        context: dict[str, Any] = {
            "scene_id": scene_id,
            "image_paths": image_paths,
            "_scene_root": scene_root,
        }
        if len(image_paths) == 1:
            context["image_path"] = image_paths[0]
        if len(image_paths) >= 2:
            context["image_path_1"] = image_paths[0]
            context["image_path_2"] = image_paths[1]
        if metadata_dir:
            context["metadata_dir"] = metadata_dir
        if scene_data_path:
            context["scene_data_path"] = scene_data_path
        if isinstance(item, dict):
            summary_context = item.get("_round1_summary_context")
            if isinstance(summary_context, dict):
                context["_round1_summary_context"] = deepcopy(summary_context)
        return context

    def _get_world_simulator_for_root(self, scene_root: Optional[str]) -> WorldSimulator:
        cache_key = scene_root or "__default__"
        simulator = self._world_simulator_cache.get(cache_key)
        if simulator is not None:
            return simulator

        simulator = self._create_world_simulator(scene_root)
        self._world_simulator_cache[cache_key] = simulator
        return simulator

    def _create_world_simulator(self, scene_root: Optional[str]) -> WorldSimulator:
        init_kwargs = {"enable_vlm": True}
        if isinstance(scene_root, str) and scene_root.strip():
            init_kwargs["scannet_root"] = scene_root
        return WorldSimulator(**init_kwargs)

    def _get_thread_local_world_simulator_for_root(self, scene_root: Optional[str]) -> WorldSimulator:
        cache = getattr(self._world_simulator_thread_local, "cache", None)
        if cache is None:
            cache = {}
            self._world_simulator_thread_local.cache = cache

        cache_key = scene_root or "__default__"
        simulator = cache.get(cache_key)
        if simulator is None:
            simulator = self._create_world_simulator(scene_root)
            cache[cache_key] = simulator
        return simulator

    def _get_world_simulator_max_workers(self, batch_size: int) -> int:
        configured = getattr(self.config.trainer, "world_simulator_max_workers", None)
        try:
            configured_workers = int(configured) if configured is not None else 0
        except (TypeError, ValueError):
            configured_workers = 0

        if configured_workers <= 0:
            configured_workers = 4
        return max(1, min(batch_size, configured_workers))

    @staticmethod
    def _build_simulator_query_task(
        media_context: dict[str, Any],
        canonical_task_type: str,
        question_text: str,
        prefilled_params: Optional[dict] = None,
    ) -> dict[str, Any]:
        summary_context = media_context.get("_round1_summary_context")
        prompt_mode = "scene"
        if isinstance(summary_context, dict):
            prompt_mode = str(summary_context.get("prompt_mode", "scene") or "scene")

        query_task = {
            "task_type": canonical_task_type,
            "scene_id": media_context.get("scene_id", "unknown"),
            "question": question_text,
        }
        # VERSE note: inject structured_imitate params directly so DGE skips the
        # lossy LLM extraction round-trip (params → template → question → LLM → params).
        if isinstance(prefilled_params, dict) and prefilled_params:
            for k, v in prefilled_params.items():
                query_task[k] = v
        allowed_media_keys = ["metadata_dir", "scene_data_path"]
        if prompt_mode == "single_image":
            allowed_media_keys.extend(["image_path", "image_paths"])
        elif prompt_mode == "image_pair":
            allowed_media_keys.extend(["image_paths", "image_path_1", "image_path_2"])
        else:
            allowed_media_keys.append("image_paths")

        for key in allowed_media_keys:
            value = media_context.get(key)
            if isinstance(value, str) and value.strip():
                query_task[key] = value
            elif isinstance(value, list) and value:
                query_task[key] = value
        if isinstance(summary_context, dict):
            for key in (
                "prompt_mode",
                "scene_label_counts",
                "frame_visible_label_counts",
                "image1_visible_label_counts",
                "image2_visible_label_counts",
                "pair_visibility_contrast_labels",
                "pair_camera_elevation_answer",
            ):
                value = summary_context.get(key)
                if isinstance(value, dict) and value:
                    query_task[key] = deepcopy(value)
                elif isinstance(value, list) and value:
                    query_task[key] = list(value)
                elif isinstance(value, str) and value.strip():
                    query_task[key] = value.strip()
        return query_task

    def _run_simulator_validation_job(
        self,
        *,
        media_context: dict[str, Any],
        canonical_task_type: str,
        question_text: str,
        use_thread_local_simulator: bool,
        prefilled_params: Optional[dict] = None,
    ) -> dict[str, Any]:
        if canonical_task_type == MAZE_TASK_TYPE:
            # Deterministic maze verifier: simulate the proposed action sequence
            # against the maze metadata; no DGE / WorldSimulator involved.
            image_paths = media_context.get("image_paths", []) or []
            res = MazeEngine.validate_and_answer(
                question_text=question_text,
                image_path=image_paths[0] if image_paths else None,
            )
            return {
                "resolved_task_type": MAZE_TASK_TYPE,
                "answer": res.get("answer", "invalid") or "invalid",
                "is_valid": bool(res.get("is_valid", False)),
                "error": str(res.get("error", "") or ""),
                "error_code": str(res.get("error_code", "") or ""),
                "failure_stage": str(res.get("failure_stage", "") or ("ok" if res.get("is_valid") else "unknown")),
                "parsed_params": res.get("parsed_params", {}) or {},
                "validation_result": res.get("validation_result"),
                "judge_reference": res.get("judge_reference"),
                "sim_result": res,
            }
        if use_thread_local_simulator:
            simulator = self._get_thread_local_world_simulator_for_root(media_context.get("_scene_root"))
        else:
            simulator = self._get_world_simulator_for_root(media_context.get("_scene_root"))

        query_task = self._build_simulator_query_task(
            media_context=media_context,
            canonical_task_type=canonical_task_type,
            question_text=question_text,
            prefilled_params=prefilled_params,
        )
        res = simulator.validate_and_answer(query_task)
        resolved_task_type = res.get("task_type")
        if not (isinstance(resolved_task_type, str) and resolved_task_type.strip()):
            resolved_task_type = canonical_task_type

        answer = res.get("answer")
        if answer == "":
            answer = "invalid"
        is_valid = res.get("is_valid", False)
        failure_stage = res.get("failure_stage")
        if not isinstance(failure_stage, str) or not failure_stage.strip():
            failure_stage = "ok" if bool(is_valid) else "unknown"

        return {
            "resolved_task_type": resolved_task_type,
            "answer": answer,
            "is_valid": bool(is_valid),
            "error": str(res.get("error", "") or ""),
            "error_code": str(res.get("error_code", "") or ""),
            "failure_stage": failure_stage,
            "parsed_params": res.get("parsed_params", {}) or {},
            "validation_result": res.get("validation_result"),
            "judge_reference": res.get("judge_reference"),
            "sim_result": res,
        }

    @staticmethod
    def _resolve_canonical_task_type(simulator: WorldSimulator, raw_question_type: Any) -> str:
        resolution = simulator.resolve_task_type_name(raw_question_type)
        task_type = resolution.get("task_type")
        if isinstance(task_type, str) and task_type.strip():
            return task_type
        return "unknown"

    @staticmethod
    def _format_label_counts_text(label_counts: Optional[dict[str, Any]]) -> str:
        if not isinstance(label_counts, dict) or not label_counts:
            return ""

        formatted = []
        for label, count in sorted(label_counts.items()):
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                numeric_count = 1
            suffix = "(Unique)" if numeric_count == 1 else "(Non-Unique)"
            formatted.append(f"{label}{suffix}")
        return ", ".join(formatted)

    @staticmethod
    def _format_label_list_text(labels: Optional[list[str]]) -> str:
        if not isinstance(labels, list) or not labels:
            return ""
        return ", ".join(sorted(str(label) for label in labels if str(label).strip()))

    @staticmethod
    def _split_label_counts(label_counts: Optional[dict[str, Any]]) -> tuple[list[str], list[str]]:
        if not isinstance(label_counts, dict):
            return [], []

        unique_labels = []
        non_unique_labels = []
        for label, count in sorted(label_counts.items()):
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                numeric_count = 1
            if numeric_count == 1:
                unique_labels.append(str(label))
            elif numeric_count > 1:
                non_unique_labels.append(str(label))
        return unique_labels, non_unique_labels

    @staticmethod
    def _get_label_count(label_counts: Optional[dict[str, Any]], label: str) -> int:
        if not isinstance(label_counts, dict):
            return 0
        try:
            return int(label_counts.get(label, 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _is_usable_prompt_label(label: Any) -> bool:
        text = str(label or "").strip().lower()
        if not text:
            return False
        return text not in PROMPT_NON_ANCHOR_LABELS

    @classmethod
    def _filter_prompt_label_counts(
        cls,
        label_counts: Optional[dict[str, Any]],
    ) -> dict[str, int]:
        if not isinstance(label_counts, dict):
            return {}

        filtered: dict[str, int] = {}
        for label, count in label_counts.items():
            text = str(label or "").strip()
            if not cls._is_usable_prompt_label(text):
                continue
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                numeric_count = 1
            if numeric_count > 0:
                filtered[text] = numeric_count
        return dict(sorted(filtered.items()))

    @staticmethod
    def _visible_object_instance_key(
        obj: dict[str, Any],
        frame_index: int,
        object_index: int,
    ) -> str:
        object_id = obj.get("object_id", obj.get("id"))
        if object_id not in (None, ""):
            return f"id::{object_id}"

        location = obj.get("3d_location")
        if isinstance(location, (list, tuple)) and len(location) >= 3:
            try:
                xyz = tuple(round(float(location[idx]), 4) for idx in range(3))
                return f"loc::{xyz[0]}::{xyz[1]}::{xyz[2]}"
            except (TypeError, ValueError):
                pass

        label = str(obj.get("label", "")).strip().lower()
        return f"fallback::{frame_index}::{object_index}::{label}"

    @classmethod
    def _build_observed_scene_label_counts(
        cls,
        frame_summaries: Optional[list[dict[str, Any]]],
    ) -> dict[str, int]:
        if not isinstance(frame_summaries, list):
            return {}

        label_to_instances: dict[str, set[str]] = defaultdict(set)
        for frame_index, frame_summary in enumerate(frame_summaries, 1):
            visible_objects = frame_summary.get("visible_objects", []) if isinstance(frame_summary, dict) else []
            if not isinstance(visible_objects, list):
                continue
            for object_index, obj in enumerate(visible_objects):
                if not isinstance(obj, dict):
                    continue
                label = str(obj.get("label", "")).strip()
                if not cls._is_usable_prompt_label(label):
                    continue
                instance_key = cls._visible_object_instance_key(obj, frame_index, object_index)
                label_to_instances[label].add(instance_key)

        return {
            label: len(instance_keys)
            for label, instance_keys in sorted(label_to_instances.items())
            if instance_keys
        }

    def _get_pair_non_ambiguous_labels(
        self,
        image1_counts: Optional[dict[str, Any]],
        image2_counts: Optional[dict[str, Any]],
    ) -> list[str]:
        labels = set()
        if isinstance(image1_counts, dict):
            labels.update(str(label) for label in image1_counts.keys())
        if isinstance(image2_counts, dict):
            labels.update(str(label) for label in image2_counts.keys())

        valid_labels = []
        for label in sorted(labels):
            count1 = self._get_label_count(image1_counts, label)
            count2 = self._get_label_count(image2_counts, label)
            if count1 + count2 >= 1 and count1 <= 1 and count2 <= 1:
                valid_labels.append(label)
        return valid_labels

    def _get_shared_unique_pair_labels(
        self,
        image1_counts: Optional[dict[str, Any]],
        image2_counts: Optional[dict[str, Any]],
    ) -> list[str]:
        shared_unique = []
        labels = set()
        if isinstance(image1_counts, dict):
            labels.update(str(label) for label in image1_counts.keys())
        if isinstance(image2_counts, dict):
            labels.update(str(label) for label in image2_counts.keys())

        for label in sorted(labels):
            if self._get_label_count(image1_counts, label) == 1 and self._get_label_count(image2_counts, label) == 1:
                shared_unique.append(label)
        return shared_unique

    @staticmethod
    def _build_label_visibility_map(visible_objects: Optional[list[dict[str, Any]]]) -> dict[str, float]:
        if not isinstance(visible_objects, list):
            return {}

        label_to_visibility: dict[str, float] = {}
        for obj in visible_objects:
            if not isinstance(obj, dict):
                continue
            label = str(obj.get("label") or obj.get("name") or "").strip()
            if not RayPPOTrainer._is_usable_prompt_label(label):
                continue
            try:
                visibility = float(obj.get("visibility", 0.0))
            except (TypeError, ValueError):
                visibility = 0.0
            prev_visibility = label_to_visibility.get(label)
            if prev_visibility is None or visibility > prev_visibility:
                label_to_visibility[label] = visibility
        return dict(sorted(label_to_visibility.items()))

    def _get_pair_visibility_contrast_candidates(
        self,
        image1_counts: Optional[dict[str, Any]],
        image2_counts: Optional[dict[str, Any]],
        image1_visibility_map: Optional[dict[str, Any]],
        image2_visibility_map: Optional[dict[str, Any]],
        *,
        min_delta: float = 0.15,
    ) -> list[dict[str, Any]]:
        candidate_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        candidates: list[dict[str, Any]] = []

        for label in candidate_labels:
            try:
                image1_visibility = float((image1_visibility_map or {}).get(label, 0.0))
            except (TypeError, ValueError):
                image1_visibility = 0.0
            try:
                image2_visibility = float((image2_visibility_map or {}).get(label, 0.0))
            except (TypeError, ValueError):
                image2_visibility = 0.0

            delta = abs(image1_visibility - image2_visibility)
            if delta < min_delta:
                continue

            if image1_visibility > image2_visibility:
                preferred_answer = "image1"
            elif image2_visibility > image1_visibility:
                preferred_answer = "image2"
            else:
                preferred_answer = "same"

            candidates.append(
                {
                    "label": label,
                    "image1_visibility": image1_visibility,
                    "image2_visibility": image2_visibility,
                    "delta": delta,
                    "preferred_answer": preferred_answer,
                }
            )

        candidates.sort(
            key=lambda item: (
                -float(item.get("delta", 0.0)),
                -max(float(item.get("image1_visibility", 0.0)), float(item.get("image2_visibility", 0.0))),
                str(item.get("label", "")),
            )
        )
        return candidates

    @staticmethod
    def _format_visibility_contrast_text(candidates: Optional[list[dict[str, Any]]], max_items: int = 12) -> str:
        if not isinstance(candidates, list) or not candidates:
            return ""

        formatted: list[str] = []
        for item in candidates[:max_items]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            try:
                delta = float(item.get("delta", 0.0))
            except (TypeError, ValueError):
                delta = 0.0
            try:
                image1_visibility = float(item.get("image1_visibility", 0.0))
            except (TypeError, ValueError):
                image1_visibility = 0.0
            try:
                image2_visibility = float(item.get("image2_visibility", 0.0))
            except (TypeError, ValueError):
                image2_visibility = 0.0
            preferred_answer = str(item.get("preferred_answer", "")).strip()
            formatted.append(
                f"{label}(Δvis={delta:.2f}, I1={image1_visibility:.2f}, I2={image2_visibility:.2f}, likely={preferred_answer})"
            )
        return ", ".join(formatted)

    @staticmethod
    def _dedupe_string_list(values: Optional[list[str]]) -> list[str]:
        if not isinstance(values, list):
            return []
        deduped = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    def _shuffle_string_list(self, values: Optional[list[str]]) -> list[str]:
        items = self._dedupe_string_list(values)
        if len(items) <= 1:
            return items
        indices = self._round1_task_rng.permutation(len(items)).tolist()
        return [items[idx] for idx in indices]

    @staticmethod
    def _region_name_from_anchor(anchor_label: str) -> str:
        anchor = str(anchor_label or "").strip()
        return f"{anchor} area" if anchor else ""

    @staticmethod
    def _format_prompt_question(template_text: str, **kwargs: str) -> str:
        try:
            question = str(template_text).format(**kwargs).strip()
        except Exception:
            question = ""
        return question

    @staticmethod
    def _replace_prompt_example_tokens(template_text: str, replacements: dict[str, str]) -> str:
        text = str(template_text or "").strip()
        if not text:
            return ""
        rendered = text
        expanded_replacements: dict[str, str] = {}
        for key, value in replacements.items():
            cleaned_value = str(value or "").strip()
            if not cleaned_value:
                continue
            expanded_replacements[str(key)] = cleaned_value
            capitalized_key = str(key)[:1].upper() + str(key)[1:]
            expanded_replacements.setdefault(capitalized_key, cleaned_value)

        for key in sorted(expanded_replacements.keys(), key=len, reverse=True):
            value = expanded_replacements[key]
            if not value:
                continue
            rendered = re.sub(rf"\b{re.escape(key)}\b", value, rendered)
        return " ".join(rendered.split())

    @staticmethod
    def _build_round1_example_replacements(format_kwargs: dict[str, str]) -> dict[str, str]:
        replacements = {
            str(key): str(value).strip()
            for key, value in (format_kwargs or {}).items()
            if str(value).strip()
        }
        if "object_label" in replacements:
            replacements.setdefault("object1", replacements["object_label"])
        if "object_a" in replacements:
            replacements.setdefault("object1", replacements["object_a"])
        if "object_b" in replacements:
            replacements.setdefault("object2", replacements["object_b"])
        if "object_c" in replacements:
            replacements.setdefault("object3", replacements["object_c"])
        if "target_object" in replacements:
            replacements.setdefault("object1", replacements["target_object"])
        if "reference_object" in replacements:
            replacements.setdefault("object2", replacements["reference_object"])
        if "stand_object" in replacements:
            replacements.setdefault("object1", replacements["stand_object"])
            replacements.setdefault("stand_object", replacements["stand_object"])
        if "face_object" in replacements:
            replacements.setdefault("object2", replacements["face_object"])
            replacements.setdefault("face_object", replacements["face_object"])
        if "query_object" in replacements:
            replacements.setdefault("object3", replacements["query_object"])
            replacements.setdefault("query_object", replacements["query_object"])
        if "anchor_object" in replacements:
            replacements.setdefault("anchor_object", replacements["anchor_object"])
        if "region_name" in replacements:
            replacements.setdefault("region1", replacements["region_name"])
            anchor_object = re.sub(r"\s+area$", "", replacements["region_name"]).strip()
            if anchor_object:
                replacements.setdefault("object1", anchor_object)
        return replacements

    def _build_round1_valid_example_specs(
        self,
        task_type: str,
        summary_context: dict[str, Any],
        max_examples: int = 4,
    ) -> list[dict[str, Any]]:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        templates = [
            str(template).strip()
            for template in (prompt_spec.get("question_templates", []) or [])
            if str(template).strip()
        ]
        if not templates:
            return []

        scene_counts = summary_context.get("scene_label_counts", {}) or {}
        scene_unique, scene_non_unique = self._split_label_counts(scene_counts)
        frame_counts = summary_context.get("frame_visible_label_counts", {}) or {}
        frame_unique, _ = self._split_label_counts(frame_counts)
        image1_counts = summary_context.get("image1_visible_label_counts", {}) or {}
        image2_counts = summary_context.get("image2_visible_label_counts", {}) or {}
        image1_unique, _ = self._split_label_counts(image1_counts)
        image2_unique, _ = self._split_label_counts(image2_counts)
        pair_non_ambiguous_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        pair_shared_unique_labels = self._get_shared_unique_pair_labels(image1_counts, image2_counts)
        pair_visibility_contrast_labels = summary_context.get("pair_visibility_contrast_labels", []) or []

        specs: list[dict[str, Any]] = []
        seen_questions: set[str] = set()

        def add_spec(template_index: int = 0, **kwargs: str) -> None:
            if len(specs) >= max_examples:
                return
            template_text = templates[template_index % len(templates)]
            question = self._format_prompt_question(template_text, **kwargs)
            if question and question not in seen_questions:
                seen_questions.add(question)
                specs.append(
                    {
                        "template_index": template_index % len(templates),
                        "format_kwargs": dict(kwargs),
                        "question": question,
                    }
                )

        if task_type == "object_counting":
            labels = self._shuffle_string_list(scene_non_unique)
            for idx, label in enumerate(labels[:max_examples]):
                add_spec(idx, object_label=label)
            return specs

        if task_type == "absolute_distance":
            labels = self._shuffle_string_list(scene_unique)
            for idx, (object_a, object_b) in enumerate(itertools.combinations(labels, 2)):
                add_spec(idx, object_a=object_a, object_b=object_b)
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "object_size":
            labels = self._shuffle_string_list(scene_unique)
            for idx, label in enumerate(labels[:max_examples]):
                add_spec(idx, object_label=label)
            return specs

        if task_type == "room_size":
            for idx in range(min(len(templates), max_examples)):
                add_spec(idx)
            return specs

        if task_type == "relative_distance":
            labels = self._shuffle_string_list(scene_unique)
            if len(labels) < 4:
                return specs
            for idx, anchor_object in enumerate(labels):
                remaining = [label for label in labels if label != anchor_object]
                if len(remaining) < 3:
                    continue
                add_spec(
                    idx,
                    anchor_object=anchor_object,
                    object_a=remaining[0],
                    object_b=remaining[1],
                    object_c=remaining[2],
                )
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "relative_direction_hard":
            labels = self._shuffle_string_list(scene_unique)
            if len(labels) < 3:
                return specs
            for idx, triple in enumerate(itertools.permutations(labels, 3)):
                stand_object, face_object, query_object = triple
                add_spec(
                    idx,
                    stand_object=stand_object,
                    face_object=face_object,
                    query_object=query_object,
                )
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "route_planning":
            labels = self._shuffle_string_list(scene_unique)
            if len(labels) < 4:
                return specs
            for idx, quad in enumerate(itertools.permutations(labels, 4)):
                start_object, second_object, third_object, destination_object = quad
                add_spec(
                    idx,
                    start_object=start_object,
                    second_object=second_object,
                    third_object=third_object,
                    destination_object=destination_object,
                )
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "single_image_relative_direction":
            labels = self._shuffle_string_list(frame_unique)
            for idx, (reference_object, target_object) in enumerate(itertools.permutations(labels, 2)):
                add_spec(idx, reference_object=reference_object, target_object=target_object)
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "distance_cam_obj":
            labels = self._shuffle_string_list(frame_unique)
            for idx, label in enumerate(labels[:max_examples]):
                add_spec(idx, object_label=label)
            return specs

        if task_type == "depth_order_obj_obj":
            labels = self._shuffle_string_list(frame_unique)
            for idx, (object_a, object_b) in enumerate(itertools.combinations(labels, 2)):
                add_spec(idx, object_a=object_a, object_b=object_b)
                if len(specs) >= max_examples:
                    break
            return specs

        if task_type == "elevation_cam_cam":
            pair_elevation_answer = self._normalize_answer_token(
                summary_context.get("pair_camera_elevation_answer", "")
            )
            if pair_elevation_answer == "same_level":
                return specs
            for idx in range(min(len(templates), max_examples)):
                add_spec(idx)
            return specs

        if task_type in {"position_cam_cam", "motion_camera"}:
            for idx in range(min(len(templates), max_examples)):
                add_spec(idx)
            return specs

        if task_type == "visibility_compare":
            labels = self._shuffle_string_list(
                pair_visibility_contrast_labels or pair_shared_unique_labels or pair_non_ambiguous_labels
            )
            for idx, label in enumerate(labels[:max_examples]):
                add_spec(idx, object_label=label)
            return specs

        if task_type == "position_cam_obj":
            image1_labels = self._shuffle_string_list(image1_unique)
            image2_labels = self._shuffle_string_list(image2_unique)
            balanced_candidates: list[tuple[int, str]] = []
            max_branch = max(len(image1_labels), len(image2_labels))
            for idx in range(max_branch):
                if idx < len(image1_labels):
                    balanced_candidates.append((0, image1_labels[idx]))
                if len(templates) > 1 and idx < len(image2_labels):
                    balanced_candidates.append((1, image2_labels[idx]))
            for template_index, label in balanced_candidates[:max_examples]:
                add_spec(template_index, object_label=label)
            return specs

        if task_type == "position_cam_reg":
            image1_regions = [self._region_name_from_anchor(label) for label in self._shuffle_string_list(image1_unique)]
            image2_regions = [self._region_name_from_anchor(label) for label in self._shuffle_string_list(image2_unique)]
            balanced_candidates: list[tuple[int, str]] = []
            max_branch = max(len(image1_regions), len(image2_regions))
            for idx in range(max_branch):
                if idx < len(image1_regions):
                    balanced_candidates.append((0, image1_regions[idx]))
                if len(templates) > 1 and idx < len(image2_regions):
                    balanced_candidates.append((1, image2_regions[idx]))
            for template_index, region_name in balanced_candidates[:max_examples]:
                add_spec(template_index, region_name=region_name)
            return specs

        if task_type == "attribute_measurement":
            image1_labels = self._shuffle_string_list(image1_unique)
            image2_labels = self._shuffle_string_list(image2_unique)
            cross_pairs = []
            for object_a in image1_labels:
                for object_b in image2_labels:
                    if object_a != object_b:
                        cross_pairs.append((object_a, object_b))
            candidate_pairs = cross_pairs
            if not candidate_pairs:
                pair_labels = self._shuffle_string_list(pair_non_ambiguous_labels)
                candidate_pairs = list(itertools.combinations(pair_labels, 2))
            for idx, (object_a, object_b) in enumerate(candidate_pairs):
                add_spec(idx, object_a=object_a, object_b=object_b)
                if len(specs) >= max_examples:
                    break
            return specs

        return specs

    def _get_round1_valid_example_budget(self, task_type: str) -> int:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        return 6 if str(prompt_spec.get("task_group", "") or "") == "image_pair" else 4

    def _build_round1_valid_question_examples(
        self,
        task_type: str,
        summary_context: dict[str, Any],
        max_examples: int | None = None,
    ) -> list[str]:
        budget = max_examples if max_examples is not None else self._get_round1_valid_example_budget(task_type)
        specs = self._build_round1_valid_example_specs(
            task_type=task_type,
            summary_context=summary_context,
            max_examples=budget,
        )
        return [str(item.get("question", "")).strip() for item in specs if str(item.get("question", "")).strip()]

    @staticmethod
    def _infer_prompt_mode_from_media_context(media_context: dict[str, Any]) -> str:
        image_paths = media_context.get("image_paths", [])
        # Synthetic maze contexts (generated pool) are validated by MazeEngine, not the DGE.
        if len(image_paths) == 1 and MazeEngine.is_maze_image_path(image_paths[0]):
            return "maze"
        if len(image_paths) == 1:
            return "single_image"
        if len(image_paths) == 2:
            return "image_pair"
        return "scene"

    @staticmethod
    def _normalize_answer_token(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        return re.sub(r"_+", "_", text)

    def _get_image_pair_camera_relation_summary(
        self,
        *,
        simulator: WorldSimulator,
        media_context: dict[str, Any],
    ) -> dict[str, Any]:
        scene_data_path = media_context.get("scene_data_path")
        image_paths = media_context.get("image_paths", []) or []
        if not isinstance(scene_data_path, str) or not scene_data_path.strip() or len(image_paths) < 2:
            return {}

        camera_pair_tool = simulator.tools.get("camera_pair_tool")
        camera_elevation_tool = simulator.tools.get("camera_elevation_tool")
        if camera_pair_tool is None or camera_elevation_tool is None:
            return {}

        try:
            pair_result = camera_pair_tool.execute(
                scene_data_path=scene_data_path,
                image_paths=image_paths[:2],
                answer_mode="position",
            )
            if not pair_result.get("success", False):
                return {}

            elevation_result = camera_elevation_tool.execute(
                camera_entity_1=pair_result.get("camera_entity_1"),
                camera_entity_2=pair_result.get("camera_entity_2"),
                camera_reference_image_idx=1,
                camera_target_image_idx=2,
            )
            if not elevation_result.get("success", False):
                return {}

            return {
                "pair_camera_elevation_answer": str(elevation_result.get("answer", "") or ""),
                "pair_camera_height_delta_m": elevation_result.get("height_delta_m"),
            }
        except Exception:
            return {}

    def _get_round1_summary_context(self, media_context: dict[str, Any]) -> dict[str, Any]:
        prompt_mode = self._infer_prompt_mode_from_media_context(media_context)
        scene_id = media_context.get("scene_id", "unknown")
        metadata_dir = media_context.get("metadata_dir")
        image_paths = media_context.get("image_paths", [])
        scene_root = media_context.get("_scene_root")

        context = {
            "prompt_mode": prompt_mode,
            "scene_id": scene_id,
            "metadata_dir": metadata_dir,
            "scene_label_counts": {},
            "scene_labels_text": "",
            "scene_unique_labels_text": "",
            "scene_non_unique_labels_text": "",
            "frame_visible_label_counts": {},
            "frame_visible_labels_text": "",
            "frame_unique_labels_text": "",
            "frame_non_unique_labels_text": "",
            "image1_visible_label_counts": {},
            "image1_visible_labels_text": "",
            "image1_unique_labels_text": "",
            "image1_label_visibility_map": {},
            "image2_visible_label_counts": {},
            "image2_visible_labels_text": "",
            "image2_unique_labels_text": "",
            "image2_label_visibility_map": {},
            "pair_union_visible_label_counts": {},
            "pair_union_visible_labels_text": "",
            "shared_visible_labels": [],
            "shared_visible_labels_text": "",
            "pair_visibility_contrast_candidates": [],
            "pair_visibility_contrast_labels": [],
            "pair_visibility_contrast_text": "",
            "pair_camera_elevation_answer": "",
            "pair_camera_height_delta_m": None,
        }
        if prompt_mode == "maze":
            # Maze contexts carry their own deterministic metadata (walls / start /
            # A-D destinations) generated together with the maze image.
            try:
                maze_meta = MazeEngine.load_metadata_for_image(image_paths[0])
            except Exception as exc:
                print(f"Failed to load maze metadata for {image_paths[0]}: {exc}")
                maze_meta = None
            if maze_meta is not None:
                context["scene_id"] = str(maze_meta.get("maze_id", "maze"))
                context["maze_grid_size"] = int(maze_meta.get("grid_size", 0))
                context["maze_metadata_path"] = MazeEngine.metadata_path_for_image(image_paths[0])
                context["maze_context_text"] = MazeEngine.build_prompt_context_text(maze_meta)
                context["maze_min_moves"] = int(maze_meta.get("suggested_min_moves", 3))
                context["maze_max_moves"] = int(maze_meta.get("suggested_max_moves", 10))
            return context
        if not isinstance(scene_id, str) or not scene_id.strip() or scene_id == "unknown":
            return context

        cache_key = (prompt_mode, scene_id, metadata_dir or "", tuple(image_paths))
        cached = self._round1_summary_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        simulator = self._get_world_simulator_for_root(scene_root)
        try:
            if prompt_mode == "single_image":
                summary = simulator.get_environment_summary(
                    {
                        "summary_type": "single_image",
                        "scene_id": scene_id,
                        "image_path": image_paths[0] if image_paths else None,
                        "metadata_dir": metadata_dir,
                    }
                )
                label_counts = self._filter_prompt_label_counts(summary.get("visible_label_counts", {}) or {})
                unique_labels, non_unique_labels = self._split_label_counts(label_counts)
                context["frame_visible_label_counts"] = label_counts
                context["frame_visible_labels_text"] = self._format_label_counts_text(label_counts)
                context["frame_unique_labels_text"] = self._format_label_list_text(unique_labels)
                context["frame_non_unique_labels_text"] = self._format_label_list_text(non_unique_labels)
            elif prompt_mode == "image_pair":
                summary = simulator.get_environment_summary(
                    {
                        "summary_type": "image_pair",
                        "scene_id": scene_id,
                        "image_paths": image_paths[:2],
                        "metadata_dir": metadata_dir,
                        "include_objects": True,
                    }
                )
                frame_summaries = summary.get("frame_summaries", []) or []
                image1_counts = self._filter_prompt_label_counts(
                    frame_summaries[0].get("visible_label_counts", {}) if len(frame_summaries) >= 1 else {}
                )
                image2_counts = self._filter_prompt_label_counts(
                    frame_summaries[1].get("visible_label_counts", {}) if len(frame_summaries) >= 2 else {}
                )
                image1_visibility_map = self._build_label_visibility_map(
                    frame_summaries[0].get("visible_objects", []) if len(frame_summaries) >= 1 else []
                )
                image2_visibility_map = self._build_label_visibility_map(
                    frame_summaries[1].get("visible_objects", []) if len(frame_summaries) >= 2 else []
                )
                image1_unique, _ = self._split_label_counts(image1_counts)
                image2_unique, _ = self._split_label_counts(image2_counts)
                visibility_contrast_candidates = self._get_pair_visibility_contrast_candidates(
                    image1_counts=image1_counts,
                    image2_counts=image2_counts,
                    image1_visibility_map=image1_visibility_map,
                    image2_visibility_map=image2_visibility_map,
                )
                aggregate = summary.get("aggregate", {}) or {}
                shared_labels = [
                    str(label).strip()
                    for label in (aggregate.get("intersection_visible_labels", []) or [])
                    if self._is_usable_prompt_label(label)
                ]
                union_counts = self._filter_prompt_label_counts(
                    aggregate.get("sum_visible_label_counts", {}) or {}
                )
                context["image1_visible_label_counts"] = image1_counts
                context["image1_visible_labels_text"] = self._format_label_counts_text(image1_counts)
                context["image1_unique_labels_text"] = self._format_label_list_text(image1_unique)
                context["image1_label_visibility_map"] = image1_visibility_map
                context["image2_visible_label_counts"] = image2_counts
                context["image2_visible_labels_text"] = self._format_label_counts_text(image2_counts)
                context["image2_unique_labels_text"] = self._format_label_list_text(image2_unique)
                context["image2_label_visibility_map"] = image2_visibility_map
                context["pair_union_visible_label_counts"] = union_counts
                context["pair_union_visible_labels_text"] = self._format_label_counts_text(union_counts)
                context["shared_visible_labels"] = shared_labels
                context["shared_visible_labels_text"] = self._format_label_list_text(shared_labels)
                context["pair_visibility_contrast_candidates"] = visibility_contrast_candidates
                context["pair_visibility_contrast_labels"] = [
                    str(item.get("label", "")).strip()
                    for item in visibility_contrast_candidates
                    if str(item.get("label", "")).strip()
                ]
                context["pair_visibility_contrast_text"] = self._format_visibility_contrast_text(
                    visibility_contrast_candidates
                )
                context.update(
                    self._get_image_pair_camera_relation_summary(
                        simulator=simulator,
                        media_context=media_context,
                    )
                )
            else:
                summary = simulator.get_environment_summary(
                    {
                        "summary_type": "multi_image",
                        "scene_id": scene_id,
                        "image_paths": image_paths,
                        "metadata_dir": metadata_dir,
                        "include_objects": True,
                    }
                )
                label_counts = self._build_observed_scene_label_counts(
                    summary.get("frame_summaries", []) or []
                )
                if not label_counts:
                    fallback_summary = simulator.get_environment_summary(
                        {
                            "summary_type": "scene",
                            "scene_id": scene_id,
                            "metadata_dir": metadata_dir,
                        }
                    )
                    label_counts = self._filter_prompt_label_counts(
                        fallback_summary.get("label_counts", {}) or {}
                    )
                unique_labels, non_unique_labels = self._split_label_counts(label_counts)
                context["scene_label_counts"] = label_counts
                context["scene_labels_text"] = self._format_label_counts_text(label_counts)
                context["scene_unique_labels_text"] = self._format_label_list_text(unique_labels)
                context["scene_non_unique_labels_text"] = self._format_label_list_text(non_unique_labels)
        except Exception:
            return context

        self._round1_summary_cache[cache_key] = dict(context)
        return dict(context)

    def _infer_feasible_round1_tasks(self, prompt_mode: str, summary_context: dict[str, Any]) -> list[str]:
        if prompt_mode == "maze":
            # Maze contexts support exactly one deterministic verifier task.
            if summary_context.get("maze_grid_size"):
                return [MAZE_TASK_TYPE]
            return []
        default_tasks = list(ROUND1_TASKS_BY_MODALITY.get(prompt_mode, []))
        if prompt_mode == "scene":
            label_counts = summary_context.get("scene_label_counts", {}) or {}
            unique_labels, non_unique_labels = self._split_label_counts(label_counts)
            num_labels = len(label_counts)
            feasible = []
            if len(non_unique_labels) >= 1:
                feasible.append("object_counting")
            if len(unique_labels) >= 2:
                feasible.append("absolute_distance")
            if len(unique_labels) >= 1:
                feasible.append("object_size")
            feasible.append("room_size")
            if len(unique_labels) >= 4:
                feasible.append("relative_distance")
            if len(unique_labels) >= 3:
                feasible.append("relative_direction_hard")
            if len(unique_labels) >= 4:
                feasible.append("route_planning")
            return feasible or default_tasks

        if prompt_mode == "single_image":
            label_counts = summary_context.get("frame_visible_label_counts", {}) or {}
            if not label_counts:
                return default_tasks
            unique_labels, _ = self._split_label_counts(label_counts)
            feasible = []
            if len(unique_labels) >= 2:
                feasible.append("single_image_relative_direction")
            if len(unique_labels) >= 1:
                feasible.append("distance_cam_obj")
            if len(unique_labels) >= 2:
                feasible.append("depth_order_obj_obj")
            return feasible

        image1_counts = summary_context.get("image1_visible_label_counts", {}) or {}
        image2_counts = summary_context.get("image2_visible_label_counts", {}) or {}
        pair_non_ambiguous_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        image1_unique, _ = self._split_label_counts(image1_counts)
        image2_unique, _ = self._split_label_counts(image2_counts)
        pair_elevation_answer = self._normalize_answer_token(
            summary_context.get("pair_camera_elevation_answer", "")
        )

        feasible = ["position_cam_cam", "motion_camera"]
        if pair_elevation_answer and pair_elevation_answer != "same_level":
            feasible.append("elevation_cam_cam")
        if len(pair_non_ambiguous_labels) >= 1:
            feasible.append("visibility_compare")
        if len(image1_unique) >= 1 or len(image2_unique) >= 1:
            feasible.append("position_cam_obj")
        if len(image1_counts) >= 1 or len(image2_counts) >= 1:
            feasible.append("position_cam_reg")
        if len(pair_non_ambiguous_labels) >= 2:
            feasible.append("attribute_measurement")
        return feasible or default_tasks

    def _build_round1_task_context_sections(
        self,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> list[dict[str, str]]:
        scene_counts = summary_context.get("scene_label_counts", {}) or {}
        scene_unique, scene_non_unique = self._split_label_counts(scene_counts)
        frame_counts = summary_context.get("frame_visible_label_counts", {}) or {}
        frame_unique, _ = self._split_label_counts(frame_counts)
        image1_counts = summary_context.get("image1_visible_label_counts", {}) or {}
        image2_counts = summary_context.get("image2_visible_label_counts", {}) or {}
        image1_unique, _ = self._split_label_counts(image1_counts)
        image2_unique, _ = self._split_label_counts(image2_counts)
        pair_union_counts = summary_context.get("pair_union_visible_label_counts", {}) or {}
        pair_non_ambiguous_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        pair_shared_unique_labels = self._get_shared_unique_pair_labels(image1_counts, image2_counts)
        pair_visibility_contrast_labels = summary_context.get("pair_visibility_contrast_labels", []) or []
        pair_visibility_contrast_text = str(summary_context.get("pair_visibility_contrast_text", "") or "").strip()

        def make_section(title: str, text: str) -> dict[str, str] | None:
            cleaned = str(text or "").strip()
            if not cleaned:
                return None
            return {"title": title, "text": cleaned}

        def format_list(labels: list[str]) -> str:
            return self._format_label_list_text(labels)

        def format_counts(label_counts: dict[str, Any]) -> str:
            return self._format_label_counts_text(label_counts)

        sections: list[dict[str, str]] = []

        if task_type == "object_counting":
            repeated_text = format_list(scene_non_unique)
            section = make_section("Observed Repeated Labels For Counting (count >= 2)", repeated_text)
            return [section] if section is not None else []

        if task_type in {"absolute_distance", "object_size", "relative_distance", "relative_direction_hard", "route_planning"}:
            section = make_section("Scene Unique Labels", format_list(scene_unique))
            return [section] if section is not None else []

        if task_type == "room_size":
            section = make_section(
                "Task-Specific Focus",
                "No object-name pool is provided for this task. Focus on room boundary, occupied edges, and open floor layout.",
            )
            return [section] if section is not None else []

        if task_type in {"single_image_relative_direction", "distance_cam_obj"}:
            primary_text = format_list(frame_unique)
            title = "Unique Visible Labels In This Image"
            if not primary_text:
                primary_text = format_counts(frame_counts)
                title = "Visible Labels In This Image"
            section = make_section(title, primary_text)
            return [section] if section is not None else []

        if task_type == "depth_order_obj_obj":
            primary_text = format_list(frame_unique)
            title = "Unique Visible Labels For Depth Comparison"
            section = make_section(title, primary_text)
            return [section] if section is not None else []

        if task_type == "elevation_cam_cam":
            elevation_answer = self._normalize_answer_token(
                summary_context.get("pair_camera_elevation_answer", "")
            )
            if elevation_answer in {"higher", "lower"}:
                focus_text = (
                    f"Current pair already shows a non-zero vertical relation: relative to Image 1, "
                    f"Image 2 is {elevation_answer}. Focus on camera height change and shared scene structure."
                )
            else:
                focus_text = (
                    "This task is only appropriate when the two images have a noticeable vertical difference. "
                    "same_level pairs are invalid."
                )
            section = make_section("Task-Specific Focus", focus_text)
            return [section] if section is not None else []

        if task_type in {"position_cam_cam", "motion_camera"}:
            section = make_section(
                "Task-Specific Focus",
                "No object-name pool is provided for this task. Focus on camera relation, framing change, and shared scene structure.",
            )
            return [section] if section is not None else []

        if task_type == "visibility_compare":
            contrast_section = make_section(
                "Strong Visibility-Contrast Labels Across The Pair",
                pair_visibility_contrast_text or format_list(pair_visibility_contrast_labels),
            )
            shared_section = make_section(
                "Shared Non-Ambiguous Labels Across Both Images",
                format_list(pair_shared_unique_labels),
            )
            pair_section = make_section(
                "Non-Ambiguous Visibility Labels Across The Pair",
                format_list(pair_non_ambiguous_labels),
            )
            sections = [section for section in (contrast_section, shared_section, pair_section) if section is not None]
            return sections

        if task_type == "position_cam_obj":
            image1_section = make_section("Unique Visible Labels In Image 1", format_list(image1_unique))
            image2_section = make_section("Unique Visible Labels In Image 2", format_list(image2_unique))
            sections = [section for section in (image1_section, image2_section) if section is not None]
            return sections

        if task_type == "position_cam_reg":
            image1_section = make_section(
                "Unique Object Anchors In Image 1",
                format_list(image1_unique),
            )
            image2_section = make_section(
                "Unique Object Anchors In Image 2",
                format_list(image2_unique),
            )
            sections = [section for section in (image1_section, image2_section) if section is not None]
            if sections:
                return sections
            fallback_section = make_section(
                "Visible Object Anchors Across The Pair",
                format_counts(pair_union_counts),
            )
            return [fallback_section] if fallback_section is not None else []

        if task_type == "attribute_measurement":
            image1_section = make_section(
                "Non-Ambiguous Labels In Image 1",
                format_list(image1_unique),
            )
            image2_section = make_section(
                "Non-Ambiguous Labels In Image 2",
                format_list(image2_unique),
            )
            pair_section = make_section(
                "Pair-Grounded Non-Ambiguous Object Labels",
                format_list(pair_non_ambiguous_labels),
            )
            sections = [section for section in (image1_section, image2_section, pair_section) if section is not None]
            return sections

        default_section = make_section("Grounded Context", format_counts(scene_counts) or format_counts(frame_counts) or format_counts(pair_union_counts))
        return [default_section] if default_section is not None else []

    def _build_round1_task_candidate_guidance(
        self,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> list[str]:
        scene_counts = summary_context.get("scene_label_counts", {}) or {}
        scene_unique, scene_non_unique = self._split_label_counts(scene_counts)
        frame_counts = summary_context.get("frame_visible_label_counts", {}) or {}
        frame_unique, _ = self._split_label_counts(frame_counts)
        image1_counts = summary_context.get("image1_visible_label_counts", {}) or {}
        image2_counts = summary_context.get("image2_visible_label_counts", {}) or {}
        pair_non_ambiguous_labels = self._get_pair_non_ambiguous_labels(image1_counts, image2_counts)
        pair_shared_unique_labels = self._get_shared_unique_pair_labels(image1_counts, image2_counts)
        pair_visibility_contrast_labels = summary_context.get("pair_visibility_contrast_labels", []) or []

        def format_list(labels: list[str]) -> str:
            return self._format_label_list_text(labels) or "None"

        default_guidance = [
            "Only use labels or regions grounded by the provided context.",
            "If a candidate looks weak, ambiguous, or awkward, replace it with a clearer grounded candidate before writing the question.",
        ]

        guidance_by_task = {
            "object_counting": [
                "Use only the counting label pool shown above for this task.",
                "Each chosen category must have observed count >= 2 in the provided scene views.",
                "Copy the category name exactly from that counting pool; do not paraphrase it, pluralize it differently, or wrap it in extra attributes.",
                "Prefer a category that is clearly distributed across the scene instead of a tiny or doubtful label.",
                "Do not ask about an absent category or any category whose answer would be 0 or 1.",
            ],
            "absolute_distance": [
                f"Allowed object pool: scene Unique labels only = {format_list(scene_unique)}.",
                "Pick two different isolated landmarks from that Unique pool; do not use repeated labels or room regions.",
                "If one chosen label feels ambiguous, replace it with another Unique label before asking the distance question.",
                "Copy the object labels exactly from the provided pool; do not add modifiers such as left, right, on the table, partially visible, or attached descriptions.",
            ],
            "object_size": [
                f"Allowed object pool: scene Unique labels only = {format_list(scene_unique)}.",
                "Prefer one prominent, well-bounded object rather than clutter or a weakly grounded label.",
                "Keep the target on one object only.",
            ],
            "room_size": [
                "Do not choose an object as the measurement target; the target must stay on the room or combined visible room area.",
                "Do not convert this task into an object-centric question just because objects are visible in the images.",
                "Avoid drifting into object size, object distance, or counting just because objects are listed in the context.",
            ],
            "relative_distance": [
                f"Use only the scene Unique-label pool shown above: {format_list(scene_unique)}.",
                "Use exactly four distinct candidates that are spatially separable around the anchor, and avoid duplicated clutter when a clearer label exists.",
                "The anchor object must not appear in the candidate list.",
            ],
            "relative_direction_hard": [
                f"Allowed standing, facing, and query pool: scene Unique labels only = {format_list(scene_unique)}.",
                "Use exactly three different labels from that Unique pool and keep the standing point, facing object, and query object all distinct.",
                "Write one single-sentence question in the canonical form: standing at/by X and facing Y, where is Z located relative to me/you.",
                "Do not use alternate forms such as look toward, look up toward, will I see, which object, or right side of the room.",
                "Copy each object label exactly from the provided Unique pool; do not add location words, attributes, or descriptive suffixes.",
                "Prefer landmarks with a stable orientation frame instead of nearly overlapping, repeated-category, or awkward triples.",
            ],
            "route_planning": [
                f"Allowed waypoint pool: scene Unique labels only = {format_list(scene_unique)}.",
                "Pick exactly four different labels from that Unique pool, in travel order: a start object, two intermediate stops, and a final destination.",
                "Copy each label exactly from the provided Unique pool; do not add locations, attributes, or descriptive suffixes.",
                "Keep the canonical wording: begin at the first object facing the second, then 'Go forward until' each next object, with a '[please fill in]' turn step between consecutive forward segments (exactly two fill-in turns).",
                "Prefer four well-separated landmarks that make the turns clearly left, right, or back; avoid nearly straight-line routes.",
            ],
            "single_image_relative_direction": [
                "Use only the visible-label pool shown above for this task, preferring Unique labels when available.",
                "Choose two different objects with a clear directional relation and avoid heavily overlapping or ambiguous pairs.",
            ],
            "distance_cam_obj": [
                "Use only the visible-label pool shown above for this task, preferring Unique labels when available.",
                "Prefer one prominent, fully grounded object instead of a tiny, truncated, or heavily occluded target.",
            ],
            "depth_order_obj_obj": [
                f"Use only the frame Unique-label pool shown above for this task: {format_list(frame_unique)}.",
                "Choose two objects that occupy clearly different depth layers and avoid pairs that appear side-by-side at nearly the same depth.",
                "Copy the labels exactly as shown; do not rewrite them into phrases such as partially visible chair or chair on the left.",
            ],
            "position_cam_cam": [
                "Do not choose an object label as the question target; the target is the camera-to-camera direction between Image 1 and Image 2.",
                "Use shared scene structure only as evidence for the camera relation.",
                "Both question directions are valid: you may ask from Image 1 to Image 2, or from Image 2 to Image 1, as long as the wording stays explicit.",
                "Keep the final question on the two camera positions, not on object direction.",
            ],
            "elevation_cam_cam": [
                "Do not choose an object label as the question target; the target is the vertical relation between the two cameras.",
                "Use shared scene structure only as evidence for height change between the two viewpoints.",
                "Only ask this task when the pair has a noticeable vertical difference; same_level pairs are invalid.",
                "Both question directions are valid: you may compare Image 2 against Image 1, or compare Image 1 against Image 2, as long as the wording and answer space stay consistent.",
                "Keep the answer space aligned with higher or lower only.",
            ],
            "visibility_compare": [
                f"Use the strong visibility-contrast label pool shown above when it is available: {format_list(pair_visibility_contrast_labels)}.",
                "If the strong visibility-contrast pool is non-empty, choose only from that pool.",
                f"Otherwise, fall back to the pair non-ambiguous label pool shown above: {format_list(pair_non_ambiguous_labels)}.",
                "Prefer a target whose visibility clearly changes between Image 1 and Image 2 instead of a label that looks equally weak in both.",
                f"If possible, prefer the shared non-ambiguous labels visible in both images: {format_list(pair_shared_unique_labels)}.",
                "Copy the label exactly as shown; do not output placeholders such as object1 or generic words such as object.",
                "Do not always write the answer candidates in one fixed order inside the question; the answer hint can express the candidate order later.",
            ],
            "position_cam_obj": [
                "Choose the image first, then use only that image's Unique-label pool shown above.",
                "Both Image 1 and Image 2 question forms are valid; do not always default to Image 1 when Image 2 has a cleaner grounded target.",
                "Choose the image first, then choose one prominent label visible in that image; avoid tiny or weakly grounded targets.",
            ],
            "position_cam_reg": [
                "Choose the image first, then use only that image's anchor-object pool shown above.",
                "Both Image 1 and Image 2 question forms are valid; prefer the image whose anchor gives the clearest grounded region instead of always defaulting to Image 1.",
                "Convert one suitable anchor object into a short grounded region phrase such as bed area, desk area, sink area, or sofa area.",
                "Avoid vague or overly broad region names that are not visually localized in the selected image.",
            ],
            "motion_camera": [
                "Do not choose an object label as the question target; the target is the camera motion from Image 1 to Image 2.",
                "Use framing change and shared scene structure only as evidence for the camera motion.",
                "Keep the final question on camera movement, not on object visibility or object direction.",
            ],
            "attribute_measurement": [
                f"Use only the pair-grounded non-ambiguous label pool shown above: {format_list(pair_non_ambiguous_labels)}.",
                "Each object may come from Image 1 or Image 2; the two objects do not need to come from the same image.",
                "Choose two rigid, clearly identifiable objects with a meaningful size contrast; avoid tiny clutter, room regions, or nearly indistinguishable pairs.",
                "Copy the labels exactly as shown; do not invent object names that are outside the provided pool.",
            ],
        }

        return guidance_by_task.get(task_type, default_guidance)

    def _get_round1_task_sampling_weight(self, task_type: str) -> float:
        stats = self._round1_task_stats.get(task_type, {"count": 0.0, "acc_sum": 0.0, "scheduler_acc_sum": 0.0})
        count = float(stats.get("count", 0.0))
        acc_sum = float(stats.get("scheduler_acc_sum", stats.get("acc_sum", 0.0)))
        if count > 0:
            smoothed_acc = (
                acc_sum + ROUND1_TASK_SAMPLING_PRIOR_ACC * ROUND1_TASK_SAMPLING_PRIOR_COUNT
            ) / (count + ROUND1_TASK_SAMPLING_PRIOR_COUNT)
        else:
            smoothed_acc = ROUND1_TASK_SAMPLING_PRIOR_ACC
        return max(ROUND1_TASK_SAMPLING_MIN_WEIGHT, 1.0 - smoothed_acc)

    def _remap_round1_scheduler_acc(self, task_type: str, mean_acc: float) -> float:
        target = ROUND1_TASK_SCHEDULER_ACC_TARGETS.get(str(task_type or "").strip())
        clipped_acc = float(np.clip(mean_acc, 0.0, 1.0))
        if target is None or target <= 0:
            return clipped_acc
        return float(np.clip(clipped_acc / float(target), 0.0, 1.0))

    def _get_round_robin_scheduled_type(self) -> str | None:
        """Look up the scheduled task type for the current global_step.

        Reads SCHEDULE_FILE (a JSON list of task type strings). Returns the
        type at position `global_step % len(schedule)`, or None if unavailable.
        """
        schedule_file = os.environ.get("SCHEDULE_FILE", "").strip()
        if not schedule_file or not os.path.exists(schedule_file):
            return None
        try:
            if not hasattr(self, "_round_robin_schedule_cache"):
                with open(schedule_file) as f:
                    self._round_robin_schedule_cache = json.load(f)
            schedule = self._round_robin_schedule_cache
            if not isinstance(schedule, list) or not schedule:
                return None
            step = int(getattr(self, "global_step", 0))
            return str(schedule[step % len(schedule)])
        except Exception:
            return None

    def _sample_round1_task_type(
        self,
        prompt_mode: str,
        summary_context: dict[str, Any],
    ) -> tuple[str, list[str], dict[str, float]]:
        # VERSE note: single-task experiments. When SPATIALEVO_FORCE_TASK_TYPE is set,
        # pin the proposer to that one task type for every prompt (overrides the
        # adaptive scheduler). Unset => original adaptive behaviour, unchanged.
        forced_task_type = os.environ.get("SPATIALEVO_FORCE_TASK_TYPE", "").strip()
        if forced_task_type:
            return forced_task_type, [forced_task_type], {forced_task_type: 1.0}

        # VERSE note: deterministic round-robin scheduler. When SCHEDULER=round_robin
        # and SCHEDULE_FILE points to a JSON list of task types, pick the type
        # for the current global_step. All contexts in the same step share the
        # same type (requirement: every step's questions are type-consistent).
        scheduler_mode = os.environ.get("SCHEDULER", "").strip().lower()
        if scheduler_mode == "round_robin":
            scheduled_type = self._get_round_robin_scheduled_type()
            if scheduled_type is not None:
                return scheduled_type, [scheduled_type], {scheduled_type: 1.0}

        feasible_tasks = self._infer_feasible_round1_tasks(prompt_mode, summary_context)
        if not feasible_tasks:
            fallback_task = {
                "scene": "room_size",
                "image_pair": "motion_camera",
                "single_image": "distance_cam_obj",
                "maze": MAZE_TASK_TYPE,
            }.get(prompt_mode, "unknown")
            return fallback_task, [fallback_task], {fallback_task: 1.0}

        weights = np.array([self._get_round1_task_sampling_weight(task) for task in feasible_tasks], dtype=np.float64)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones(len(feasible_tasks), dtype=np.float64)
        probs = weights / weights.sum()
        sampled_idx = int(self._round1_task_rng.choice(len(feasible_tasks), p=probs))
        weight_map = {task: float(prob) for task, prob in zip(feasible_tasks, probs.tolist())}
        return feasible_tasks[sampled_idx], feasible_tasks, weight_map

    def _build_round1_observation_style_example(
        self,
        task_type: str,
        prompt_mode: str,
    ) -> str:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        text = str(prompt_spec.get("example_observation", "") or "").strip()
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)
        filtered_sentences = []
        skip_markers = (
            "naturally leads to asking",
            "this naturally leads to asking",
            "this naturally supports",
            "supports either phrasing",
        )
        for sentence in sentences:
            normalized = sentence.strip().lower()
            if not normalized:
                continue
            if any(marker in normalized for marker in skip_markers):
                continue
            filtered_sentences.append(sentence.strip())

        example = " ".join(filtered_sentences).strip() or text
        if prompt_mode == "scene":
            tail = (
                " End by narrowing from the full-scene caption to the specific scene-level target and why it is valid "
                "for this task, then stop before writing the concrete question (...)."
            )
        elif prompt_mode == "image_pair":
            tail = (
                " End by narrowing from the pair-level caption to the task-relevant relation or target and why it is "
                "valid for this task, then stop before writing the concrete question (...)."
            )
        else:
            tail = (
                " End by narrowing from the whole-image caption to the task-relevant target and why it is valid for "
                "this task, then stop before writing the concrete question (...)."
            )
        return (example + tail).strip()

    @staticmethod
    def _extract_dge_question_params(text: Any) -> Optional[dict]:
        """Extract the JSON object from a <question_params>{...}</question_params> block."""
        if not isinstance(text, str):
            return None
        m = re.search(r"<question_params>\s*(\{.*?\})\s*</question_params>", text, re.DOTALL)
        if not m:
            # fallback: first {...} after the tag opening
            m2 = re.search(r"<question_params>\s*(\{.*\})", text, re.DOTALL)
            if not m2:
                return None
            blob = m2.group(1)
            # truncate to last balanced brace
            depth = 0
            end = None
            for idx, ch in enumerate(blob):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            if end is None:
                return None
            blob = blob[:end]
        else:
            blob = m.group(1)
        try:
            obj = json.loads(blob)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    # VERSE note: placeholder/garbage labels that the Proposer sometimes outputs
    # instead of real object names (template variable names, schema examples).
    _PLACEHOLDER_LABEL_PATTERNS = (
        "object_a", "object_b", "object_c", "object1", "object2", "object3",
        "object_label", "target_object", "reference_object", "anchor_object",
        "anchor object", "anchor table", "stand_object", "face_object",
        "query_object", "start_object", "second_object", "third_object",
        "destination_object", "option_a_object", "option_b_object",
        "unique - label", "unique anchor object", "unique_anchor",
        "table_anchor", "artistic composition", "panels", "null", "none",
        "<label", "<label1", "<label2", "<labeln",
    )

    @staticmethod
    def _is_placeholder_label(label: Any) -> bool:
        """Check if a label looks like a template placeholder rather than a real object."""
        if not isinstance(label, str):
            return True
        s = label.strip().lower()
        if not s:
            return True
        # angle-bracket template vars
        if s.startswith("<") and s.endswith(">"):
            return True
        # exact match against known placeholder patterns
        for pat in RayPPOTrainer._PLACEHOLDER_LABEL_PATTERNS:
            if s == pat:
                return True
        # "label1", "label2", ... generic
        import re as _re
        if _re.fullmatch(r"(label|object|item)\d+", s):
            return True
        return False

    @classmethod
    def _clean_dge_structured_params(
        cls, task_type: str, params: dict
    ) -> tuple[Optional[dict], str]:
        """Clean structured_imitate params: filter placeholders, enforce constraints.

        Returns (cleaned_params, reason). If reason is non-empty the params
        are invalid and should be marked format_score=0.
        """
        if not isinstance(params, dict) or not params:
            return None, "empty params"
        cfg = DGE_TASK_CONFIG.get(task_type, {})
        dge_to_tmpl = cfg.get("dge_to_template", {})
        cleaned = {}
        for dge_field, tmpl_var in dge_to_tmpl.items():
            val = params.get(dge_field)
            if val is None:
                return None, f"missing field `{dge_field}`"
            if isinstance(tmpl_var, list):
                # list-valued field (e.g. candidate_labels)
                if not isinstance(val, list) or not val:
                    return None, f"`{dge_field}` is not a non-empty list"
                labels = []
                seen = set()
                for v in val:
                    sv = str(v).strip()
                    # strip "(Unique)" / "(Non-Unique)" suffixes the model may copy
                    for _suf in ("(Unique)", "(Non-Unique)"):
                        if sv.endswith(_suf):
                            sv = sv[: -len(_suf)].strip()
                    # strip leading "unique " prefix (from "(Unique)" mis-copy)
                    if sv.lower().startswith("unique "):
                        sv = sv[7:].strip()
                    if cls._is_placeholder_label(sv):
                        return None, f"`{dge_field}` contains placeholder `{sv}`"
                    # dedupe duplicate candidates
                    if sv.lower() in seen:
                        continue
                    seen.add(sv.lower())
                    labels.append(sv)
                if not labels:
                    return None, f"`{dge_field}` empty after dedup"
                cleaned[dge_field] = labels
            else:
                sv = str(val).strip()
                for _suf in ("(Unique)", "(Non-Unique)"):
                    if sv.endswith(_suf):
                        sv = sv[: -len(_suf)].strip()
                if sv.lower().startswith("unique "):
                    sv = sv[7:].strip()
                if cls._is_placeholder_label(sv):
                    return None, f"`{dge_field}` is placeholder `{sv}`"
                cleaned[dge_field] = sv
        # Enforce target ∉ candidates for relative_distance
        if task_type == "relative_distance":
            target = cleaned.get("target_label")
            cands = cleaned.get("candidate_labels")
            if isinstance(target, str) and isinstance(cands, list) and target in cands:
                cands = [c for c in cands if c != target]
                if not cands:
                    return None, "target removed from candidates leaves empty"
                cleaned["candidate_labels"] = cands
            # VERSE note: VSI-Bench object_rel_distance always lists
            # exactly 4 candidate objects. Truncate extras; reject fewer so the
            # rendered text never contains an unfilled template slot.
            cands = cleaned.get("candidate_labels")
            if isinstance(cands, list):
                if len(cands) > 4:
                    cleaned["candidate_labels"] = cands[:4]
                elif len(cands) < 4:
                    return None, f"`candidate_labels` needs exactly 4 labels (got {len(cands)})"
        # VERSE note: route_planning needs exactly 4 ordered waypoints
        # (start, two stops, destination). Truncate extras; reject fewer.
        if task_type == "route_planning":
            wps = cleaned.get("waypoint_labels")
            if isinstance(wps, list):
                if len(wps) > 4:
                    cleaned["waypoint_labels"] = wps[:4]
                elif len(wps) < 4:
                    return None, f"`waypoint_labels` needs exactly 4 labels (got {len(wps)})"
        return cleaned, ""

    # VERSE note:selection-over-reward: label pools per task type for the
    # system-side resampler. Mirrors the pools shown in the Round-1 prompt
    # (_build_round1_task_candidate_guidance) and the feasibility rules in
    # _infer_feasible_round1_tasks.
    _RESAMPLE_UNSUPPORTED_REASON = "task not supported by system resampler"

    def _system_sample_dge_params(
        self,
        task_type: str,
        summary_context: dict[str, Any],
        rng: "random.Random",
        excluded_labels: Optional[set] = None,
    ) -> Optional[dict[str, Any]]:
        """Sample a label combination for `task_type` from the scene pools.

        Returns DGE-input params (same field names the structured_imitate
        Proposer emits) or None when the context cannot support the task.
        The caller re-verifies the rendered question with the DGE, so this
        only needs to be a reasonable proposal distribution, not perfect.
        `excluded_labels` lets the retry loop drop labels the DGE already
        rejected (not visible in the sampled frames / ambiguous), which is
        what makes multi-label tasks converge within few tries.
        """
        scene_counts = summary_context.get("scene_label_counts", {}) or {}
        scene_unique, scene_non_unique = self._split_label_counts(scene_counts)
        frame_counts = summary_context.get("frame_visible_label_counts", {}) or {}
        frame_unique, _ = self._split_label_counts(frame_counts)
        excluded = {str(l).strip().lower() for l in (excluded_labels or set())}

        def pick_distinct(pool: list[str], k: int) -> Optional[list[str]]:
            usable = [
                l for l in pool
                if self._is_usable_prompt_label(l) and str(l).strip().lower() not in excluded
            ]
            if len(usable) < k:
                return None
            return rng.sample(usable, k)

        if task_type == "object_counting":
            picked = pick_distinct(scene_non_unique, 1)
            return {"target_category": picked[0]} if picked else None
        if task_type == "object_size":
            picked = pick_distinct(scene_unique, 1)
            return {"object_label": picked[0]} if picked else None
        if task_type == "absolute_distance":
            picked = pick_distinct(scene_unique, 2)
            return {"object1_label": picked[0], "object2_label": picked[1]} if picked else None
        if task_type == "relative_distance":
            picked = pick_distinct(scene_unique, 5)
            if not picked:
                return None
            return {"target_label": picked[0], "candidate_labels": picked[1:]}
        if task_type == "relative_direction_hard":
            picked = pick_distinct(scene_unique, 3)
            if not picked:
                return None
            return {
                "positioning_label": picked[0],
                "orienting_label": picked[1],
                "querying_label": picked[2],
            }
        if task_type == "route_planning":
            picked = pick_distinct(scene_unique, 4)
            return {"waypoint_labels": picked} if picked else None
        if task_type == "single_image_relative_direction":
            picked = pick_distinct(frame_unique, 2)
            return {"target_label": picked[0], "reference_label": picked[1]} if picked else None
        if task_type == "distance_cam_obj":
            picked = pick_distinct(frame_unique, 1)
            return {"target_label": picked[0]} if picked else None
        if task_type == "depth_order_obj_obj":
            picked = pick_distinct(frame_unique, 2)
            return {"object1_label": picked[0], "object2_label": picked[1]} if picked else None
        # room_size has no label params: an invalid proposal means the context
        # itself failed, so resampling cannot help. SPAR/pair tasks are not
        # supported yet (not scheduled in the current runs).
        return None

    @staticmethod
    def _resample_quality_gate(
        task_type: str,
        sim_result: dict[str, Any],
        attempt: int,
        max_tries: int,
    ) -> bool:
        """Reject valid-but-distribution-shifted resampled questions.

        VERSE note: the step-50 compare eval showed route_planning
        -5.67 vs baseline — random 4-waypoint combos produce back-heavy routes
        ("turn back, turn back") while VSI-Bench GT turns are left/right
        dominated (left 140 / right 158 / back 57, no double-back in top
        patterns). Gate: never accept >=2 'back' turns; prefer 0-back routes
        in the first half of the tries, allow a single 'back' afterwards.
        """
        if task_type != "route_planning":
            return True
        answer = str(sim_result.get("answer", "") or "").lower()
        n_back = answer.count("back")
        if n_back >= 2:
            return False
        if n_back == 1 and attempt < max(1, max_tries // 2):
            return False
        return True

    def _resample_one_invalid_question(
        self,
        *,
        index: int,
        task_type: str,
        media_context: dict[str, Any],
        global_step: int,
        max_tries: int,
    ) -> Optional[dict[str, Any]]:
        """Try to replace one invalid proposal with a system-sampled valid question.

        Returns {"question_text", "params", "sim_result", "tries"} on success,
        None when the context is infeasible or all tries failed.
        """
        summary_context = self._get_round1_summary_context(media_context)
        rng = random.Random((int(global_step) * 100003 + index * 1009) & 0x7FFFFFFF)
        # VERSE note:adaptive exclusion: DGE errors name the offending
        # label ("waypoint `X` not found or ambiguous", "`Y` is not in the scene
        # unique pool", ...). Dropping that label from the pool between tries
        # turns multi-label tasks (relative_distance needs 5 labels to all pass
        # visibility/ambiguity) from ~17-30% success into fast convergence.
        excluded: set = set()
        for attempt in range(max_tries):
            params = self._system_sample_dge_params(
                task_type, summary_context, rng, excluded_labels=excluded
            )
            if params is None:
                return None
            try:
                rendered, _style, _opts, _gt = render_dge_structured_question(task_type, params)
            except Exception:
                continue
            if not rendered:
                continue
            sim_result = self._run_simulator_validation_job(
                media_context=media_context,
                canonical_task_type=task_type,
                question_text=rendered,
                use_thread_local_simulator=True,
                prefilled_params=params,
            )
            if bool(sim_result.get("is_valid", False)):
                if not self._resample_quality_gate(task_type, sim_result, attempt, max_tries):
                    continue
                return {
                    "question_text": rendered,
                    "params": params,
                    "sim_result": sim_result,
                    "tries": attempt + 1,
                }
            # Extract the offending label(s) from the DGE error and exclude them.
            error_text = str(sim_result.get("error", "") or "")
            sampled_labels = {
                str(v).strip().lower()
                for val in params.values()
                for v in (val if isinstance(val, list) else [val])
            }
            for m in re.finditer(r"[`']([^`']{1,60})[`']", error_text):
                label = m.group(1).strip().lower()
                if label in sampled_labels:
                    excluded.add(label)
        return None

    @staticmethod
    def _dge_structured_param_schema(task_type: str) -> tuple[str, str]:
        """Return (schema_description, example_params_json) for a task's <question_params>."""
        cfg = DGE_TASK_CONFIG.get(task_type, {})
        dge_to_tmpl = cfg.get("dge_to_template", {})
        fields: list[str] = []
        example: dict[str, Any] = {}
        for dge_field, tmpl_var in dge_to_tmpl.items():
            if isinstance(tmpl_var, list):
                fields.append(f'  "{dge_field}": ["<label1>", "<label2>", ...]')
                example[dge_field] = ["<label1>"] + ["<labelN>"] * (len(tmpl_var) - 1) if len(tmpl_var) > 1 else ["<label1>"]
            else:
                fields.append(f'  "{dge_field}": "<label>"')
                example[dge_field] = "<label>"
        schema_desc = "{\n" + ",\n".join(fields) + "\n}"
        return schema_desc, json.dumps(example, ensure_ascii=False)

    def _render_round1_dge_structured_prompt(
        self,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> str:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        schema_desc, example_params_json = self._dge_structured_param_schema(task_type)
        task_context_sections = self._build_round1_task_context_sections(task_type, summary_context)
        task_candidate_guidance = self._build_round1_task_candidate_guidance(task_type, summary_context)
        example_observation_style = self._build_round1_observation_style_example(
            task_type, str(summary_context.get("prompt_mode", "scene") or "scene")
        )
        prompt_text = Template(ROUND1_DGE_STRUCTURED_PROMPT_TEMPLATE).render(
            task_type=task_type,
            task_display_name=prompt_spec.get("display_name", task_type),
            task_goal=prompt_spec.get("task_goal", f"Pick object labels for `{task_type}`."),
            task_requirements=prompt_spec.get(
                "task_requirements",
                ["Pick labels exactly from the provided pools."],
            ),
            task_context_sections=task_context_sections,
            task_candidate_guidance=task_candidate_guidance,
            param_schema_description=schema_desc,
            example_observation_style=example_observation_style or "",
            example_observation=prompt_spec.get("example_observation", ""),
            example_params_json=example_params_json,
        )
        return prompt_text.strip()

    def _render_round1_task_prompt(
        self,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> str:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        prompt_mode = str(summary_context.get("prompt_mode", "scene") or "scene")
        if prompt_mode == "maze":
            if MAZE_PROPOSER_MODE in ("pick_letter", "round_robin"):
                maze_template = Template(ROUND1_MAZE_PICK_TASK_PROMPT_TEMPLATE)
            else:
                maze_template = Template(ROUND1_MAZE_TASK_PROMPT_TEMPLATE)
            return maze_template.render(**summary_context).strip()
        # Phase-2 structured_imitate: Proposer outputs <question_params> JSON only;
        # system renders the final GRPO-style question text from it.
        if PROPOSER_MODE == "structured_imitate" and task_type in DGE_TASK_CONFIG:
            return self._render_round1_dge_structured_prompt(task_type, summary_context)
        if prompt_mode == "single_image":
            template_text = ROUND1_SINGLE_IMAGE_TASK_PROMPT_TEMPLATE
        elif prompt_mode == "image_pair":
            template_text = ROUND1_IMAGE_PAIR_TASK_PROMPT_TEMPLATE
        else:
            template_text = ROUND1_SCENE_TASK_PROMPT_TEMPLATE
        prompt_template = Template(template_text)
        task_context_sections = self._build_round1_task_context_sections(task_type, summary_context)
        task_candidate_guidance = self._build_round1_task_candidate_guidance(task_type, summary_context)
        valid_question_examples = self._build_round1_valid_question_examples(task_type, summary_context)
        example_observation_style = self._build_round1_observation_style_example(task_type, prompt_mode)
        prompt_text = prompt_template.render(
            **summary_context,
            task_type=task_type,
            task_display_name=prompt_spec.get("display_name", task_type),
            task_goal=prompt_spec.get("task_goal", f"Generate one grounded question for `{task_type}`."),
            task_requirements=prompt_spec.get(
                "task_requirements",
                ["Keep the observation grounded.", "Generate exactly one question for the assigned task."],
            ),
            question_templates=prompt_spec.get("question_templates", []),
            example_observation_style=example_observation_style,
            task_context_sections=task_context_sections,
            task_candidate_guidance=task_candidate_guidance,
            valid_question_examples=valid_question_examples,
        )
        return prompt_text.strip()

    @staticmethod
    def _json_dumps_non_tensor(value: Any) -> str:
        if value is None:
            return ""
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)

    @staticmethod
    def _compact_invalid_reference_line(text: Any) -> str:
        line = str(text or "").strip()
        if not line:
            return ""

        line = re.sub(r"\bshown above\b", "for this task", line, flags=re.IGNORECASE)
        line = re.sub(r"\bprovided pool\b", "task-allowed pool", line, flags=re.IGNORECASE)

        if any(token in line.lower() for token in ["pool", "labels visible", "label pool", "object pool"]):
            if ":" in line:
                line = line.split(":", 1)[0].rstrip(" .") + "."
            elif "=" in line:
                line = line.split("=", 1)[0].rstrip(" .") + "."

        return line.strip()

    def _build_task_reference_text(
        self,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> str:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        guidance = self._build_round1_task_candidate_guidance(task_type, summary_context)
        requirements = prompt_spec.get("task_requirements", [])

        lines: list[str] = []
        merged_rules: list[str] = []
        seen_rules: set[str] = set()

        for source_items in (guidance, requirements):
            for item in source_items or []:
                cleaned = self._compact_invalid_reference_line(item)
                if not cleaned:
                    continue
                dedupe_key = re.sub(r"\s+", " ", cleaned).strip().lower()
                if dedupe_key in seen_rules:
                    continue
                seen_rules.add(dedupe_key)
                merged_rules.append(cleaned)

        if merged_rules:
            lines.append("# TASK VALIDITY RULES:")
            for item in merged_rules:
                lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines).strip()

    def _render_round2_invalid_explanation_prompt(
        self,
        *,
        question: str,
        task_type: str,
        summary_context: dict[str, Any],
    ) -> tuple[str, str]:
        prompt_spec = ROUND1_TASK_PROMPT_SPECS.get(task_type, {})
        prompt_template = Template(ROUND2_INVALID_EXPLANATION_PROMPT_TEMPLATE)
        task_reference_text = self._build_task_reference_text(task_type, summary_context)
        # Cap the (sometimes very verbose) task reference so the invalid-
        # explanation prompt + scene frames stays well within max_prompt_length.
        # Otherwise the prompt overflows, truncation cuts image placeholder tokens
        # off the front, and the Solver forward crashes with a token/feature
        # mismatch. ~6000 chars (~1500 tokens) leaves ample room for the 8 frames.
        _max_task_reference_chars = int(os.environ.get("SPATIALEVO_MAX_TASK_REF_CHARS", "6000"))
        if isinstance(task_reference_text, str) and len(task_reference_text) > _max_task_reference_chars:
            task_reference_text = task_reference_text[:_max_task_reference_chars].rstrip() + "\n... [task reference truncated to fit prompt length] ..."
        prompt_text = prompt_template.render(
            task_type=task_type,
            task_display_name=prompt_spec.get("display_name", task_type),
            question=question,
            task_reference_text=task_reference_text,
        )
        return prompt_text.strip(), task_reference_text

    @staticmethod
    def _build_round1_signature_payload(
        task_type: str,
        question: str,
        parsed_params_json: Any,
    ) -> dict[str, Any]:
        parsed_params = _load_optional_json_payload(parsed_params_json)
        if not isinstance(parsed_params, dict):
            parsed_params = {}

        def first_non_empty(*keys: str) -> str:
            for key in keys:
                value = _normalize_signature_text(parsed_params.get(key))
                if value:
                    return value
            return ""

        def sorted_labels(*keys: str) -> list[str]:
            labels: list[str] = []
            for key in keys:
                value = parsed_params.get(key)
                if isinstance(value, (list, tuple)):
                    labels.extend(_normalize_signature_text(item) for item in value)
                else:
                    normalized = _normalize_signature_text(value)
                    if normalized:
                        labels.append(normalized)
            return sorted(label for label in labels if label)

        refs = _extract_ordered_image_refs(question)
        payload: dict[str, Any] = {"task_type": str(task_type or "unknown").strip() or "unknown"}

        if task_type == "room_size":
            payload["scene_scope"] = "room"
        elif task_type == "object_counting":
            payload["target_category"] = first_non_empty("target_category", "object_label")
        elif task_type == "object_size":
            payload["object_label"] = first_non_empty("object_label", "target_label")
        elif task_type in {"absolute_distance", "attribute_measurement", "depth_order_obj_obj"}:
            payload["object_pair"] = sorted_labels("object1_label", "object2_label")
        elif task_type == "relative_distance":
            payload["anchor"] = first_non_empty("target_label", "anchor_object", "anchor_label")
            payload["candidate_labels"] = sorted_labels("candidate_labels")
        elif task_type == "relative_direction_hard":
            payload["positioning_label"] = first_non_empty("positioning_label")
            payload["orienting_label"] = first_non_empty("orienting_label")
            payload["querying_label"] = first_non_empty("querying_label")
        elif task_type == "route_planning":
            waypoints = parsed_params.get("waypoint_labels")
            if isinstance(waypoints, (list, tuple)):
                payload["waypoint_labels"] = [
                    _normalize_signature_text(w) for w in waypoints if _normalize_signature_text(w)
                ]
            else:
                payload["waypoint_labels"] = []
        elif task_type == "single_image_relative_direction":
            payload["reference_label"] = first_non_empty("reference_label")
            payload["target_label"] = first_non_empty("target_label")
        elif task_type == "distance_cam_obj":
            payload["target_label"] = first_non_empty("target_label", "object_label")
        elif task_type == "position_cam_obj":
            payload["camera_image"] = refs[0] if refs else ""
            payload["target_label"] = first_non_empty("target_label", "object_label")
        elif task_type == "position_cam_reg":
            payload["camera_image"] = refs[0] if refs else ""
            payload["region_name"] = first_non_empty("region_name")
        elif task_type == "visibility_compare":
            payload["target_label"] = first_non_empty("target_label", "object_label")
        elif task_type in {"position_cam_cam", "elevation_cam_cam"}:
            payload["reference_image"] = refs[0] if len(refs) >= 1 else ""
            payload["target_image"] = refs[1] if len(refs) >= 2 else ""
        elif task_type == "motion_camera":
            payload["pair_scope"] = "camera_motion"
        else:
            cleaned_params = {
                _normalize_signature_text(key): _normalize_signature_text(value)
                for key, value in parsed_params.items()
                if _normalize_signature_text(key) and _normalize_signature_text(value)
            }
            if cleaned_params:
                payload["parsed_params"] = cleaned_params

        compact_payload = {k: v for k, v in payload.items() if v not in ("", [], {}, None)}
        if len(compact_payload) == 1:
            compact_payload["question_fallback"] = _normalize_signature_text(question)
        return compact_payload

    def _build_round1_semantic_signature(
        self,
        *,
        task_type: str,
        question: Any,
        parsed_params_json: Any,
    ) -> str:
        payload = self._build_round1_signature_payload(
            task_type=str(task_type or "").strip() or "unknown",
            question=str(question or ""),
            parsed_params_json=parsed_params_json,
        )
        signature_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(signature_text.encode("utf-8")).hexdigest()

    def _deduplicate_round1_batch_for_round2(
        self,
        round1_batch: DataProto,
        metrics: dict[str, Any],
    ) -> DataProto:
        if len(round1_batch) == 0:
            return round1_batch

        source_uids = round1_batch.non_tensor_batch.get("uid")
        if source_uids is None:
            round1_batch.non_tensor_batch["duplicate_weight"] = np.ones(len(round1_batch), dtype=np.int32)
            return round1_batch

        question_types = round1_batch.non_tensor_batch.get("question_type")
        scheduled_task_types = round1_batch.non_tensor_batch.get("scheduled_task_type")
        questions = round1_batch.non_tensor_batch.get("question")
        parsed_params_list = round1_batch.non_tensor_batch.get("sim_parsed_params_json")

        grouped_indices: dict[str, list[int]] = defaultdict(list)
        ordered_group_ids: list[str] = []
        for idx, uid in enumerate(source_uids.tolist()):
            uid_key = str(uid)
            if uid_key not in grouped_indices:
                ordered_group_ids.append(uid_key)
            grouped_indices[uid_key].append(idx)

        kept_indices: list[int] = []
        kept_weights: list[int] = []
        kept_signatures: list[str] = []

        for uid_key in ordered_group_ids:
            seen_signatures: dict[str, int] = {}
            for idx in grouped_indices[uid_key]:
                task_type = ""
                if question_types is not None:
                    task_type = str(question_types[idx] or "").strip()
                if not task_type and scheduled_task_types is not None:
                    task_type = str(scheduled_task_types[idx] or "").strip()
                if not task_type:
                    task_type = "unknown"

                signature = self._build_round1_semantic_signature(
                    task_type=task_type,
                    question=questions[idx] if questions is not None else "",
                    parsed_params_json=parsed_params_list[idx] if parsed_params_list is not None else None,
                )
                if signature in seen_signatures:
                    kept_weights[seen_signatures[signature]] += 1
                    continue

                seen_signatures[signature] = len(kept_indices)
                kept_indices.append(idx)
                kept_weights.append(1)
                kept_signatures.append(signature)

        if len(kept_indices) == len(round1_batch):
            round1_batch.non_tensor_batch["duplicate_weight"] = np.ones(len(round1_batch), dtype=np.int32)
            round1_batch.non_tensor_batch["semantic_signature"] = np.array(kept_signatures, dtype=object)
            metrics["round2/dedup_removed_question_count"] = 0.0
            metrics["round2/dedup_removed_question_ratio"] = 0.0
            metrics["round2/unique_question_count_before_answer"] = float(len(round1_batch))
            metrics["round2/raw_question_count_before_answer"] = float(len(round1_batch))
            return round1_batch

        deduped_batch = round1_batch.index_select(np.array(kept_indices, dtype=np.int64))
        deduped_batch.non_tensor_batch["duplicate_weight"] = np.array(kept_weights, dtype=np.int32)
        deduped_batch.non_tensor_batch["semantic_signature"] = np.array(kept_signatures, dtype=object)

        removed_count = len(round1_batch) - len(deduped_batch)
        metrics["round2/dedup_removed_question_count"] = float(removed_count)
        metrics["round2/dedup_removed_question_ratio"] = float(removed_count / max(len(round1_batch), 1))
        metrics["round2/unique_question_count_before_answer"] = float(len(deduped_batch))
        metrics["round2/raw_question_count_before_answer"] = float(len(round1_batch))
        return deduped_batch

    def _get_round2_padding_requirements(self) -> tuple[int, int]:
        divisors: list[int] = []
        update_batch_sizes: list[int] = []

        world_size = int(getattr(self.actor_rollout_ref_wg, "world_size", 1) or 1)
        if world_size > 1:
            divisors.append(world_size)

        rollout_repeat = int(getattr(self.config.worker.rollout, "n", 1) or 1)

        actor_batch_size = int(getattr(self.config.worker.actor, "global_batch_size", 0) or 0)
        if actor_batch_size > 0:
            effective_actor_batch_size = actor_batch_size * max(1, rollout_repeat)
            divisors.append(effective_actor_batch_size)
            update_batch_sizes.append(effective_actor_batch_size)

        if self.use_critic:
            critic_batch_size = int(getattr(self.config.worker.critic, "global_batch_size", 0) or 0)
            if critic_batch_size > 0:
                effective_critic_batch_size = critic_batch_size * max(1, rollout_repeat)
                divisors.append(effective_critic_batch_size)
                update_batch_sizes.append(effective_critic_batch_size)

        if not divisors:
            return 1, max(1, self.config.worker.rollout.n)

        target_divisor = divisors[0]
        for value in divisors[1:]:
            target_divisor = math.lcm(target_divisor, value)

        if update_batch_sizes:
            padding_chunk = update_batch_sizes[0]
            for value in update_batch_sizes[1:]:
                padding_chunk = math.gcd(padding_chunk, value)
        else:
            padding_chunk = target_divisor

        return max(1, target_divisor), max(1, padding_chunk)

    def _pad_round2_batch_for_balance(
        self,
        round2_batch: DataProto,
        metrics: dict[str, Any],
    ) -> DataProto:
        world_size = int(getattr(self.actor_rollout_ref_wg, "world_size", 1) or 1)
        batch_size = len(round2_batch)
        if world_size <= 1 or batch_size == 0 or batch_size % world_size == 0:
            metrics["round2/balance_pad_group_count"] = 0.0
            metrics["round2/balance_pad_sample_count"] = 0.0
            return round2_batch

        if "is_padding" not in round2_batch.non_tensor_batch:
            round2_batch.non_tensor_batch["is_padding"] = np.zeros(batch_size, dtype=bool)

        group_size = int(self.config.worker.rollout.n)
        if group_size <= 0 or batch_size % group_size != 0:
            return round2_batch

        pad_group_count = 0
        while ((batch_size + pad_group_count * group_size) % world_size) != 0:
            pad_group_count += 1
            if pad_group_count > world_size:
                return round2_batch

        if pad_group_count == 0:
            metrics["round2/balance_pad_group_count"] = 0.0
            metrics["round2/balance_pad_sample_count"] = 0.0
            return round2_batch

        uid_values = round2_batch.non_tensor_batch.get("uid")
        if uid_values is None:
            return round2_batch

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        ordered_uids: list[str] = []
        for idx, uid in enumerate(uid_values.tolist()):
            uid_key = str(uid)
            if uid_key not in uid_to_indices:
                ordered_uids.append(uid_key)
            uid_to_indices[uid_key].append(idx)

        source_groups = [uid_to_indices[uid_key] for uid_key in ordered_uids if len(uid_to_indices[uid_key]) == group_size]
        if not source_groups:
            return round2_batch

        pad_groups: list[DataProto] = []
        for pad_idx in range(pad_group_count):
            source_indices = source_groups[pad_idx % len(source_groups)]
            pad_group = round2_batch.index_select(np.array(source_indices, dtype=np.int64))
            pad_size = len(pad_group)
            pad_group.non_tensor_batch["uid"] = np.array(
                [f"round2-balance-pad-{pad_idx}"] * pad_size,
                dtype=object,
            )
            pad_group.non_tensor_batch["is_padding"] = np.ones(pad_size, dtype=bool)
            pad_group.non_tensor_batch["duplicate_weight"] = np.zeros(pad_size, dtype=np.int32)
            pad_groups.append(pad_group)

        pad_batch = DataProto.concat(pad_groups)
        metrics["round2/balance_pad_group_count"] = float(pad_group_count)
        metrics["round2/balance_pad_sample_count"] = float(len(pad_batch))
        return DataProto.concat([round2_batch, pad_batch])

    @staticmethod
    def _interleave_round2_padding(
        real_batch: DataProto,
        pad_batch: DataProto,
        chunk_size: int,
    ) -> DataProto:
        combined = DataProto.concat([real_batch, pad_batch])
        if len(pad_batch) == 0 or chunk_size <= 0:
            return combined

        total_size = len(combined)
        if total_size % chunk_size != 0:
            return combined

        num_real = len(real_batch)
        num_chunks = total_size // chunk_size
        if num_real < num_chunks:
            return combined

        real_indices = list(range(num_real))
        pad_indices = list(range(num_real, total_size))
        ordered_indices: list[int] = []

        base_real = num_real // num_chunks
        extra_real = num_real % num_chunks
        real_counts = [base_real + (1 if chunk_idx < extra_real else 0) for chunk_idx in range(num_chunks)]
        if any(count <= 0 or count > chunk_size for count in real_counts):
            return combined

        for real_count in real_counts:
            pad_count = chunk_size - real_count
            if real_count > len(real_indices) or pad_count > len(pad_indices):
                return combined
            current_chunk = [real_indices.pop(0) for _ in range(real_count)]
            current_chunk.extend(pad_indices.pop(0) for _ in range(pad_count))
            ordered_indices.extend(current_chunk)

        if real_indices or pad_indices:
            return combined
        return combined.index_select(np.array(ordered_indices, dtype=np.int64))

    def _pad_round2_batch_for_updates(
        self,
        round2_batch: DataProto,
        metrics: dict[str, Any],
    ) -> DataProto:
        batch_size = len(round2_batch)
        if batch_size == 0:
            return round2_batch

        if "is_padding" not in round2_batch.non_tensor_batch:
            round2_batch.non_tensor_batch["is_padding"] = np.zeros(batch_size, dtype=bool)

        existing_padding_mask = np.asarray(round2_batch.non_tensor_batch["is_padding"], dtype=bool)
        existing_pad_batch = None
        if existing_padding_mask.any():
            existing_pad_indices = np.flatnonzero(existing_padding_mask).astype(np.int64)
            real_indices = np.flatnonzero(~existing_padding_mask).astype(np.int64)
            padding_tensor_mask = torch.from_numpy(existing_padding_mask)
            for field_name in ("token_level_scores", "token_level_rewards", "old_log_probs", "ref_log_probs", "values"):
                if field_name in round2_batch.batch:
                    field = round2_batch.batch[field_name]
                    round2_batch.batch[field_name][padding_tensor_mask.to(field.device)] = 0
            if "response_mask" in round2_batch.batch:
                response_mask = round2_batch.batch["response_mask"]
                round2_batch.batch["response_mask"][padding_tensor_mask.to(response_mask.device)] = 0
            existing_pad_batch = round2_batch.index_select(existing_pad_indices)
            real_batch = round2_batch.index_select(real_indices)
        else:
            real_batch = round2_batch

        target_divisor, padding_chunk = self._get_round2_padding_requirements()
        group_size = int(self.config.worker.rollout.n)
        if group_size <= 0:
            return round2_batch
        if len(real_batch) == 0 or batch_size % group_size != 0:
            return round2_batch

        pad_group_count = 0
        while ((batch_size + pad_group_count * group_size) % target_divisor) != 0:
            pad_group_count += 1
            if pad_group_count > target_divisor:
                return round2_batch

        if pad_group_count == 0:
            if existing_pad_batch is not None and len(existing_pad_batch) > 0:
                padded_batch = self._interleave_round2_padding(real_batch, existing_pad_batch, chunk_size=padding_chunk)
                metrics["round2/pad_group_count"] = 0.0
                metrics["round2/pad_sample_count"] = float(len(existing_pad_batch))
                metrics["round2/padded_batch_size"] = float(len(padded_batch))
                return padded_batch
            metrics["round2/pad_group_count"] = 0.0
            metrics["round2/pad_sample_count"] = 0.0
            return round2_batch

        uid_values = real_batch.non_tensor_batch.get("uid")
        if uid_values is None:
            return round2_batch

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        ordered_uids: list[str] = []
        for idx, uid in enumerate(uid_values.tolist()):
            uid_key = str(uid)
            if uid_key not in uid_to_indices:
                ordered_uids.append(uid_key)
            uid_to_indices[uid_key].append(idx)

        source_groups = [uid_to_indices[uid_key] for uid_key in ordered_uids if len(uid_to_indices[uid_key]) == group_size]
        if not source_groups:
            return round2_batch

        pad_groups: list[DataProto] = []
        for pad_idx in range(pad_group_count):
            source_indices = source_groups[pad_idx % len(source_groups)]
            pad_group = real_batch.index_select(np.array(source_indices, dtype=np.int64))
            pad_size = len(pad_group)

            pad_group.non_tensor_batch["uid"] = np.array(
                [f"round2-pad-{pad_idx}"] * pad_size,
                dtype=object,
            )
            pad_group.non_tensor_batch["is_padding"] = np.ones(pad_size, dtype=bool)
            pad_group.non_tensor_batch["duplicate_weight"] = np.zeros(pad_size, dtype=np.int32)

            for field_name in ("token_level_scores", "token_level_rewards", "old_log_probs", "ref_log_probs", "values"):
                if field_name in pad_group.batch:
                    pad_group.batch[field_name] = torch.zeros_like(pad_group.batch[field_name])
            if "response_mask" in pad_group.batch:
                pad_group.batch["response_mask"] = torch.zeros_like(pad_group.batch["response_mask"])

            pad_groups.append(pad_group)

        pad_batches: list[DataProto] = []
        if existing_pad_batch is not None and len(existing_pad_batch) > 0:
            pad_batches.append(existing_pad_batch)
        if pad_groups:
            pad_batches.append(DataProto.concat(pad_groups))
        if not pad_batches:
            metrics["round2/pad_group_count"] = 0.0
            metrics["round2/pad_sample_count"] = float(existing_padding_mask.sum())
            return round2_batch

        pad_batch = pad_batches[0] if len(pad_batches) == 1 else DataProto.concat(pad_batches)
        padded_batch = self._interleave_round2_padding(real_batch, pad_batch, chunk_size=padding_chunk)
        metrics["round2/pad_group_count"] = float(pad_group_count)
        metrics["round2/pad_sample_count"] = float(len(pad_batch))
        metrics["round2/padded_batch_size"] = float(len(padded_batch))
        return padded_batch

    @staticmethod
    def _filter_reward_metrics_for_padding(
        reward_metrics: dict[str, Any],
        padding_mask: np.ndarray,
    ) -> dict[str, Any]:
        if padding_mask.size == 0 or not bool(padding_mask.any()):
            return reward_metrics

        keep_mask = ~padding_mask
        filtered: dict[str, Any] = {}
        for key, value in reward_metrics.items():
            if isinstance(value, np.ndarray) and len(value) == len(padding_mask):
                filtered[key] = value[keep_mask].tolist()
            elif isinstance(value, (list, tuple)) and len(value) == len(padding_mask):
                filtered[key] = [item for item, keep in zip(value, keep_mask.tolist()) if keep]
            else:
                filtered[key] = value
        return filtered

    @staticmethod
    def _strip_padding_for_metrics(batch: DataProto) -> DataProto:
        if "is_padding" not in batch.non_tensor_batch:
            return batch

        padding_mask = np.asarray(batch.non_tensor_batch["is_padding"], dtype=bool)
        if padding_mask.size == 0 or not bool(padding_mask.any()):
            return batch

        keep_indices = np.flatnonzero(~padding_mask).astype(np.int64)
        if keep_indices.size == 0:
            return batch
        return batch.index_select(keep_indices)

    def _get_proposer_openai_client(self) -> OpenAI:
        if self._proposer_openai_client is None:
            self._proposer_openai_client = OpenAI(
                api_key=self._proposer_openai_api_key,
                base_url=self._proposer_openai_base_url,
                timeout=self._proposer_openai_timeout,
            )
        return self._proposer_openai_client

    def _call_frozen_proposer_once(self, prompt_text: str, image_paths: list[str]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_path_to_data_url(image_path)},
                }
            )

        last_exc: Exception | None = None
        for attempt in range(max(1, self._proposer_openai_max_retries)):
            try:
                completion = self._get_proposer_openai_client().chat.completions.create(
                    model=self._proposer_openai_model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self._proposer_openai_temperature,
                    top_p=self._proposer_openai_top_p,
                    max_tokens=int(self.config.data.max_response_length),
                )
                text = _openai_message_text(completion)
                if text:
                    return text
                raise RuntimeError("empty proposer response")
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= max(1, self._proposer_openai_max_retries):
                    break
                time.sleep(min(60.0, 2.0 * (attempt + 1)))
        raise RuntimeError(f"Frozen proposer API failed after retries: {last_exc}")

    def _generate_round1_with_frozen_proposer(self, prompts: DataProto) -> DataProto:
        if "round1_prompt_text" not in prompts.non_tensor_batch:
            raise RuntimeError("Frozen proposer generation requires `round1_prompt_text` in Round1 prompts.")

        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        batch_size = input_ids.size(0)
        repeat_times = int(self.config.worker.rollout.n)

        prompt_texts = [str(item or "") for item in prompts.non_tensor_batch["round1_prompt_text"].tolist()]
        multi_modal_data = prompts.non_tensor_batch.get("multi_modal_data")

        def _images_for(idx: int) -> list[str]:
            if multi_modal_data is None:
                return []
            item = multi_modal_data[idx]
            if not isinstance(item, dict):
                return []
            images = item.get("images") or []
            if isinstance(images, (list, tuple)):
                return [str(path) for path in images]
            return [str(images)] if images else []

        jobs: list[tuple[int, int, str, list[str]]] = []
        for prompt_idx, prompt_text in enumerate(prompt_texts):
            images = _images_for(prompt_idx)
            for sample_idx in range(repeat_times):
                jobs.append((prompt_idx, sample_idx, prompt_text, images))

        def _run_job(job: tuple[int, int, str, list[str]]) -> str:
            _prompt_idx, _sample_idx, prompt_text, images = job
            return self._call_frozen_proposer_once(prompt_text, images)

        max_workers = max(1, min(self._proposer_openai_max_workers, len(jobs)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            responses_text = list(executor.map(_run_job, jobs))

        response_token_ids = [
            self.tokenizer.encode(text, add_special_tokens=False)[: int(self.config.data.max_response_length)]
            for text in responses_text
        ]
        response_ids = VF.pad_2d_list_to_length(
            response_token_ids,
            self.tokenizer.pad_token_id,
            max_length=int(self.config.data.max_response_length),
        ).to(input_ids.device)

        if repeat_times > 1:
            input_ids = input_ids.repeat_interleave(repeat_times, dim=0)
            attention_mask = attention_mask.repeat_interleave(repeat_times, dim=0)
            position_ids = position_ids.repeat_interleave(repeat_times, dim=0)
            if multi_modal_data is not None:
                multi_modal_data = np.repeat(multi_modal_data, repeat_times, axis=0)

        response_length = response_ids.size(1)
        repeated_batch_size = batch_size * repeat_times
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(repeated_batch_size, -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(repeated_batch_size, 1, -1).expand(
                repeated_batch_size,
                position_ids.size(1),
                -1,
            )
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        eos_token_id = prompts.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        response_mask = VF.get_response_mask(
            response_ids=response_ids,
            eos_token_id=eos_token_id,
            dtype=attention_mask.dtype,
        )
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        non_tensors = {}
        if multi_modal_data is not None:
            non_tensors["multi_modal_data"] = multi_modal_data
        return DataProto.from_dict(
            tensors={
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            non_tensors=non_tensors,
            meta_info=prompts.meta_info,
        )

    def _build_round1_task_conditioned_batch(self, original_batch: DataProto) -> Optional[DataProto]:
        multi_modal_data = original_batch.non_tensor_batch.get("multi_modal_data", None)
        uid_array = original_batch.non_tensor_batch.get("uid")

        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        raw_prompt_ids_list = []
        new_multi_modal_data = []
        scheduled_task_types = []
        prompt_modes = []
        scene_ids = []
        feasible_task_lists = []
        sampling_prob_lists = []
        kept_uids = []
        round1_prompt_texts = []

        for i in range(len(original_batch)):
            image_data = multi_modal_data[i] if multi_modal_data is not None else None
            media_context = self._build_simulator_media_context(image_data)
            summary_context = self._get_round1_summary_context(media_context)
            prompt_mode = summary_context.get("prompt_mode", "scene")
            feasible_tasks = self._infer_feasible_round1_tasks(prompt_mode, summary_context)

            if (
                prompt_mode == "single_image"
                and summary_context.get("frame_visible_label_counts")
                and not feasible_tasks
            ):
                continue

            scheduled_task_type, feasible_tasks, sampling_probs = self._sample_round1_task_type(
                prompt_mode=prompt_mode,
                summary_context=summary_context,
            )

            # VERSE note: round-robin scheduler modality guard. The deterministic schedule
            # pins one task type per step; skip contexts whose media modality cannot
            # serve that task (e.g. a maze context scheduled for a SPAR task). This
            # keeps every step's questions type-consistent without crashing the DGE.
            scheduler_mode = os.environ.get("SCHEDULER", "").strip().lower()
            if scheduler_mode == "round_robin":
                required_mode = TASK_TYPE_TO_PROMPT_MODE.get(scheduled_task_type)
                if required_mode and prompt_mode != required_mode:
                    continue

            prompt_text = self._render_round1_task_prompt(
                task_type=scheduled_task_type,
                summary_context=summary_context,
            )

            if image_data is not None and isinstance(image_data, dict) and "images" in image_data:
                images = image_data["images"]
                messages = self._build_answer_messages_with_images(prompt_text, len(images))
                prompt_formatted = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                processed_images = [
                    process_image(img, self.config.data.min_pixels, self.config.data.max_pixels)
                    for img in images
                ]
                model_inputs = self.processor(
                    processed_images,
                    [prompt_formatted],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                new_multi_modal_data.append(
                    {
                        "images": images,
                        "_round1_summary_context": dict(summary_context),
                    }
                )
            else:
                messages = [{"role": "user", "content": prompt_text}]
                prompt_formatted = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                model_inputs = self.tokenizer(
                    [prompt_formatted], add_special_tokens=False, return_tensors="pt"
                )
                new_multi_modal_data.append(
                    {"_round1_summary_context": dict(summary_context)}
                )

            input_ids = model_inputs["input_ids"][0]
            attention_mask = model_inputs["attention_mask"][0]

            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from ..models.transformers.qwen3_vl import get_rope_index
                else:
                    from ..models.transformers.qwen2_vl import get_rope_index

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                    attention_mask=attention_mask,
                )
                text_position_ids = torch.arange(len(input_ids), device=input_ids.device).unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)

            input_ids, attention_mask, position_ids = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",
            )

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            raw_prompt_ids_list.append(self.tokenizer.encode(prompt_formatted, add_special_tokens=False))
            scheduled_task_types.append(scheduled_task_type)
            prompt_modes.append(prompt_mode)
            scene_ids.append(summary_context.get("scene_id", "unknown"))
            feasible_task_lists.append(",".join(feasible_tasks))
            sampling_prob_lists.append(
                json.dumps(sampling_probs, ensure_ascii=False, sort_keys=True)
            )
            kept_uids.append(uid_array[i])
            round1_prompt_texts.append(prompt_text)

        if not input_ids_list:
            return None

        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "position_ids": torch.stack(position_ids_list),
            },
            non_tensors={
                "raw_prompt_ids": np.array(raw_prompt_ids_list, dtype=object),
                "multi_modal_data": np.array(new_multi_modal_data, dtype=object),
                "uid": np.array(kept_uids, dtype=object),
                "scheduled_task_type": np.array(scheduled_task_types, dtype=object),
                "prompt_mode": np.array(prompt_modes, dtype=object),
                "scene_id": np.array(scene_ids, dtype=object),
                "feasible_task_types": np.array(feasible_task_lists, dtype=object),
                "task_sampling_probs": np.array(sampling_prob_lists, dtype=object),
                "round1_prompt_text": np.array(round1_prompt_texts, dtype=object),
            },
            meta_info={
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            },
        )
        return batch

    def _update_round1_task_stats(self, round2_batch: DataProto, reward_metrics_round2: dict[str, Any]) -> None:
        accuracies = reward_metrics_round2.get("accuracy")
        if not isinstance(accuracies, list) or not accuracies:
            return
        if "question_type" not in round2_batch.non_tensor_batch or "uid" not in round2_batch.non_tensor_batch:
            return

        question_types = round2_batch.non_tensor_batch["question_type"].tolist()
        uids = round2_batch.non_tensor_batch["uid"].tolist()
        if "is_valid" in round2_batch.non_tensor_batch:
            valid_flags = np.asarray(round2_batch.non_tensor_batch["is_valid"], dtype=bool).tolist()
        else:
            valid_flags = [True] * len(question_types)
        if "duplicate_weight" in round2_batch.non_tensor_batch:
            duplicate_weights = np.asarray(round2_batch.non_tensor_batch["duplicate_weight"], dtype=np.float32).tolist()
        else:
            duplicate_weights = [1.0] * len(question_types)
        if "is_padding" in round2_batch.non_tensor_batch:
            padding_flags = np.asarray(round2_batch.non_tensor_batch["is_padding"], dtype=bool).tolist()
        else:
            padding_flags = [False] * len(question_types)

        if not (
            len(question_types)
            == len(uids)
            == len(valid_flags)
            == len(accuracies)
            == len(duplicate_weights)
            == len(padding_flags)
        ):
            return

        uid_to_task = {}
        uid_to_accs = defaultdict(list)
        uid_to_weight = {}

        for uid, task_type, is_valid, accuracy, duplicate_weight, is_padding in zip(
            uids, question_types, valid_flags, accuracies, duplicate_weights, padding_flags
        ):
            if bool(is_padding):
                continue
            if uid not in uid_to_task:
                uid_to_task[uid] = str(task_type or "").strip()
                uid_to_weight[uid] = max(1.0, float(duplicate_weight))
            if not bool(is_valid):
                uid_to_accs[uid].append(0.0)
                continue
            if accuracy is None:
                continue
            uid_to_accs[uid].append(float(accuracy))

        for uid, task_type in uid_to_task.items():
            if not task_type or task_type == "unknown":
                continue
            acc_values = uid_to_accs.get(uid, [])
            if not acc_values:
                continue
            mean_acc = float(np.mean(acc_values))
            scheduler_acc = self._remap_round1_scheduler_acc(task_type, mean_acc)
            weight = float(uid_to_weight.get(uid, 1.0))
            self._round1_task_stats[task_type]["count"] += weight
            self._round1_task_stats[task_type]["acc_sum"] += mean_acc * weight
            self._round1_task_stats[task_type]["scheduler_acc_sum"] += scheduler_acc * weight

    def _add_round1_task_scheduler_metrics(self, metrics: dict[str, Any]) -> None:
        for task_names in ROUND1_TASKS_BY_MODALITY.values():
            for task_name in task_names:
                stats = self._round1_task_stats.get(
                    task_name,
                    {"count": 0.0, "acc_sum": 0.0, "scheduler_acc_sum": 0.0},
                )
                count = float(stats.get("count", 0.0))
                mean_acc = float(stats.get("acc_sum", 0.0) / count) if count > 0 else 0.0
                scheduler_mean_acc = float(stats.get("scheduler_acc_sum", 0.0) / count) if count > 0 else 0.0
                weight = self._get_round1_task_sampling_weight(task_name)
                metrics[f"round1_scheduler/history_count/{task_name}"] = count
                metrics[f"round1_scheduler/history_acc/{task_name}"] = mean_acc
                metrics[f"round1_scheduler/history_direct_acc/{task_name}"] = mean_acc
                metrics[f"round1_scheduler/history_scheduler_acc/{task_name}"] = scheduler_mean_acc
                metrics[f"round1_scheduler/sampling_weight/{task_name}"] = weight

    def _log_generation_io(self, inputs: DataProto, outputs: DataProto, scores: torch.Tensor, step: int, tag: str = "train"):
        # VERSE note: only sample full interaction logs every N steps (default 100) + step 1,
        # to keep io_log.jsonl small and act as a periodic interaction sampler.
        _sample_freq = int(os.environ.get("SPATIALEVO_SAMPLE_LOG_FREQ", "100"))
        if not (step % _sample_freq == 0 or step == 1):
            return
        try:
            def _to_python_value(value):
                if isinstance(value, np.generic):
                    return value.item()
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        return value.item()
                    return value.detach().cpu().tolist()
                return value

            def _extract_optional_field(field_name: str, default=None):
                source = None
                if field_name in outputs.non_tensor_batch:
                    source = outputs.non_tensor_batch[field_name]
                elif field_name in inputs.non_tensor_batch:
                    source = inputs.non_tensor_batch[field_name]
                else:
                    return [default] * len(prompts)

                if isinstance(source, np.ndarray):
                    values = source.tolist()
                elif torch.is_tensor(source):
                    values = source.detach().cpu().tolist()
                elif isinstance(source, (list, tuple)):
                    values = list(source)
                else:
                    values = [source] * len(prompts)

                values = [_to_python_value(v) for v in values]
                if len(values) < len(prompts):
                    values.extend([default] * (len(prompts) - len(values)))
                elif len(values) > len(prompts):
                    values = values[: len(prompts)]
                return values

            log_dir = os.path.join(self.config.trainer.save_checkpoint_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "io_log.jsonl")

            if torch.is_tensor(scores):
                if scores.dim() > 1:
                    sample_scores = scores.sum(dim=-1).cpu().tolist()
                else:
                    sample_scores = scores.cpu().tolist()
            else:
                sample_scores = scores

            input_ids = inputs.batch["input_ids"]
            attention_mask = inputs.batch["attention_mask"]
            prompts = []
            for i in range(len(input_ids)):
                valid_ids = input_ids[i][attention_mask[i] == 1]
                p_text = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
                prompts.append(p_text)

            responses = self._decode_responses_from_batch(outputs)
            
            images = []
            if "multi_modal_data" in inputs.non_tensor_batch:
                mm_data = inputs.non_tensor_batch["multi_modal_data"]
                images = [item.get("images") if isinstance(item, dict) else None for item in mm_data]
            else:
                images = [None] * len(prompts)

            question_types = _extract_optional_field("question_type", "")
            question_types_raw = _extract_optional_field("question_type_raw", "")
            questions = _extract_optional_field("question", "")
            ground_truths = _extract_optional_field("gt", "")
            validity_flags = _extract_optional_field("is_valid", None)
            question_uids = _extract_optional_field("question_uid", "")
            sim_errors = _extract_optional_field("sim_error", "")
            sim_error_codes = _extract_optional_field("sim_error_code", "")
            sim_failure_stages = _extract_optional_field("sim_failure_stage", "")
            round2_modes = _extract_optional_field("round2_mode", "")
            padding_flags = _extract_optional_field("is_padding", False)

            with open(log_file, "a", encoding="utf-8") as f:
                for p, r, img, s, q, q_type, q_type_raw, gt, is_valid, q_uid, sim_error, sim_error_code, sim_failure_stage, round2_mode, is_padding in zip(
                    prompts,
                    responses,
                    images,
                    sample_scores,
                    questions,
                    question_types,
                    question_types_raw,
                    ground_truths,
                    validity_flags,
                    question_uids,
                    sim_errors,
                    sim_error_codes,
                    sim_failure_stages,
                    round2_modes,
                    padding_flags,
                ):
                    if bool(is_padding):
                        continue
                    log_entry = {
                        "step": step,
                        "tag": tag,
                        "reward": round(float(s), 4),  # Keep logs compact.
                        "image": img,
                        "prompt": p,
                        "response": r,
                        "question": q,
                        "question_type": q_type,
                        "question_type_raw": q_type_raw,
                        "gt": gt,
                        "is_valid": is_valid,
                        "sim_error": sim_error,
                        "sim_error_code": sim_error_code,
                        "sim_failure_stage": sim_failure_stage,
                        "round2_mode": round2_mode,
                        "question_uid": q_uid,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Warning] Failed to log IO with rewards: {e}")

    def _append_trajectory_bank(
        self,
        round2_batch: DataProto,
        reward_metrics_round2: dict[str, Any],
        *,
        step: int,
    ) -> int:
        """VERSE note:verified trajectory bank (cheap train-time-search).

        Persist every valid + format-correct + high-accuracy solver rollout
        (maze additionally gated on trajectory fidelity) to
        `<ckpt>/logs/trajectory_bank.jsonl`. These are later distilled offline
        via rejection-sampling SFT (see run_scripts/export_traj_bank_sft.py).
        Unlike round2_valid_qa.jsonl (sampled every N steps), the bank keeps
        every qualifying rollout of every step. Returns the number saved.
        """
        if not self._traj_bank_enabled or len(round2_batch) == 0:
            return 0
        try:
            bs = len(round2_batch)
            accs = reward_metrics_round2.get("accuracy", [])
            if not isinstance(accs, list) or len(accs) != bs:
                return 0
            fmts = reward_metrics_round2.get("format", [])
            if not isinstance(fmts, list) or len(fmts) != bs:
                fmts = [None] * bs
            fids = reward_metrics_round2.get("maze_traj_fidelity", [])
            if not isinstance(fids, list) or len(fids) != bs:
                fids = [None] * bs
            groundings = reward_metrics_round2.get("vsi_grounding", [])
            if not isinstance(groundings, list) or len(groundings) != bs:
                groundings = [None] * bs

            ntb = round2_batch.non_tensor_batch
            validity_flags = ntb.get("is_valid")
            padding_flags = ntb.get("is_padding")
            question_types = ntb.get("question_type")
            questions = ntb.get("question")
            ground_truths = ntb.get("gt")
            gt_letters = ntb.get("gt_letter")
            mc_options = ntb.get("mc_options_json")
            answer_styles = ntb.get("answer_style")
            sequence_texts = ntb.get("agentic_sequence_text")
            response_texts = ntb.get("agentic_response_text")
            scene_ids = ntb.get("scene_id")
            multi_modal_data = ntb.get("multi_modal_data")
            question_sources = ntb.get("question_source")
            if validity_flags is None or questions is None or sequence_texts is None:
                return 0

            log_dir = os.path.join(self.config.trainer.save_checkpoint_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            bank_file = os.path.join(log_dir, "trajectory_bank.jsonl")

            saved = 0
            with open(bank_file, "a", encoding="utf-8") as f:
                for idx in range(bs):
                    if padding_flags is not None and bool(padding_flags[idx]):
                        continue
                    if not bool(validity_flags[idx]):
                        continue
                    try:
                        acc = float(accs[idx])
                    except (TypeError, ValueError):
                        continue
                    if acc < self._traj_bank_acc_threshold:
                        continue
                    fmt = fmts[idx]
                    if fmt is not None and float(fmt) < 1.0:
                        continue
                    task_type = str(question_types[idx]) if question_types is not None else ""
                    fid = fids[idx]
                    if task_type == MAZE_TASK_TYPE and fid is not None:
                        if float(fid) < self._traj_bank_fidelity_threshold:
                            continue
                    image_paths = self._extract_media_paths_from_item(
                        multi_modal_data[idx] if multi_modal_data is not None else None
                    )
                    row = {
                        "step": int(step),
                        "task_type": task_type,
                        "question": str(questions[idx]),
                        "gt": str(ground_truths[idx]) if ground_truths is not None else "",
                        "gt_letter": str(gt_letters[idx]) if gt_letters is not None else "",
                        "mc_options_json": str(mc_options[idx]) if mc_options is not None else "",
                        "answer_style": str(answer_styles[idx]) if answer_styles is not None else "",
                        "scene_id": str(scene_ids[idx]) if scene_ids is not None else "",
                        "images": image_paths,
                        "accuracy": acc,
                        "format": float(fmt) if fmt is not None else None,
                        "maze_traj_fidelity": float(fid) if fid is not None else None,
                        "vsi_grounding": (
                            float(groundings[idx]) if idx < len(groundings) and groundings[idx] is not None else None
                        ),
                        "question_source": str(question_sources[idx]) if question_sources is not None else "",
                        "sequence_text": str(sequence_texts[idx]),
                        "response_text": str(response_texts[idx]) if response_texts is not None else "",
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    saved += 1
            return saved
        except Exception as exc:
            print(f"[traj-bank] failed to append: {exc}")
            return 0

    def _log_round2_valid_qa(
        self,
        round2_batch: DataProto,
        reward_metrics_round2: dict[str, Any],
        *,
        step: int,
    ) -> None:
        try:
            # VERSE note: only sample interaction logs every N steps (default 100) + step 1.
            _sample_freq = int(os.environ.get("SPATIALEVO_SAMPLE_LOG_FREQ", "100"))
            if not (step % _sample_freq == 0 or step == 1):
                return
            if len(round2_batch) == 0:
                return
            if "uid" not in round2_batch.non_tensor_batch or "is_valid" not in round2_batch.non_tensor_batch:
                return

            log_dir = os.path.join(self.config.trainer.save_checkpoint_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            valid_log_file = os.path.join(log_dir, "round2_valid_qa.jsonl")
            # VERSE note: also sample INVALID questions + the solver's invalidity
            # explanation (+ DGE reason / invalid-reason judge score) to a sibling
            # file, so single-task runs can inspect what gets rejected and why.
            invalid_log_file = os.path.join(log_dir, "round2_invalid_qa.jsonl")

            responses = self._decode_responses_from_batch(round2_batch)
            accuracies = reward_metrics_round2.get("accuracy", [])
            if not isinstance(accuracies, list) or len(accuracies) != len(round2_batch):
                accuracies = [None] * len(round2_batch)
            invalid_reason_scores = reward_metrics_round2.get("invalid_reason_score", [])
            if not isinstance(invalid_reason_scores, list) or len(invalid_reason_scores) != len(round2_batch):
                invalid_reason_scores = [None] * len(round2_batch)
            # VERSE note: split-role PRM -- per-response reasoning score (mean over steps)
            # and the per-step breakdown, so valid-QA samples show what the PRM rewarded.
            prm_reasoning_scores = reward_metrics_round2.get("prm_reasoning_score", [])
            if not isinstance(prm_reasoning_scores, list) or len(prm_reasoning_scores) != len(round2_batch):
                prm_reasoning_scores = [None] * len(round2_batch)
            prm_step_scores = reward_metrics_round2.get("prm_step_scores", [])
            if not isinstance(prm_step_scores, list) or len(prm_step_scores) != len(round2_batch):
                prm_step_scores = [None] * len(round2_batch)
            # VERSE note: maze trajectory fidelity (engine-verified process
            # signal) — present with full length only on maze steps.
            maze_fidelities = reward_metrics_round2.get("maze_traj_fidelity", [])
            if not isinstance(maze_fidelities, list) or len(maze_fidelities) != len(round2_batch):
                maze_fidelities = [None] * len(round2_batch)
            # VERSE note: VSI grounding fidelity (engine-verified process
            # signal) — present with full length only on groundable scene steps.
            vsi_groundings = reward_metrics_round2.get("vsi_grounding", [])
            if not isinstance(vsi_groundings, list) or len(vsi_groundings) != len(round2_batch):
                vsi_groundings = [None] * len(round2_batch)

            question_types = round2_batch.non_tensor_batch.get("question_type")
            questions = round2_batch.non_tensor_batch.get("question")
            ground_truths = round2_batch.non_tensor_batch.get("gt")
            validity_flags = round2_batch.non_tensor_batch.get("is_valid")
            duplicate_weights = round2_batch.non_tensor_batch.get("duplicate_weight")
            padding_flags = round2_batch.non_tensor_batch.get("is_padding")
            prompt_modes = round2_batch.non_tensor_batch.get("prompt_mode")
            scene_ids = round2_batch.non_tensor_batch.get("scene_id")
            multi_modal_data = round2_batch.non_tensor_batch.get("multi_modal_data")
            sim_errors = round2_batch.non_tensor_batch.get("sim_error")
            sim_error_codes = round2_batch.non_tensor_batch.get("sim_error_code")
            sim_failure_stages = round2_batch.non_tensor_batch.get("sim_failure_stage")
            round2_modes = round2_batch.non_tensor_batch.get("round2_mode")

            if any(field is None for field in (question_types, questions, ground_truths, validity_flags)):
                return

            if duplicate_weights is None:
                duplicate_weights = np.ones(len(round2_batch), dtype=np.int32)
            if padding_flags is None:
                padding_flags = np.zeros(len(round2_batch), dtype=bool)
            if prompt_modes is None:
                prompt_modes = np.array(["unknown"] * len(round2_batch), dtype=object)
            if scene_ids is None:
                scene_ids = np.array(["unknown"] * len(round2_batch), dtype=object)

            def _images_for(idx: int) -> list[str]:
                image_paths: list[str] = []
                if multi_modal_data is not None:
                    mm_item = multi_modal_data[idx]
                    if isinstance(mm_item, dict):
                        raw_images = mm_item.get("images")
                        if isinstance(raw_images, (list, tuple)):
                            image_paths = [str(item) for item in raw_images]
                        elif raw_images:
                            image_paths = [str(raw_images)]
                return image_paths

            valid_grouped: dict[str, dict[str, Any]] = {}
            valid_order: list[str] = []
            invalid_grouped: dict[str, dict[str, Any]] = {}
            invalid_order: list[str] = []

            for idx in range(len(round2_batch)):
                if bool(padding_flags[idx]):
                    continue
                uid = str(round2_batch.non_tensor_batch["uid"][idx] or "")
                if not uid:
                    continue

                if bool(validity_flags[idx]):
                    if uid not in valid_grouped:
                        valid_order.append(uid)
                        valid_grouped[uid] = {
                            "step": int(step),
                            "question_uid": uid,
                            "is_valid": True,
                            "question_type": str(question_types[idx] or ""),
                            "question": str(questions[idx] or ""),
                            "gt": str(ground_truths[idx] or ""),
                            "images": _images_for(idx),
                            "prompt_mode": str(prompt_modes[idx] or ""),
                            "scene_id": str(scene_ids[idx] or ""),
                            "duplicate_weight": int(duplicate_weights[idx]),
                            "responses": [],
                            "accuracies": [],
                            "prm_reasoning_scores": [],
                            "prm_step_scores": [],
                            "maze_traj_fidelities": [],
                            "vsi_groundings": [],
                        }
                    valid_grouped[uid]["responses"].append(str(responses[idx] or ""))
                    valid_grouped[uid]["accuracies"].append(
                        None if accuracies[idx] is None else float(accuracies[idx])
                    )
                    valid_grouped[uid]["maze_traj_fidelities"].append(
                        None if maze_fidelities[idx] is None else float(maze_fidelities[idx])
                    )
                    valid_grouped[uid]["vsi_groundings"].append(
                        None if vsi_groundings[idx] is None else float(vsi_groundings[idx])
                    )
                    valid_grouped[uid]["prm_reasoning_scores"].append(
                        None if prm_reasoning_scores[idx] is None else float(prm_reasoning_scores[idx])
                    )
                    _step_list = prm_step_scores[idx]
                    valid_grouped[uid]["prm_step_scores"].append(
                        list(_step_list) if isinstance(_step_list, (list, tuple)) else []
                    )
                else:
                    if uid not in invalid_grouped:
                        invalid_order.append(uid)
                        invalid_grouped[uid] = {
                            "step": int(step),
                            "question_uid": uid,
                            "is_valid": False,
                            "question_type": str(question_types[idx] or ""),
                            "question": str(questions[idx] or ""),
                            "images": _images_for(idx),
                            "prompt_mode": str(prompt_modes[idx] or ""),
                            "scene_id": str(scene_ids[idx] or ""),
                            "duplicate_weight": int(duplicate_weights[idx]),
                            "sim_error": "" if sim_errors is None else str(sim_errors[idx] or ""),
                            "sim_error_code": "" if sim_error_codes is None else str(sim_error_codes[idx] or ""),
                            "sim_failure_stage": "" if sim_failure_stages is None else str(sim_failure_stages[idx] or ""),
                            "round2_mode": "" if round2_modes is None else str(round2_modes[idx] or ""),
                            "solver_explanations": [],
                            "invalid_reason_scores": [],
                        }
                    invalid_grouped[uid]["solver_explanations"].append(str(responses[idx] or ""))
                    invalid_grouped[uid]["invalid_reason_scores"].append(
                        None if invalid_reason_scores[idx] is None else float(invalid_reason_scores[idx])
                    )

            if valid_order:
                with open(valid_log_file, "a", encoding="utf-8") as f:
                    for uid in valid_order:
                        f.write(json.dumps(valid_grouped[uid], ensure_ascii=False) + "\n")
            if invalid_order:
                with open(invalid_log_file, "a", encoding="utf-8") as f:
                    for uid in invalid_order:
                        f.write(json.dumps(invalid_grouped[uid], ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[Warning] Failed to log Round2 sampled QA: {exc}")

    def _add_per_type_reward_metrics(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        metrics: dict[str, Any],
        metric_prefix: str,
    ) -> None:
        try:
            if reward_tensor is None:
                return
            if "question_type" not in batch.non_tensor_batch:
                return

            if torch.is_tensor(reward_tensor):
                if reward_tensor.dim() > 1:
                    sample_rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
                else:
                    sample_rewards = reward_tensor.detach().cpu().tolist()
            else:
                sample_rewards = list(reward_tensor)

            q_types = batch.non_tensor_batch["question_type"].tolist()
            if "is_padding" in batch.non_tensor_batch:
                is_padding = np.asarray(batch.non_tensor_batch["is_padding"], dtype=bool).tolist()
            else:
                is_padding = [False] * len(q_types)
            if len(q_types) != len(sample_rewards) or len(is_padding) != len(sample_rewards):
                return

            supported_task_names = TASK_ANSWER_HINT_TASKS or {
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

            def _normalize_type_name(s: str) -> str:
                if not isinstance(s, str):
                    return ""
                return re.sub(r"[\s_\[\]【】]+", "", s).lower()

            def _canonicalize_type_name(raw_type: str) -> str:
                norm_input = _normalize_type_name(raw_type)
                if not norm_input:
                    return "unknown"

                for registry_type in supported_task_names:
                    if norm_input == _normalize_type_name(registry_type):
                        return registry_type
                for registry_type in supported_task_names:
                    norm_registry = _normalize_type_name(registry_type)
                    if norm_input in norm_registry or norm_registry in norm_input:
                        return registry_type
                return "unknown"

            type_to_rewards = defaultdict(list)
            for t, r, pad in zip(q_types, sample_rewards, is_padding):
                if bool(pad):
                    continue
                t_key = _canonicalize_type_name(str(t))
                type_to_rewards[t_key].append(float(r))

            for t_key, vals in type_to_rewards.items():
                metrics[f"{metric_prefix}/type_reward_mean/{t_key}"] = float(np.mean(vals))
                metrics[f"{metric_prefix}/type_count/{t_key}"] = float(len(vals))
        except Exception:
            pass

    def init_workers(self) -> None:
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], 
                config=self.config.worker, 
                role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError("Only hybrid-engine mode is supported at the moment.")

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], 
                config=self.config.worker, 
                role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        if self.use_reward_model:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], 
                config=self.config.worker, 
                role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Loading checkpoint from: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Dataloader state not found at {dataloader_path}; starting from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        if self.config.trainer.val_generations_to_log <= 0:
            return

        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []  # Per-batch token rewards.
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)  # Scalar reward metrics.
        length_metrics_lst = defaultdict(list)  # Sequence-length metrics.
        
        print("Starting validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()  # Collapse token rewards to sequence scores.
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        
        print("Validation finished.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None  # Release the batch reference early.
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Starting batch generation...")

        while True:
            num_try_make_batch += 1

            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }

            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            gen_batch, gen_pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            gen_batch_output = unpad_dataproto(
                gen_batch_output,
                pad_size=gen_pad_size * int(self.config.worker.rollout.n),
            )

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1  # One sample for baseline scoring.
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)
                gen_baseline_output = unpad_dataproto(gen_baseline_output, pad_size=gen_pad_size)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No samples remained after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size  # Prompt batch size.
            
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continuing generation...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Too many generation attempts. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Generation finished.")
                
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def _init_modality_dataloaders(self):
        """Create per-modality StatefulDataLoaders by scanning the training JSONL.

        Groups dataset indices by prompt mode (scene / image_pair / single_image /
        maze) and builds a separate StatefulDataLoader with SubsetRandomSampler for
        each.  This lets _make_round1_batch_data pull a homogeneous batch whose
        modality matches the round-robin schedule's task type, eliminating the
        skip-and-crash issue caused by modality-ordered data.
        """
        if hasattr(self, "_modality_dataloaders"):
            return

        import json as _json
        from torch.utils.data import SubsetRandomSampler

        train_file = self.config.data.train_files
        modality_indices: dict[str, list[int]] = {
            "scene": [],
            "image_pair": [],
            "single_image": [],
            "maze": [],
        }

        with open(train_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                d = _json.loads(line)
                imgs = d.get("images", d.get("image", []))
                if isinstance(imgs, str):
                    imgs = [imgs]
                n = len(imgs) if isinstance(imgs, list) else 0
                if n >= 3:
                    modality_indices["scene"].append(i)
                elif n == 2:
                    modality_indices["image_pair"].append(i)
                elif n == 1:
                    path = str(imgs[0]) if imgs else ""
                    if "maze" in path.lower():
                        modality_indices["maze"].append(i)
                    else:
                        modality_indices["single_image"].append(i)
                else:
                    modality_indices["single_image"].append(i)

        dataset = self.train_dataloader.dataset
        batch_size = self.config.data.mini_rollout_batch_size or self.config.data.rollout_batch_size
        collate = self.train_dataloader.collate_fn

        self._modality_dataloaders: dict[str, StatefulDataLoader] = {}
        self._modality_iterators: dict[str, Any] = {}
        for mode, indices in modality_indices.items():
            if not indices:
                continue
            sampler = SubsetRandomSampler(indices)
            dl = StatefulDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=8,
                collate_fn=collate,
                pin_memory=False,
                drop_last=True,
            )
            self._modality_dataloaders[mode] = dl
            self._modality_iterators[mode] = iter(dl)

        print("[modality-aware] Created per-modality dataloaders:")
        for mode, indices in modality_indices.items():
            if indices:
                print(f"  {mode}: {len(indices)} samples, {len(self._modality_dataloaders[mode])} batches")

    def _make_round1_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Starting Round1 batch generation...")

        # --- Modality-aware data loading (round-robin scheduler) ---
        # Determine the required modality for this step from the schedule and pull
        # from a modality-specific dataloader.  Every sample in the batch is
        # guaranteed to match the scheduled task's modality, so the modality guard
        # in _build_round1_task_conditioned_batch never triggers.
        scheduler_mode = os.environ.get("SCHEDULER", "").strip().lower()
        use_modality_dl = False
        modality_iterator = None
        _required_mode = None

        if scheduler_mode == "round_robin":
            if not hasattr(self, "_modality_iterators"):
                self._init_modality_dataloaders()

            scheduled_type = self._get_round_robin_scheduled_type()
            _required_mode = TASK_TYPE_TO_PROMPT_MODE.get(scheduled_type) if scheduled_type else None

            if _required_mode and _required_mode in self._modality_iterators:
                modality_iterator = self._modality_iterators[_required_mode]
                use_modality_dl = True
                print(f"[modality-aware] step {self.global_step}: task={scheduled_type}, mode={_required_mode}")

        while True:
            num_try_make_batch += 1

            if use_modality_dl and modality_iterator is not None:
                try:
                    batch_dict = next(modality_iterator)
                except StopIteration:
                    # Reset this modality's iterator and retry.
                    self._modality_iterators[_required_mode] = iter(
                        self._modality_dataloaders[_required_mode]
                    )
                    modality_iterator = self._modality_iterators[_required_mode]
                    batch_dict = next(modality_iterator)
            else:
                try:
                    batch_dict = next(self.data_iterator)
                except StopIteration:
                    self.data_iterator = iter(self.train_dataloader)
                    batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }

            source_batch = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            source_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(source_batch.batch))], dtype=object
            )
            new_batch = self._build_round1_task_conditioned_batch(source_batch)
            if new_batch is None or len(new_batch) == 0:
                metrics["round1/skipped_no_feasible_prompt_count"] = (
                    metrics.get("round1/skipped_no_feasible_prompt_count", 0.0) + float(len(source_batch))
                )
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch > 0 and num_try_make_batch >= max_try_make_batch:
                    raise RuntimeError(
                        f"Too many consecutively skipped samples: {num_try_make_batch=} >= {max_try_make_batch=}. "
                        "No feasible Round1 tasks remain in the current batch; check your data or feasibility filter."
                    )
                continue

            # The frozen-proposer (local-server) path needs the clean prompt TEXT
            # (no vision placeholder tokens) to send via the OpenAI API; carry it
            # into gen_batch alongside the tokenized inputs.
            pop_non_tensor_keys = ["raw_prompt_ids", "multi_modal_data"]
            if self._split_role_frozen_proposer and "round1_prompt_text" in new_batch.non_tensor_batch:
                pop_non_tensor_keys.append("round1_prompt_text")
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=pop_non_tensor_keys,
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )
            # Round-1 (Questioner) is single-turn: cap its generation budget so the
            # large data.max_response_length reserved for the agentic Round-2 Solver
            # does not let the Questioner ramble for thousands of tokens.
            gen_batch.meta_info["max_tokens"] = 1024

            if self._split_role_frozen_proposer:
                gen_batch.meta_info.update(
                    {
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                    }
                )
                gen_batch_output = self._generate_round1_with_frozen_proposer(gen_batch)
                gen_pad_size = 0
            else:
                gen_batch, gen_pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
                gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
                gen_batch_output = unpad_dataproto(
                    gen_batch_output,
                    pad_size=gen_pad_size * int(self.config.worker.rollout.n),
                )

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)
                gen_baseline_output = unpad_dataproto(gen_baseline_output, pad_size=gen_pad_size)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No samples remained after filtering. Please check your data.")
                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch

            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continuing generation...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Too many generation attempts. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Generation finished.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                batch = batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
                break

        def parse_vlm_output(vlm_response, fallback_model_client=True):
            def _contains_invalid_inner_tags(text: str) -> bool:
                return bool(re.search(r"<\s*/?\s*[A-Za-z][^>]*>", str(text or "")))

            observation_open = len(re.findall(r"<\s*observation\s*>", vlm_response, flags=re.IGNORECASE))
            observation_close = len(re.findall(r"<\s*/\s*observation\s*>", vlm_response, flags=re.IGNORECASE))
            question_open = len(re.findall(r"<\s*question\s*>", vlm_response, flags=re.IGNORECASE))
            question_close = len(re.findall(r"<\s*/\s*question\s*>", vlm_response, flags=re.IGNORECASE))
            observation_count = observation_open
            question_count = question_open
            is_multi_question = observation_count > 1 or question_count > 1

            result = {
                "observation": "",
                "question": "Unknown",
                "format_score": 0.0,
                "observation_count": observation_count,
                "question_count": question_count,
                "is_multi_question": is_multi_question,
                "has_invalid_inner_tags": False,
            }

            obs_match = re.search(
                r"<\s*observation\s*>(.*?)<\s*/\s*observation\s*>",
                vlm_response,
                flags=re.IGNORECASE | re.DOTALL,
            )
            ques_match = re.search(
                r"<\s*question\s*>(.*?)<\s*/\s*question\s*>",
                vlm_response,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if ques_match:
                extracted_obs = obs_match.group(1).strip() if obs_match else ""
                extracted_question = ques_match.group(1).strip()
                stripped = re.sub(
                    r"<\s*observation\s*>.*?<\s*/\s*observation\s*>",
                    "",
                    vlm_response,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                stripped = re.sub(
                    r"<\s*question\s*>.*?<\s*/\s*question\s*>",
                    "",
                    stripped,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                is_clean_format = (
                    question_open == 1
                    and question_close == 1
                    and observation_open == observation_close
                    and observation_open <= 1
                    and not stripped.strip()
                )
                has_invalid_inner_tags = _contains_invalid_inner_tags(extracted_obs) or _contains_invalid_inner_tags(
                    extracted_question
                )
                result.update(
                    {
                        "observation": extracted_obs,
                        "question": extracted_question,
                        "format_score": 1.0 if (is_clean_format and extracted_obs) else 0.5 if is_clean_format else 0.0,
                        "has_invalid_inner_tags": has_invalid_inner_tags,
                    }
                )
                return result

            if fallback_model_client:
                fallback_res = call_fallback_agent(vlm_response)
                fallback_res.update(
                    {
                        "observation_count": observation_count,
                        "question_count": question_count,
                        "is_multi_question": is_multi_question,
                    }
                )
                return fallback_res

            return result

        def call_fallback_agent(raw_text, ak=None, model=None, url=None):
            # Repair malformed Proposer output via an LLM. In split-role we point
            # this at the LOCAL frozen-proposer server (no external API). This must
            # NEVER crash training: degrade to a safe no-op if no endpoint is
            # configured or all attempts fail.
            if url is None:
                url = getattr(self, "_proposer_openai_base_url", "") or ""
            if ak is None:
                ak = getattr(self, "_proposer_openai_api_key", "") or "EMPTY"
            if model is None:
                model = getattr(self, "_proposer_openai_model", "") or "gpt-oss-120b-ldm"
            _fallback_no_op = {"observation": "", "question": "Unknown", "format_score": 0.0, "is_fallback": True}
            if not url:
                return _fallback_no_op
            repair_prompt = f"""You are a data recovery assistant. Your task is to extract one grounded observation and one spatial question from the raw text below.

                ### RAW TEXT TO REPAIR:
                {raw_text}

                ### CONSTRAINTS:
                1. If the raw text contains an <observation> block or grounded scene description, preserve it inside <observation>. If none exists, leave <observation> empty.
                2. Extract exactly one final question and place it inside <question>.
                3. Output EXACTLY ONE <observation> tag and EXACTLY ONE <question> tag.
                4. No extra text outside the tags.

                ### OUTPUT FORMAT:
                <observation>[Extracted grounded observation, or empty if not found]</observation>
                <question>[Extracted Question]</question>"""

            messages = [{"role": "user", "content": repair_prompt}]

            for attempt in range(10):
                try:
                    client = OpenAI(api_key=ak or "EMPTY", base_url=url)
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model,
                        temperature=0.1,
                        timeout=30,
                    )
                    repaired_response = chat_completion.choices[0].message.content.strip()
                    if repaired_response:
                        repaired = parse_vlm_output(repaired_response, fallback_model_client=False)
                        repaired["format_score"] = 0.0
                        repaired["is_fallback"] = True
                        return repaired
                except Exception as exc:
                    if attempt < 9:
                        time.sleep((attempt + 1) * 2)
                    else:
                        print(f"Fallback repair failed: {exc}")

            return {
                "observation": "",
                "question": "Unknown",
                "format_score": 0.0,
                "is_fallback": True,
            }

        def process_question_content(q_item, canonical_task_type: str):
            original_text = q_item["question"]
            return _build_question_with_answer_hint(
                original_text,
                canonical_task_type,
                round_stage=1,
            )

        def _extract_maze_pick_letter(text: str) -> Optional[str]:
            """Extract A/B/C/D from pick-letter Questioner output."""
            m = re.search(r"<answer>\s*([ABCD])\s*</answer>", str(text or ""), re.IGNORECASE)
            if m:
                return m.group(1).upper()
            m = re.search(r"\b([ABCD])\b\s*$", str(text or "").strip())
            if m:
                return m.group(1).upper()
            return None

        questions_raw = self._decode_responses_from_batch(batch)
        questions = [parse_vlm_output(question) for question in questions_raw]
        scheduled_task_types = batch.non_tensor_batch["scheduled_task_type"].tolist()
        multi_modal_data = batch.non_tensor_batch.get("multi_modal_data", [])
        media_contexts = [
            self._build_simulator_media_context(multi_modal_data[i] if i < len(multi_modal_data) else None)
            for i in range(len(questions))
        ]

        # Maze pick-letter / round-robin mode: construct the full question text
        # (with verified action sequence) from the maze metadata.
        if MAZE_PROPOSER_MODE in ("pick_letter", "round_robin"):
            # Build a per-context letter assignment map for round_robin mode.
            # Each context's n rollouts get A, B, C, D, A, B, ... in order.
            context_letter_idx: dict[Any, int] = {}
            for i, q in enumerate(questions):
                if scheduled_task_types[i] != MAZE_TASK_TYPE:
                    continue
                if MAZE_PROPOSER_MODE == "round_robin":
                    # Identify the context by the maze image path (all rollouts
                    # for the same maze share the same image).
                    image_paths = media_contexts[i].get("image_paths", []) or []
                    ctx_key = image_paths[0] if image_paths else i
                    idx = context_letter_idx.get(ctx_key, 0)
                    letter = "ABCD"[idx % 4]
                    context_letter_idx[ctx_key] = idx + 1
                else:
                    # pick_letter: extract from model output.
                    letter = _extract_maze_pick_letter(q.get("question", ""))
                    if letter is None and i < len(questions_raw):
                        letter = _extract_maze_pick_letter(questions_raw[i])
                    if letter is None:
                        q["question"] = "Unknown"
                        q["format_score"] = 0.0
                        continue
                image_paths = media_contexts[i].get("image_paths", []) or []
                built = MazeEngine.build_question_from_letter(letter, image_paths[0] if image_paths else None)
                if built is None:
                    q["question"] = "Unknown"
                    q["format_score"] = 0.0
                    continue
                q["question"] = built["question_text"]
                q["format_score"] = 1.0

        # Phase-2 structured_imitate: parse <question_params>{...}</question_params>
        # from the Proposer output and render the final GRPO-style question text.
        # The Proposer only picks object labels; the system fills the question text,
        # so format is guaranteed valid. Validity (DGE) is then checked by the
        # simulator below as usual. Maze tasks are skipped (handled above).
        if PROPOSER_MODE == "structured_imitate":
            for i, q in enumerate(questions):
                tt = scheduled_task_types[i]
                if tt not in DGE_TASK_CONFIG:
                    continue
                raw = questions_raw[i] if i < len(questions_raw) else q.get("question", "")
                params = self._extract_dge_question_params(raw)
                if params is None:
                    q["question"] = "Unknown"
                    q["format_score"] = 0.0
                    q["dge_prefilled_params"] = None
                    continue
                # VERSE note: clean params — filter placeholders, enforce target ∉ candidates
                cleaned_params, clean_reason = self._clean_dge_structured_params(tt, params)
                if cleaned_params is None:
                    print(f"[structured_imitate] {tt} invalid params: {clean_reason}")
                    q["question"] = "Unknown"
                    q["format_score"] = 0.0
                    q["dge_prefilled_params"] = None
                    continue
                try:
                    rendered, _style, _opts, _gt = render_dge_structured_question(tt, cleaned_params)
                except Exception as exc:
                    print(f"[structured_imitate] render failed for {tt}: {exc}")
                    q["question"] = "Unknown"
                    q["format_score"] = 0.0
                    q["dge_prefilled_params"] = None
                    continue
                if not rendered:
                    q["question"] = "Unknown"
                    q["format_score"] = 0.0
                    q["dge_prefilled_params"] = None
                    continue
                q["question"] = rendered
                q["format_score"] = 1.0
                # VERSE note: store cleaned params for DGE prefill (skip lossy LLM extraction)
                q["dge_prefilled_params"] = cleaned_params

        batch.non_tensor_batch["question"] = np.array(
            [process_question_content(q, scheduled_task_types[i]) for i, q in enumerate(questions)],
            dtype=object,
        )
        batch.non_tensor_batch["question_type_raw"] = np.array(scheduled_task_types, dtype=object)
        batch.non_tensor_batch["observation"] = np.array(
            [question.get("observation", "") for question in questions], dtype=object
        )
        batch.non_tensor_batch["format_score"] = np.array(
            [question["format_score"] for question in questions], dtype=np.float32
        )
        batch.non_tensor_batch["question_count"] = np.array(
            [q["question_count"] for q in questions], dtype=np.int32
        )
        batch.non_tensor_batch["has_invalid_inner_tags"] = np.array(
            [bool(q.get("has_invalid_inner_tags", False)) for q in questions], dtype=bool
        )

        sim_jobs = [
            {
                "media_context": media_contexts[i],
                "canonical_task_type": scheduled_task_types[i],
                "question_text": batch.non_tensor_batch["question"][i],
                "prefilled_params": questions[i].get("dge_prefilled_params"),
            }
            for i in range(len(questions))
        ]

        simulator_max_workers = self._get_world_simulator_max_workers(len(sim_jobs))
        if simulator_max_workers > 1 and len(sim_jobs) > 1:
            try:
                with ThreadPoolExecutor(max_workers=simulator_max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._run_simulator_validation_job,
                            media_context=job["media_context"],
                            canonical_task_type=job["canonical_task_type"],
                            question_text=job["question_text"],
                            use_thread_local_simulator=True,
                            prefilled_params=job.get("prefilled_params"),
                        )
                        for job in sim_jobs
                    ]
                    sim_results = [future.result() for future in futures]
            except Exception as exc:
                print(f"WorldSimulator parallel validation failed; falling back to serial execution: {exc}")
                sim_results = [
                    self._run_simulator_validation_job(
                        media_context=job["media_context"],
                        canonical_task_type=job["canonical_task_type"],
                        question_text=job["question_text"],
                        use_thread_local_simulator=False,
                        prefilled_params=job.get("prefilled_params"),
                    )
                    for job in sim_jobs
                ]
        else:
            sim_results = [
                self._run_simulator_validation_job(
                    media_context=job["media_context"],
                    canonical_task_type=job["canonical_task_type"],
                    question_text=job["question_text"],
                    use_thread_local_simulator=False,
                    prefilled_params=job.get("prefilled_params"),
                )
                for job in sim_jobs
            ]

        # VERSE note:selection-over-reward: replace invalid proposals with
        # system-sampled valid questions (same pools, same templates, DGE re-verified)
        # so Round-2 agentic rollouts are spent on real answering signal instead of
        # the "explain why invalid" branch. The proposer's own validity is kept in
        # `proposer_is_valid` for logging/reward; `is_valid` reflects the final
        # question actually handed to the Solver.
        proposer_is_valid_list = [bool(result.get("is_valid", False)) for result in sim_results]
        question_source_list = ["proposer"] * len(sim_results)
        if self._question_resample and PROPOSER_MODE == "structured_imitate":
            resample_indices = [
                i
                for i, result in enumerate(sim_results)
                if not bool(result.get("is_valid", False))
                and scheduled_task_types[i] in DGE_TASK_CONFIG
                and scheduled_task_types[i] != MAZE_TASK_TYPE
                and scheduled_task_types[i] not in self._question_resample_blacklist
            ]
            resample_success = 0
            if resample_indices:
                def _run_resample(i: int):
                    try:
                        return i, self._resample_one_invalid_question(
                            index=i,
                            task_type=scheduled_task_types[i],
                            media_context=media_contexts[i],
                            global_step=self.global_step,
                            max_tries=max(1, self._question_resample_tries),
                        )
                    except Exception as exc:
                        print(f"[question-resample] idx={i} failed: {exc}")
                        return i, None

                resample_workers = self._get_world_simulator_max_workers(len(resample_indices))
                if resample_workers > 1 and len(resample_indices) > 1:
                    with ThreadPoolExecutor(max_workers=resample_workers) as executor:
                        resample_results = list(executor.map(_run_resample, resample_indices))
                else:
                    resample_results = [_run_resample(i) for i in resample_indices]

                for i, res in resample_results:
                    if res is None:
                        continue
                    sim_results[i] = res["sim_result"]
                    questions[i]["question"] = res["question_text"]
                    questions[i]["dge_prefilled_params"] = res["params"]
                    batch.non_tensor_batch["question"][i] = process_question_content(
                        questions[i], scheduled_task_types[i]
                    )
                    question_source_list[i] = "system_resample"
                    resample_success += 1
            n_total = max(1, len(sim_results))
            metrics["round1/proposer_invalid_ratio"] = float(
                sum(1 for v in proposer_is_valid_list if not v)
            ) / float(n_total)
            metrics["round1/resample_attempted"] = float(len(resample_indices))
            metrics["round1/resample_succeeded"] = float(resample_success)

        batch.non_tensor_batch["proposer_is_valid"] = np.array(proposer_is_valid_list, dtype=bool)
        batch.non_tensor_batch["question_source"] = np.array(question_source_list, dtype=object)

        resolved_question_types = [result.get("resolved_task_type", "unknown") for result in sim_results]
        sim_answers = [result.get("answer", "invalid") for result in sim_results]
        is_valid_list = [bool(result.get("is_valid", False)) for result in sim_results]
        sim_error_list = [str(result.get("error", "") or "") for result in sim_results]
        sim_error_code_list = [str(result.get("error_code", "") or "") for result in sim_results]
        sim_failure_stage_list = [str(result.get("failure_stage", "") or "") for result in sim_results]
        sim_parsed_params_list = [
            self._json_dumps_non_tensor(result.get("parsed_params", {}) or {})
            for result in sim_results
        ]
        sim_validation_result_list = [
            self._json_dumps_non_tensor(result.get("validation_result"))
            for result in sim_results
        ]
        sim_judge_reference_list = [
            self._json_dumps_non_tensor(result.get("judge_reference"))
            for result in sim_results
        ]
        sim_result_json_list = [
            self._json_dumps_non_tensor(result.get("sim_result"))
            for result in sim_results
        ]

        batch.non_tensor_batch["question_type"] = np.array(resolved_question_types, dtype=object)
        batch.non_tensor_batch["gt"] = np.array(sim_answers, dtype=object)
        batch.non_tensor_batch["is_valid"] = np.array(is_valid_list, dtype=bool)
        batch.non_tensor_batch["sim_error"] = np.array(sim_error_list, dtype=object)
        batch.non_tensor_batch["sim_error_code"] = np.array(sim_error_code_list, dtype=object)
        batch.non_tensor_batch["sim_failure_stage"] = np.array(sim_failure_stage_list, dtype=object)
        batch.non_tensor_batch["sim_parsed_params_json"] = np.array(sim_parsed_params_list, dtype=object)
        batch.non_tensor_batch["sim_validation_result_json"] = np.array(sim_validation_result_list, dtype=object)
        batch.non_tensor_batch["sim_judge_reference_json"] = np.array(sim_judge_reference_list, dtype=object)
        batch.non_tensor_batch["sim_result_json"] = np.array(sim_result_json_list, dtype=object)

        selected_uid_to_type = {}
        for uid, task_type in zip(
            batch.non_tensor_batch["uid"].tolist(),
            batch.non_tensor_batch["scheduled_task_type"].tolist(),
        ):
            selected_uid_to_type.setdefault(uid, task_type)
        for task_type in selected_uid_to_type.values():
            metrics[f"round1_scheduler/selected_count/{task_type}"] = (
                metrics.get(f"round1_scheduler/selected_count/{task_type}", 0.0) + 1.0
            )

        batch_size = len(batch)
        question_uids = np.array([f"question-{i}" for i in range(batch_size)], dtype=object)
        batch.non_tensor_batch["question_uid"] = question_uids
        batch.non_tensor_batch["round"] = np.full(batch_size, 1, dtype=np.int32)
        return batch

    def _extract_questions_from_batch(self, batch: DataProto) -> list[str]:
        return batch.non_tensor_batch["question"].tolist()
    def _extract_ground_truth_from_batch(self, batch: DataProto) -> list[str]:
        return batch.non_tensor_batch["gt"].tolist()

    def _extract_question_type_from_batch(self, batch: DataProto) -> list[str]:
        return batch.non_tensor_batch["question_type"].tolist()

    def _process_round1_to_round2_data(
        self,
        round1_batch: DataProto,
        questions: list[str],
        image_overrides: Optional[dict[int, list]] = None,
    ) -> DataProto:
        answer_batch = self._build_answer_batch(round1_batch, questions, image_overrides=image_overrides)
        
        return answer_batch

    def _render_spar_marked_images(self, image_item: Any, markers: list) -> Optional[list]:
        """Draw SPAR-Bench-style markers onto the first image of a sample.

        markers = [(label, color, kind)] with kind in {"point", "bbox"}.
        Object lookup replicates the DGE tool's `_find_visible_object` so the
        marker lands on the same instance used for the ground truth. Returns a
        new images list (marked PIL first, remaining entries untouched), or
        None when any marker cannot be rendered (caller then keeps the generic
        marker-free question text).
        """
        try:
            from PIL import Image as _PILImage
            from src.tools.spar_bench_tools import _find_visible_object
            from src.utils.marker_render import draw_point, draw_bbox
        except ImportError as exc:
            print(f"[spar-round2] marker imports failed: {exc}")
            return None
        if not isinstance(image_item, dict) or not markers:
            return None
        images = image_item.get("images")
        if not isinstance(images, list) or not images:
            return None
        image_paths = self._extract_media_paths_from_item(image_item)
        if not image_paths:
            return None
        first_path = str(image_paths[0])
        meta_path = os.path.splitext(first_path.replace("/train/", "/metadata/"))[0] + ".json"
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                frame_meta = json.load(f)
            img = _PILImage.open(first_path).convert("RGB")
        except Exception as exc:
            print(f"[spar-round2] failed to load image/metadata: {exc}")
            return None
        # Scale marker geometry so it stays clearly visible after the training
        # pipeline resizes the image down to the max_pixels budget.
        w, h = img.size
        max_px = float(getattr(self.config.data, "max_pixels", 200704) or 200704)
        scale = min(1.0, (max_px / max(1.0, float(w * h))) ** 0.5)
        radius = max(6, int(round(7.0 / scale)))
        box_width = max(3, int(round(3.0 / scale)))
        drawn = 0
        for label, color, kind in markers:
            obj = _find_visible_object(frame_meta, str(label), 0.1)
            bbox = obj.get("2d_bbox") if isinstance(obj, dict) else None
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            if kind == "bbox":
                img = draw_bbox(img, [float(v) for v in bbox], color, width=box_width)
            else:
                cx = (float(bbox[0]) + float(bbox[2])) / 2.0
                cy = (float(bbox[1]) + float(bbox[3])) / 2.0
                img = draw_point(img, cx, cy, color, radius=radius)
            drawn += 1
        if drawn < len(markers):
            return None
        new_images = list(images)
        new_images[0] = img
        return new_images

    def _build_maze_fidelity_refs(
        self,
        *,
        question_types: list[str],
        ground_truths: list[str],
        sim_parsed_params_list: Any,
        multi_modal_data_list: Any,
        validity_list: Any,
    ) -> list[str]:
        """Per-sample JSON refs for the maze trajectory-fidelity process reward.

        Empty string for non-maze / invalid samples. For maze samples includes the
        grid geometry (from MazeEngine metadata), the GT action sequence (parsed by
        the maze verifier in Round-1), and the processed image size the Solver
        actually sees (Path-Tracer coordinates live in that frame).
        """
        from src.maze_engine import (
            MARGIN_BOTTOM_FRAC,
            MARGIN_LEFT_FRAC,
            MARGIN_RIGHT_FRAC,
            MARGIN_TOP_FRAC,
        )

        refs: list[str] = []
        for i in range(len(question_types)):
            task_type = str(question_types[i] or "")
            is_valid = bool(validity_list[i]) if validity_list is not None and i < len(validity_list) else True
            if task_type != MAZE_TASK_TYPE or not is_valid:
                refs.append("")
                continue
            try:
                parsed_params = json.loads(sim_parsed_params_list[i]) if sim_parsed_params_list is not None else {}
            except (TypeError, ValueError):
                parsed_params = {}
            moves = parsed_params.get("moves") if isinstance(parsed_params, dict) else None
            if not isinstance(moves, list) or not moves:
                refs.append("")
                continue
            image_item = (
                multi_modal_data_list[i]
                if multi_modal_data_list is not None and i < len(multi_modal_data_list)
                else None
            )
            image_paths = self._extract_media_paths_from_item(image_item)
            if not image_paths:
                refs.append("")
                continue
            try:
                maze_meta = MazeEngine.load_metadata_for_image(image_paths[0])
            except Exception:
                refs.append("")
                continue
            try:
                images = image_item.get("images") if isinstance(image_item, dict) else None
                first_image = images[0] if isinstance(images, list) and images else image_paths[0]
                processed = process_image_agentic(
                    first_image, self.config.data.min_pixels, self.config.data.max_pixels
                )
                proc_w, proc_h = processed.size
            except Exception as exc:
                print(f"[maze-fidelity] failed to compute processed image size: {exc}")
                refs.append("")
                continue
            ref = {
                "grid_size": int(maze_meta.get("grid_size", 0)),
                "start": list(maze_meta.get("start", [])),
                "destinations": {k: list(v) for k, v in (maze_meta.get("destinations") or {}).items()},
                "moves": [str(m).lower() for m in moves],
                "gt_letter": str(ground_truths[i] or "").strip().upper(),
                "image_width": int(proc_w),
                "image_height": int(proc_h),
                "margins": [MARGIN_LEFT_FRAC, MARGIN_RIGHT_FRAC, MARGIN_TOP_FRAC, MARGIN_BOTTOM_FRAC],
            }
            refs.append(json.dumps(ref, ensure_ascii=False))
        return refs

    def _build_vsi_grounding_refs(
        self,
        *,
        question_types: list[str],
        sim_parsed_params_list: Any,
        multi_modal_data_list: Any,
        validity_list: Any,
    ) -> list[str]:
        """Per-sample JSON refs for the VSI grounding process reward.

        Empty string for non-scene / invalid / label-less samples. For groundable
        scene samples includes the question-relevant object labels and the GT 2D
        bboxes of those labels, scaled into the processed coordinate frame the
        Solver draws in. All scene frames share one source size, so GT instances
        from every frame live in one comparable coordinate system and the reward
        matches drawn boxes to GT by max IoU across frames (index-agnostic).
        """
        from PIL import Image as _PILImage

        def _relevant_labels(task_type: str, params: dict) -> list[str]:
            cfg = DGE_TASK_CONFIG.get(task_type)
            if not cfg:
                return []
            labels: list[str] = []
            for dge_field in cfg.get("dge_to_template", {}).keys():
                val = params.get(dge_field)
                if isinstance(val, list):
                    labels.extend(str(x).strip() for x in val if str(x).strip())
                elif val is not None and str(val).strip():
                    labels.append(str(val).strip())
            seen: set = set()
            uniq: list[str] = []
            for l in labels:
                k = l.lower()
                if k not in seen:
                    seen.add(k)
                    uniq.append(l)
            return uniq

        # Cache processed size per source (w,h) so we call process_image_agentic
        # once per distinct resolution rather than per frame.
        proc_size_cache: dict[tuple, tuple] = {}

        def _proc_size(path: str, orig_w: int, orig_h: int) -> tuple:
            key = (int(orig_w), int(orig_h))
            if key in proc_size_cache:
                return proc_size_cache[key]
            processed = process_image_agentic(str(path), self.config.data.min_pixels, self.config.data.max_pixels)
            size = processed.size
            proc_size_cache[key] = size
            return size

        refs: list[str] = []
        min_vis = 0.1
        for i in range(len(question_types)):
            task_type = str(question_types[i] or "")
            is_valid = bool(validity_list[i]) if validity_list is not None and i < len(validity_list) else True
            if not is_valid or task_type not in VSI_GROUNDING_TASK_TYPES:
                refs.append("")
                continue
            try:
                parsed_params = json.loads(sim_parsed_params_list[i]) if sim_parsed_params_list is not None else {}
            except (TypeError, ValueError):
                parsed_params = {}
            if not isinstance(parsed_params, dict):
                refs.append("")
                continue
            labels = _relevant_labels(task_type, parsed_params)
            if not labels:
                refs.append("")
                continue
            label_keys = {l.lower() for l in labels}

            image_item = (
                multi_modal_data_list[i]
                if multi_modal_data_list is not None and i < len(multi_modal_data_list)
                else None
            )
            image_paths = self._extract_media_paths_from_item(image_item)
            if not image_paths:
                refs.append("")
                continue

            gt_by_label: dict[str, list] = defaultdict(list)
            for p in image_paths:
                meta_path = os.path.splitext(str(p).replace("/train/", "/metadata/"))[0] + ".json"
                if not os.path.exists(meta_path):
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        frame_meta = json.load(f)
                    with _PILImage.open(p) as _im:
                        orig_w, orig_h = _im.size
                    proc_w, proc_h = _proc_size(p, orig_w, orig_h)
                except Exception as exc:
                    print(f"[vsi-grounding] frame skipped ({p}): {exc}")
                    continue
                sx = float(proc_w) / float(max(1, orig_w))
                sy = float(proc_h) / float(max(1, orig_h))
                for obj in (frame_meta.get("objects") or []):
                    lbl = str(obj.get("label", "")).strip().lower()
                    if not lbl:
                        continue
                    # keep only labels that (contains-)match a relevant label
                    if not any(lbl == k or lbl in k or k in lbl for k in label_keys):
                        continue
                    if float(obj.get("visibility", 1.0)) < min_vis:
                        continue
                    bbox = obj.get("2d_bbox")
                    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                        continue
                    gt_by_label[lbl].append(
                        [float(bbox[0]) * sx, float(bbox[1]) * sy, float(bbox[2]) * sx, float(bbox[3]) * sy]
                    )

            if not gt_by_label:
                # No GT for any referenced label visible in any frame -> nothing to
                # ground against; leave empty so the reward stays pure outcome.
                refs.append("")
                continue
            ref = {
                "relevant_labels": labels,
                "gt_by_label": {k: v for k, v in gt_by_label.items()},
            }
            refs.append(json.dumps(ref, ensure_ascii=False))
        return refs

    def _make_round2_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        if not hasattr(self, 'round1_batch') or self.round1_batch is None:
            raise RuntimeError("Round1 batch data does not exist. Call `_make_round1_batch_data` first.")
        
        round1_batch = self._deduplicate_round1_batch_for_round2(self.round1_batch, metrics=metrics)

        ground_truths = self._extract_ground_truth_from_batch(round1_batch)
        question_types = self._extract_question_type_from_batch(round1_batch)
        questions = self._extract_questions_from_batch(round1_batch)
        sim_parsed_params_list = round1_batch.non_tensor_batch.get("sim_parsed_params_json", None)
        sim_validation_result_list = round1_batch.non_tensor_batch.get("sim_validation_result_json", None)
        sim_result_json_list = round1_batch.non_tensor_batch.get("sim_result_json", None)
        multi_modal_data_list = round1_batch.non_tensor_batch.get("multi_modal_data", None)
        validity_list = round1_batch.non_tensor_batch.get("is_valid", None)
        round2_questions = []
        round2_answer_styles = []
        round2_mc_options = []
        round2_gt_letters = []
        round2_image_overrides: dict[int, list] = {}
        for i in range(len(questions)):
            _pp = sim_parsed_params_list[i] if sim_parsed_params_list is not None else None
            _vr = sim_validation_result_list[i] if sim_validation_result_list is not None else None
            _sr = sim_result_json_list[i] if sim_result_json_list is not None else None

            # VERSE note: SPAR tasks get a bench-style rebuild (marker
            # references + fixed instruction sentences + embedded options) AND
            # matching visual markers drawn onto the Solver's first image.
            # Only used when every referenced marker could be rendered.
            spar_spec = None
            _is_valid_i = bool(validity_list[i]) if validity_list is not None and i < len(validity_list) else True
            if _is_valid_i:
                spar_spec = _build_spar_bench_round2(
                    question_types[i],
                    ground_truths[i],
                    sim_parsed_params=_pp,
                    sim_validation_result=_vr,
                    sim_result=_sr,
                )
            if spar_spec is not None:
                image_item = (
                    multi_modal_data_list[i]
                    if multi_modal_data_list is not None and i < len(multi_modal_data_list)
                    else None
                )
                marked_images = self._render_spar_marked_images(image_item, spar_spec["markers"])
                if marked_images is not None:
                    round2_image_overrides[i] = marked_images
                else:
                    print(f"[spar-round2] marker render failed for {question_types[i]} idx={i}; using generic question")
                    spar_spec = None

            if spar_spec is not None:
                final_question = spar_spec["question"]
                answer_style = spar_spec["answer_style"]
                options = spar_spec["options"]
                gt_letter = spar_spec["gt_letter"]
            else:
                final_question, answer_style, options, gt_letter = _build_round2_agentic_question(
                    questions[i],
                    question_types[i],
                    ground_truths[i],
                    sim_parsed_params=_pp,
                    sim_validation_result=_vr,
                    sim_result=_sr,
                )
            round2_questions.append(final_question)
            round2_answer_styles.append(answer_style)
            round2_mc_options.append(json.dumps(options, ensure_ascii=False) if options else "")
            round2_gt_letters.append(gt_letter)

        # VERSE note:engine-verified maze process reward: attach the exact
        # grid geometry + GT action sequence so the Round-2 reward can score how
        # faithfully the Solver's Path-Tracer drawing tracks the true cell path
        # (see maze_trajectory_fidelity in agentic_solver_reward.py). Coordinates
        # drawn by the model live in the PROCESSED image frame, so we record the
        # processed size alongside the maze margins.
        maze_fidelity_refs = self._build_maze_fidelity_refs(
            question_types=question_types,
            ground_truths=ground_truths,
            sim_parsed_params_list=sim_parsed_params_list,
            multi_modal_data_list=multi_modal_data_list,
            validity_list=validity_list,
        )

        # VERSE note:engine-verified VSI grounding process reward: attach
        # the question-relevant object labels + their GT 2D bboxes (scaled to the
        # processed frame the Solver draws in) so the Round-2 reward can score how
        # faithfully the Solver's Object-Mapper boxes localize the queried objects
        # (see vsi_grounding_fidelity in agentic_solver_reward.py). VSI is
        # otherwise outcome-only; this targets the draw-then-guess decoupling.
        vsi_grounding_refs = self._build_vsi_grounding_refs(
            question_types=question_types,
            sim_parsed_params_list=sim_parsed_params_list,
            multi_modal_data_list=multi_modal_data_list,
            validity_list=validity_list,
        )

        processed_batch = self._process_round1_to_round2_data(
            round1_batch, round2_questions, image_overrides=round2_image_overrides
        )

        processed_batch.non_tensor_batch["question"] = np.array(round2_questions, dtype=object)
        processed_batch.non_tensor_batch["answer_style"] = np.array(round2_answer_styles, dtype=object)
        processed_batch.non_tensor_batch["mc_options_json"] = np.array(round2_mc_options, dtype=object)
        processed_batch.non_tensor_batch["gt_letter"] = np.array(round2_gt_letters, dtype=object)
        processed_batch.non_tensor_batch["maze_fidelity_ref_json"] = np.array(maze_fidelity_refs, dtype=object)
        processed_batch.non_tensor_batch["vsi_grounding_ref_json"] = np.array(vsi_grounding_refs, dtype=object)
        processed_batch.non_tensor_batch["gt"] = np.array(ground_truths, dtype=object)
        processed_batch.non_tensor_batch["question_type"] = np.array(question_types, dtype=object)
        if "duplicate_weight" in round1_batch.non_tensor_batch:
            processed_batch.non_tensor_batch["duplicate_weight"] = np.array(
                round1_batch.non_tensor_batch["duplicate_weight"], dtype=np.int32
            )
        else:
            processed_batch.non_tensor_batch["duplicate_weight"] = np.ones(len(processed_batch), dtype=np.int32)
        if "is_valid" in round1_batch.non_tensor_batch:
            processed_batch.non_tensor_batch["is_valid"] = np.array(
                round1_batch.non_tensor_batch["is_valid"], dtype=bool
            )
        processed_batch.non_tensor_batch["is_padding"] = np.zeros(len(processed_batch), dtype=bool)
        for field_name in (
            "sim_error",
            "sim_error_code",
            "sim_failure_stage",
            "sim_parsed_params_json",
            "sim_validation_result_json",
            "sim_judge_reference_json",
            "sim_result_json",
            "question_type_raw",
            "question_source",
            "proposer_is_valid",
        ):
            if field_name in round1_batch.non_tensor_batch:
                processed_batch.non_tensor_batch[field_name] = np.array(
                    round1_batch.non_tensor_batch[field_name], dtype=object
                )

        if "question_uid" in round1_batch.non_tensor_batch:
            question_uids = round1_batch.non_tensor_batch["question_uid"]
        else:
            question_uids = round1_batch.non_tensor_batch["uid"]
        
        processed_batch.non_tensor_batch["uid"] = question_uids.copy()

        new_batch = processed_batch.select()
        gen_batch = new_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "grid_size"],
            meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
        )
        # Round-2 Solver runs the ViLaSR agentic think->draw->observe loop.
        gen_batch.meta_info["rollout_mode"] = "agentic"

        # VERSE note:Verifier-Guided rollout Search (VGS-GRPO): widen the
        # solver frontier to effective_n = rollout.n * B for trained tasks, then
        # engine-select rollout.n rollouts per group after generation (below).
        # B=1 (default off) keeps byte-identical behavior. See tts_search.py.
        base_n = int(self.config.worker.rollout.n)
        tts_enabled = _env_flag("SPATIALEVO_TTS_SEARCH", False)
        tts_branch = int(os.environ.get("SPATIALEVO_TTS_BRANCH", "2")) if tts_enabled else 1
        tts_tasks = {
            t.strip() for t in str(os.environ.get("SPATIALEVO_TTS_TASKS", "maze,vsi")).split(",") if t.strip()
        }
        tts_fidelity_weight = float(os.environ.get("SPATIALEVO_TTS_FIDELITY_WEIGHT", "0.5"))
        step_task = str(question_types[0] or "") if len(question_types) else ""

        def _tts_applies(tt: str) -> bool:
            tt = str(tt or "")
            if tt == MAZE_TASK_TYPE:
                return "maze" in tts_tasks
            if tt.startswith("spar_"):
                return "spar" in tts_tasks
            return "vsi" in tts_tasks

        apply_search = bool(tts_enabled and tts_branch > 1 and _tts_applies(step_task))
        effective_n = base_n * (tts_branch if apply_search else 1)
        if apply_search:
            gen_batch.meta_info["n"] = effective_n
            print(
                f"[tts] step_task={step_task} branch={tts_branch} "
                f"effective_n={effective_n} keep={base_n}"
            )

        gen_batch, gen_pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_ref_wg.world_size)
        gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
        gen_batch_output = unpad_dataproto(
            gen_batch_output,
            pad_size=gen_pad_size * effective_n,
        )

        if self.config.algorithm.adv_estimator == "remax":
            gen_baseline_batch = deepcopy(gen_batch)
            gen_baseline_batch.meta_info["temperature"] = 0
            gen_baseline_batch.meta_info["n"] = 1
            gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)
            gen_baseline_output = unpad_dataproto(gen_baseline_output, pad_size=gen_pad_size)
            
            new_batch = new_batch.union(gen_baseline_output)
            reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
            
            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
            new_batch.batch["reward_baselines"] = reward_baseline_tensor
            del gen_baseline_batch, gen_baseline_output
        
        new_batch = new_batch.repeat(repeat_times=effective_n, interleave=True)
        new_batch = new_batch.union(gen_batch_output)

        # VERSE note:VGS-GRPO selection: keep base_n engine-selected
        # rollouts per uid group (class-balanced + fidelity + variance fallback).
        if apply_search and effective_n > base_n:
            from .tts_search import select_for_batch
            keep_idxs, tts_metrics = select_for_batch(
                new_batch,
                group_size=effective_n,
                keep=base_n,
                fidelity_weight=tts_fidelity_weight,
                seed=int(getattr(self, "global_step", 0) or 0),
            )
            new_batch = new_batch[keep_idxs]
            metrics.update({f"tts/{k}": v for k, v in tts_metrics.items()})

        all_metrics = defaultdict(list)
        if self.config.algorithm.online_filtering:
            reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            new_batch.batch["token_level_scores"] = reward_tensor
            for k, v in reward_metrics.items():
                all_metrics[k].extend(v)
            
            filter_scores = reward_metrics[self.config.algorithm.filter_key]
            uids = new_batch.non_tensor_batch["uid"]
            
            uid2scores = defaultdict(list)
            for uid, score in zip(uids, filter_scores):
                uid2scores[uid].append(score)
            
            uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
            
            kept_uids = [
                uid
                for uid, avg_score in uid2mean.items()
                if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
            ]
            kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
            
            if len(kept_sample_idxs) == 0:
                raise RuntimeError("No samples remained after filtering. Please check your data.")
            
            new_batch = new_batch[kept_sample_idxs]
            
            metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
        
        new_batch = self._pad_round2_batch_for_balance(new_batch, metrics=metrics)
        batch_size = len(new_batch)
        new_batch.non_tensor_batch["round"] = np.full(batch_size, 2, dtype=np.int32)

        return new_batch

    def _decode_responses_from_batch(self, gen_output: DataProto) -> list[str]:
        responses = []
        response_ids = gen_output.batch["responses"]
        response_mask = gen_output.batch["response_mask"]
        
        for i in range(len(gen_output)):
            valid_length = int(response_mask[i].sum().item())
            valid_ids = response_ids[i][:valid_length]
            text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            responses.append(text)
        
        return responses

    def _build_answer_batch(
        self, 
        original_batch: DataProto,
        questions: list[str],
        image_overrides: Optional[dict[int, list]] = None,
    ) -> DataProto:
        """Build the Round-2 Solver generation batch in ViLaSR agentic format.

        Every sample gets the ViLaSR system prompt (think->draw->observe contract)
        plus per-image index captions, tokenized against images processed with
        `process_image_agentic` — the exact processing the agentic rollout will
        re-apply, keeping image token counts aligned.
        """
        multi_modal_data = original_batch.non_tensor_batch.get("multi_modal_data", None)
        question_types = original_batch.non_tensor_batch.get("question_type", None)
        validity_flags = original_batch.non_tensor_batch.get("is_valid", None)
        
        input_ids_list = []  # Token ids.
        attention_mask_list = []
        position_ids_list = []
        raw_prompt_ids_list = []  # Tokenized prompt ids.
        new_multi_modal_data = []  # vLLM-ready multimodal payloads.
        round2_modes = []
        round2_task_references = []
        grid_sizes = []
        
        for i, question in enumerate(questions):
            task_type = ""
            if question_types is not None and i < len(question_types):
                task_type = str(question_types[i] or "")

            is_valid_question = True
            if validity_flags is not None and i < len(validity_flags):
                is_valid_question = bool(validity_flags[i])

            image_data = multi_modal_data[i] if multi_modal_data is not None and i < len(multi_modal_data) else None
            media_context = self._build_simulator_media_context(image_data)
            summary_context = self._get_round1_summary_context(media_context)
            grid_size = summary_context.get("maze_grid_size") or None
            grid_sizes.append(int(grid_size) if grid_size else None)

            if is_valid_question:
                body_text = question
                round2_modes.append("answer")
                round2_task_references.append("")
            else:
                body_text, task_reference_text = self._render_round2_invalid_explanation_prompt(
                    question=question,
                    task_type=task_type,
                    summary_context=summary_context,
                )
                round2_modes.append("invalid_explanation")
                round2_task_references.append(task_reference_text)
            
            # Keep scene images on EVERY Round2 sample that has them (valid
            # answers AND invalid explanations). Mixing image and text-only
            # samples within a step desynchronizes per-rank micro-batch counts
            # under padding_free/dynamic_batching and deadlocks NCCL.
            if isinstance(image_data, dict) and image_data.get("images"):
                images = image_data["images"]
                # VERSE note: SPAR marker-rendered images (PIL) replace
                # the originals for the Solver only; image_data itself (paths,
                # media context) stays untouched.
                if image_overrides and i in image_overrides:
                    images = image_overrides[i]
                processed_images = [
                    process_image_agentic(img, self.config.data.min_pixels, self.config.data.max_pixels)
                    for img in images
                ]
                width, height = processed_images[0].size
                system_text = VILASR_SYSTEM_PROMPT.format(width=width, height=height)

                content: list[dict[str, Any]] = []
                if len(processed_images) == 1:
                    # ViLaSR maze/vqa style: single image, index caption inside the question.
                    content.append({"type": "image"})
                    content.append(
                        {
                            "type": "text",
                            "text": VILASR_PROMPT_TEMPLATE.format(
                                question=body_text + "\nThe index of the given image is 1."
                            ),
                        }
                    )
                else:
                    # ViLaSR video style: interleave images with index captions.
                    for img_idx in range(len(processed_images)):
                        content.append({"type": "image"})
                        content.append(
                            {
                                "type": "text",
                                "text": f"The index of the given image is {img_idx + 1} "
                                f"(width: {width}, height: {height}).\n",
                            }
                        )
                    content.append({"type": "text", "text": VILASR_PROMPT_TEMPLATE.format(question=body_text)})

                messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": content},
                ]
                prompt_formatted = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                model_inputs = self.processor(
                    processed_images, [prompt_formatted],
                    add_special_tokens=False, return_tensors="pt"
                )
                new_multi_modal_data.append({"images": images})
            else:
                messages = [{"role": "user", "content": body_text}]
                prompt_formatted = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                model_inputs = self.tokenizer(
                    [prompt_formatted], add_special_tokens=False, return_tensors="pt"
                )
                new_multi_modal_data.append(None)
            
            input_ids = model_inputs["input_ids"][0]  # Token ids, shape `(seq_length,)`.
            attention_mask = model_inputs["attention_mask"][0]  # Attention mask, shape `(seq_length,)`.
            if len(input_ids) > self.config.data.max_prompt_length:
                raise RuntimeError(
                    f"Round-2 agentic prompt has {len(input_ids)} tokens > "
                    f"data.max_prompt_length={self.config.data.max_prompt_length}. "
                    f"Increase max_prompt_length or reduce images/max_pixels. "
                    f"(task_type={task_type}, num_images={len(new_multi_modal_data[-1]['images']) if new_multi_modal_data[-1] else 0})"
                )
            
            if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from ..models.transformers.qwen3_vl import get_rope_index
                else:
                    from ..models.transformers.qwen2_vl import get_rope_index
                
                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw", None),
                    video_grid_thw=model_inputs.get("video_grid_thw", None),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                    attention_mask=attention_mask,
                )  # (3, seq_length)
                
                text_position_ids = torch.arange(len(input_ids), device=input_ids.device).unsqueeze(0)
                
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)
            
            input_ids, attention_mask, position_ids = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",  # Keep the prompt tail when truncating.
            )
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            
            raw_prompt_ids_list.append(
                self.tokenizer.encode(prompt_formatted, add_special_tokens=False)
            )
        
        batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_ids_list),        # (batch_size, max_prompt_length)
                "attention_mask": torch.stack(attention_mask_list),  # (batch_size, max_prompt_length)
                "position_ids": torch.stack(position_ids_list),  # mRoPE may use `(batch_size, 4, max_prompt_length)`.
            },
            non_tensors={
                "raw_prompt_ids": np.array(raw_prompt_ids_list, dtype=object),  # Prompt token ids.
                "multi_modal_data": np.array(new_multi_modal_data, dtype=object),
                "round2_mode": np.array(round2_modes, dtype=object),
                "round2_task_reference": np.array(round2_task_references, dtype=object),
                "grid_size": np.array(grid_sizes, dtype=object),  # Maze extra-LLM-call budget.
            },
            meta_info={
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            },
        )

        return batch

    def _build_answer_messages_with_images(self, prompt_text: str, num_images: int) -> list[dict]:
        content = []
        for _ in range(num_images):
            content.append({"type": "image"})
        
        content.append({"type": "text", "text": prompt_text})
        
        return [{"role": "user", "content": content}]

    def fit(self):
        
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        
        self.global_step = 0
        
        main_tqdm = tqdm(range(self.training_steps), desc="Training steps", position=0)
        
        val_metrics: Optional[dict[str, Any]] = None

        self._load_checkpoint()
        
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            
            self.logger.log(data=val_metrics, step=self.global_step)
            
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            
            with timer("step", timing_raw):
                
                use_two_round_training = getattr(self.config.algorithm, 'two_round_training', False) 
                skip_generic_ppo_update = False

                if use_two_round_training:
                    if getattr(self.config.algorithm, "online_filtering", False):
                        raise ValueError("`two_round_training` currently requires `algorithm.online_filtering=false`.")
                    
                    print(f"========== Round 1 training: question-generation stage (step {self.global_step}) ==========")
                    
                    with timer("gen_round1", timing_raw):
                        if not self._split_role_frozen_proposer:
                            self.actor_rollout_ref_wg.prepare_rollout_engine()
                        round1_batch = self._make_round1_batch_data(metrics=metrics)
                        if not self._split_role_frozen_proposer:
                            self.actor_rollout_ref_wg.release_rollout_engine()
                    
                    self.round1_batch = round1_batch
                    
                    self._balance_batch(round1_batch, metrics=metrics)
                    
                    round1_batch.meta_info["global_token_num"] = torch.sum(round1_batch.batch["attention_mask"], dim=-1).tolist()
                    
                    if "token_level_scores" not in round1_batch.batch:
                        with timer("reward_round1", timing_raw):
                            assert "round" in round1_batch.non_tensor_batch, "Round1 batch must include the `round` field"
                            assert np.all(round1_batch.non_tensor_batch["round"] == 1), "The `round` field in Round1 batch must be 1"
                            quality_reward_ref = self.reward_fn.compute_reward.remote(round1_batch)
                            
                            quality_reward_tensor, quality_reward_metrics = ray.get(quality_reward_ref)

                            quality_reward_metrics_reduced = reduce_metrics(quality_reward_metrics)
                            self._log_generation_io(
                                inputs=round1_batch, # DataProto
                                outputs=round1_batch, # union batch
                                scores=quality_reward_tensor,
                                step=self.global_step, 
                                tag="round1_quality_check"
                            )
                            round1_batch.batch["token_level_scores"] = quality_reward_tensor
                            self._add_per_type_reward_metrics(
                                batch=round1_batch,
                                reward_tensor=quality_reward_tensor,
                                metrics=metrics,
                                metric_prefix="reward_round1",
                            )
                            
                            quality_overall = quality_reward_metrics_reduced.get("overall", 0.0)

                            quality_reward_per_sample = quality_reward_tensor.sum(dim=-1).cpu().tolist()
                            quality_reward_stats = {
                                "max": np.max(quality_reward_per_sample),
                                "min": np.min(quality_reward_per_sample),
                            }
                            
                            reward_metrics_round1 = {
                                f"reward_round1/quality": quality_overall,
                                f"reward_round1/quality_max": quality_reward_stats["max"],
                                f"reward_round1/quality_min": quality_reward_stats["min"],
                            }
                            
                            for k, v in quality_reward_metrics_reduced.items():
                                if k != "overall":
                                    reward_metrics_round1[f"reward_round1/quality_{k}"] = v

                            metrics.update(reward_metrics_round1)
                    
                    if self._split_role_frozen_proposer or self._round1_skip_update:
                        if self._split_role_frozen_proposer:
                            metrics["split_role/frozen_proposer"] = 1.0
                        metrics["split_role/round1_actor_update_skipped"] = 1.0
                    else:
                        with timer("old_round1", timing_raw):
                            old_log_probs_round1 = self.actor_rollout_ref_wg.compute_log_probs(round1_batch)
                            round1_batch = round1_batch.union(old_log_probs_round1)
                        with timer("release_gpu_post_old_round1", timing_raw):
                            self.actor_rollout_ref_wg.release_gpu_for_rollout()
                        
                        if self.use_reference_policy:
                            with timer("ref_round1", timing_raw):
                                ref_log_probs_round1 = self.actor_rollout_ref_wg.compute_ref_log_probs(round1_batch)
                                round1_batch = round1_batch.union(ref_log_probs_round1)
                            with timer("release_gpu_pre_update_round1", timing_raw):
                                self.actor_rollout_ref_wg.release_gpu_for_rollout()
                        
                        if self.use_critic:
                            with timer("values_round1", timing_raw):
                                values_round1 = self.critic_wg.compute_values(round1_batch)
                                round1_batch = round1_batch.union(values_round1)
                        
                        with timer("adv_round1", timing_raw):
                            if "token_level_scores" not in round1_batch.batch:
                                reward_ref_round1 = self.reward_fn.compute_reward.remote(round1_batch)
                                reward_tensor_round1, reward_metrics_round1 = ray.get(reward_ref_round1)
                                round1_batch.batch["token_level_scores"] = reward_tensor_round1
                                reward_metrics_round1 = {f"reward_round1/{k}": v for k, v in reduce_metrics(reward_metrics_round1).items()}
                                metrics.update(reward_metrics_round1)
                            
                            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                                round1_batch, kl_metrics_round1 = apply_kl_penalty(round1_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                                kl_metrics_round1 = {f"{k}_round1": v for k, v in kl_metrics_round1.items()}
                                metrics.update(kl_metrics_round1)
                            else:
                                round1_batch.batch["token_level_rewards"] = round1_batch.batch["token_level_scores"]
                            
                            round1_batch = compute_advantage(
                                round1_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                            )
                        
                        if self.use_critic:
                            with timer("update_critic_round1", timing_raw):
                                critic_output_round1 = self.critic_wg.update_critic(round1_batch)
                            critic_metrics_round1 = {f"{k}_round1": v for k, v in reduce_metrics(critic_output_round1.non_tensor_batch).items()}
                            metrics.update(critic_metrics_round1)
                        
                        if self.config.trainer.critic_warmup <= self.global_step:
                            with timer("update_actor_round1", timing_raw):
                                actor_output_round1 = self.actor_rollout_ref_wg.update_actor(round1_batch)
                            actor_metrics_round1 = {f"{k}_round1": v for k, v in reduce_metrics(actor_output_round1.non_tensor_batch).items()}
                            metrics.update(actor_metrics_round1)
                    
                    print(f"========== Round 2 training: answering stage (step {self.global_step}) ==========")
                    
                    # Extra flush after Round1 update before vLLM re-wake (rank1 OOM on prepare_rollout_engine).
                    self.actor_rollout_ref_wg.release_gpu_for_rollout()
                    
                    with timer("gen_round2", timing_raw):
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        round2_batch = self._make_round2_batch_data(metrics=metrics)
                        self.actor_rollout_ref_wg.release_rollout_engine()
                    
                    del self.round1_batch
                    
                    self._balance_batch(round2_batch, metrics=metrics)
                        
                    round2_batch.meta_info["global_token_num"] = torch.sum(round2_batch.batch["attention_mask"], dim=-1).tolist()
                        
                    if "token_level_scores" not in round2_batch.batch:
                        with timer("reward_round2", timing_raw):
                            assert "round" in round2_batch.non_tensor_batch, "Round2 batch must include the `round` field"
                            assert np.all(round2_batch.non_tensor_batch["round"] == 2), "The `round` field in Round2 batch must be 2"
                            assert "uid" in round2_batch.non_tensor_batch, "Round2 batch must include the `uid` field (for GRPO grouping and per-question stats)"
                            reward_ref_round2 = self.reward_fn.compute_reward.remote(round2_batch)
                        
                    with timer("old_round2", timing_raw):
                        old_log_probs_round2 = self.actor_rollout_ref_wg.compute_log_probs(round2_batch)
                        round2_batch = round2_batch.union(old_log_probs_round2)
                    with timer("release_gpu_post_old_round2", timing_raw):
                        self.actor_rollout_ref_wg.release_gpu_for_rollout()
                        
                    if self.use_reference_policy:
                        with timer("ref_round2", timing_raw):
                            ref_log_probs_round2 = self.actor_rollout_ref_wg.compute_ref_log_probs(round2_batch)
                            if "ref_log_probs" in round2_batch.batch:
                                round2_batch.batch["ref_log_probs"] = ref_log_probs_round2.batch["ref_log_probs"]
                            else:
                                round2_batch = round2_batch.union(ref_log_probs_round2)
                        with timer("release_gpu_pre_update_round2", timing_raw):
                            self.actor_rollout_ref_wg.release_gpu_for_rollout()
                        
                    if self.use_critic:
                        with timer("values_round2", timing_raw):
                            values_round2 = self.critic_wg.compute_values(round2_batch)
                            round2_batch = round2_batch.union(values_round2)

                    sft_skip_update = False
                    with timer("adv_round2", timing_raw):
                        reward_tensor_round2 = round2_batch.batch.get("token_level_scores")
                        if "token_level_scores" not in round2_batch.batch:
                            reward_tensor_round2, reward_metrics_round2 = ray.get(reward_ref_round2)
                            filtered_reward_metrics_round2 = reward_metrics_round2
                            if "is_padding" in round2_batch.non_tensor_batch:
                                padding_mask = np.asarray(round2_batch.non_tensor_batch["is_padding"], dtype=bool)
                                if padding_mask.any():
                                    reward_tensor_round2 = reward_tensor_round2.clone()
                                    reward_tensor_round2[torch.from_numpy(padding_mask).to(reward_tensor_round2.device)] = 0
                                    filtered_reward_metrics_round2 = self._filter_reward_metrics_for_padding(
                                        reward_metrics_round2,
                                        padding_mask=padding_mask,
                                    )
                            self._update_round1_task_stats(round2_batch, reward_metrics_round2)
                            self._add_round1_task_scheduler_metrics(metrics)
                            self._log_round2_valid_qa(
                                round2_batch,
                                reward_metrics_round2,
                                step=self.global_step,
                            )
                            if self._traj_bank_enabled:
                                metrics["traj_bank/saved_count"] = float(
                                    self._append_trajectory_bank(
                                        round2_batch,
                                        reward_metrics_round2,
                                        step=self.global_step,
                                    )
                                )
                            round2_batch.batch["token_level_scores"] = reward_tensor_round2
                            # ---- SFT-on-correct-rollouts (STaR/RFT) filter ----
                            if self._sft_on_correct:
                                accs = reward_metrics_round2.get(
                                    "accuracy", reward_metrics_round2.get("answer_correctness", [])
                                )
                                bs_rm = len(round2_batch)
                                if not isinstance(accs, list) or len(accs) != bs_rm:
                                    accs = [None] * bs_rm
                                if "is_valid" in round2_batch.non_tensor_batch:
                                    is_valid_np = np.asarray(round2_batch.non_tensor_batch["is_valid"], dtype=bool)
                                else:
                                    is_valid_np = np.ones(bs_rm, dtype=bool)
                                correct = np.zeros(bs_rm, dtype=bool)
                                for _i, _a in enumerate(accs):
                                    if _a is None:
                                        continue
                                    try:
                                        _af = float(_a)
                                    except (TypeError, ValueError):
                                        continue
                                    ok_acc = _af >= self._sft_correct_threshold
                                    ok_valid = bool(is_valid_np[_i]) if not self._sft_include_invalid else True
                                    correct[_i] = ok_acc and ok_valid
                                num_correct = int(correct.sum())
                                if "response_mask" in round2_batch.batch:
                                    _rm = round2_batch.batch["response_mask"].clone()
                                    _rm[~torch.from_numpy(correct).to(_rm.device)] = 0
                                    round2_batch.batch["response_mask"] = _rm
                                round2_batch.meta_info["sft_mode"] = True
                                sft_skip_update = num_correct == 0
                                metrics["sft/num_correct"] = float(num_correct)
                                metrics["sft/num_total"] = float(bs_rm)
                                metrics["sft/correct_ratio"] = (float(num_correct) / float(bs_rm)) if bs_rm > 0 else 0.0
                                metrics["sft/skipped_update"] = 1.0 if sft_skip_update else 0.0
                            reward_metrics_round2 = {
                                f"reward_round2/{k}": v for k, v in reduce_metrics(filtered_reward_metrics_round2).items()
                            }
                            metrics.update(reward_metrics_round2)
                        
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            round2_batch, kl_metrics_round2 = apply_kl_penalty(round2_batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            kl_metrics_round2 = {f"{k}_round2": v for k, v in kl_metrics_round2.items()}
                            metrics.update(kl_metrics_round2)
                        else:
                            round2_batch.batch["token_level_rewards"] = round2_batch.batch["token_level_scores"]
                        
                        self._log_generation_io(
                            inputs=round2_batch, 
                            outputs=round2_batch, 
                            scores=reward_tensor_round2, 
                            step=self.global_step, 
                            tag="round2_answer_check"
                        )
                        self._add_per_type_reward_metrics(
                            batch=round2_batch,
                            reward_tensor=reward_tensor_round2,
                            metrics=metrics,
                            metric_prefix="reward_round2",
                        )

                        if "is_valid" in round2_batch.non_tensor_batch:
                            is_valid_np = np.asarray(round2_batch.non_tensor_batch["is_valid"], dtype=bool)
                            if "is_padding" in round2_batch.non_tensor_batch:
                                is_padding_np = np.asarray(round2_batch.non_tensor_batch["is_padding"], dtype=bool)
                                is_valid_np = is_valid_np[~is_padding_np]
                            invalid_np = ~is_valid_np
                            invalid_count = int(invalid_np.sum())
                            total_count = int(len(is_valid_np))
                            metrics["round2/invalid_count"] = float(invalid_count)
                            metrics["round2/invalid_ratio"] = (float(invalid_count) / float(total_count)) if total_count > 0 else 0.0

                        round2_batch = self._pad_round2_batch_for_updates(round2_batch, metrics=metrics)
                        round2_batch.meta_info["global_token_num"] = torch.sum(
                            round2_batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        if not self._sft_on_correct:
                            round2_batch = compute_advantage(
                                round2_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                            )
                        else:
                            # SFT mode: no GRPO advantages needed; supply zero placeholders so any
                            # downstream reader (e.g. compute_data_metrics) finds the keys. The SFT
                            # actor loss ignores advantages; returns only feeds (unused) metrics.
                            round2_batch.batch["token_level_rewards"] = round2_batch.batch["token_level_scores"]
                            round2_batch.batch["advantages"] = torch.zeros_like(round2_batch.batch["token_level_scores"])
                            round2_batch.batch["returns"] = torch.zeros_like(round2_batch.batch["token_level_scores"])
                    
                    if self.use_critic:
                        with timer("update_critic_round2", timing_raw):
                            critic_output_round2 = self.critic_wg.update_critic(round2_batch)
                        critic_metrics_round2 = {f"{k}_round2": v for k, v in reduce_metrics(critic_output_round2.non_tensor_batch).items()}
                        metrics.update(critic_metrics_round2)
                    
                    if self.config.trainer.critic_warmup <= self.global_step and not sft_skip_update and not self._sft_collect_only:
                        with timer("update_actor_round2", timing_raw):
                            actor_output_round2 = self.actor_rollout_ref_wg.update_actor(round2_batch)
                        actor_metrics_round2 = {f"{k}_round2": v for k, v in reduce_metrics(actor_output_round2.non_tensor_batch).items()}
                        metrics.update(actor_metrics_round2)
                    
                    batch = round2_batch
                    skip_generic_ppo_update = True
                    
                else:
                    with timer("gen", timing_raw):
                        self.actor_rollout_ref_wg.prepare_rollout_engine()
                        
                        batch = self._make_batch_data(metrics=metrics)
                        
                        self.actor_rollout_ref_wg.release_rollout_engine()

                if not skip_generic_ppo_update:
                    self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if "token_level_scores" not in batch.batch:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(batch)

                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        
                        if "old_log_probs" in batch.batch:
                            batch.batch["old_log_probs"] = old_log_probs.batch["old_log_probs"]
                        else:
                            batch = batch.union(old_log_probs)
                    with timer("release_gpu_post_old", timing_raw):
                        self.actor_rollout_ref_wg.release_gpu_for_rollout()

                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)

                            if "ref_log_probs" in batch.batch:
                                batch.batch["ref_log_probs"] = ref_log_probs.batch["ref_log_probs"]
                            else:
                                batch = batch.union(ref_log_probs)
                        with timer("release_gpu_pre_update", timing_raw):
                            self.actor_rollout_ref_wg.release_gpu_for_rollout()

                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            
                            batch = batch.union(values)

                    with timer("adv", timing_raw):
                        if "token_level_scores" not in batch.batch:
                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            
                            batch.batch["token_level_scores"] = reward_tensor
                            
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            
                            metrics.update(reward_metrics)

                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        
                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        
                        metrics.update(critic_metrics)

                    if self.config.trainer.critic_warmup <= self.global_step:
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                        
                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        
                        metrics.update(actor_metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0 # > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            
            metrics_batch = self._strip_padding_for_metrics(batch)
            metrics.update(compute_data_metrics(batch=metrics_batch, use_critic=self.use_critic))
            
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            
            main_tqdm.update()

        # VERSE note: only run a final validation when periodic validation is enabled
        # (val_freq > 0). Our single-task runs use val_freq=-1 (no val; we eval via
        # VSI-Bench separately), so this whole block is skipped -> the forced final
        # validation can never crash before the final checkpoint save. The try/except
        # below is kept as a safety net for val_freq>0 runs, because the val set can
        # contain >8-image samples that vLLM rejects ("At most 8 images"), which
        # previously crashed fit() right before the save -> supervisor restart loop.
        if self.val_reward_fn is not None and self.config.trainer.val_freq > 0:
            try:
                if (
                    val_metrics is None
                    or self.config.trainer.val_freq <= 0
                    or self.global_step % self.config.trainer.val_freq != 0
                ):
                    val_metrics = self._validate()

                    self.logger.log(data=val_metrics, step=self.global_step)

                print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")
            except Exception as _val_exc:
                print(f"[Warning] Final validation failed (non-fatal); proceeding to save checkpoint: {_val_exc}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
