"""Structured question renderer for Phase-2 (structured_imitate) Proposer.

The Phase-2 Proposer only picks object labels (the task_type is pinned per step
by the round-robin schedule). This module renders the final GRPO-style question
text from those picks + the per-task templates already defined in
``question_generate_task_conditioned_specs.json``.

Mapping: DGE input field names (what WorldSimulator.validate_and_answer expects)
  <-> template variable names (what the jinja templates use).

This is deterministic and side-effect free, so validity of the rendered question
is guaranteed by construction as long as the Proposer's labels pass DGE.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
SPECS_PATH = os.path.join(
    HERE, "..", "..", "easy_r1", "training", "format_prompt",
    "question_generate_task_conditioned_specs.json",
)

# Per-task config:
#   dge_to_template: maps DGE input field -> template variable name
#   answer_style: "numerical" | "mc"
#   mc_option_fields: for mc tasks, the DGE fields holding option labels (in order)
TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "object_counting": {
        "dge_to_template": {"target_category": "object_label"},
        "answer_style": "numerical",
    },
    "absolute_distance": {
        "dge_to_template": {"object1_label": "object_a", "object2_label": "object_b"},
        "answer_style": "numerical",
    },
    "object_size": {
        "dge_to_template": {"object_label": "object_label"},
        "answer_style": "numerical",
    },
    "room_size": {
        "dge_to_template": {},
        "answer_style": "numerical",
    },
    "relative_distance": {
        # VSI-Bench object_rel_distance always uses 4 candidate objects.
        "dge_to_template": {
            "target_label": "anchor_object",
            "candidate_labels": ["object_a", "object_b", "object_c", "object_d"],
        },
        "answer_style": "mc",
        "mc_option_fields": ["candidate_labels"],
    },
    "relative_direction_hard": {
        "dge_to_template": {
            "positioning_label": "stand_object",
            "orienting_label": "face_object",
            "querying_label": "query_object",
        },
        "answer_style": "mc",
        "mc_option_fields": [],
    },
    "route_planning": {
        "dge_to_template": {
            "waypoint_labels": ["start_object", "second_object", "third_object", "destination_object"],
        },
        "answer_style": "mc",
        "mc_option_fields": [],
    },
    "single_image_relative_direction": {
        "dge_to_template": {"target_label": "target_object", "reference_label": "reference_object"},
        "answer_style": "mc",
        "mc_option_fields": [],
    },
    "distance_cam_obj": {
        "dge_to_template": {"target_label": "object_label"},
        "answer_style": "numerical",
    },
    "depth_order_obj_obj": {
        "dge_to_template": {"object1_label": "object_a", "object2_label": "object_b"},
        "answer_style": "mc",
        "mc_option_fields": [],
    },
    "spar_distance_prediction_oc": {
        "dge_to_template": {"target_label": "object_label"},
        "answer_style": "numerical",
    },
    "spar_distance_prediction_oc_mv": {
        "dge_to_template": {"target_label": "object_label"},
        "answer_style": "numerical",
    },
    "spar_distance_prediction_oo": {
        "dge_to_template": {"object1_label": "object_a", "object2_label": "object_b"},
        "answer_style": "numerical",
    },
    "spar_distance_prediction_oo_mv": {
        "dge_to_template": {"object1_label": "object_a", "object2_label": "object_b"},
        "answer_style": "numerical",
    },
    "spar_depth_prediction_oc": {
        "dge_to_template": {"object1_label": "object_a", "object2_label": "object_b"},
        "answer_style": "numerical",
    },
    "spar_depth_prediction_oc_mv": {
        "dge_to_template": {"object1_label": "object_a", "object2_label": "object_b"},
        "answer_style": "numerical",
    },
    "spar_depth_prediction_oo": {
        "dge_to_template": {"object1_label": "object_a", "object2_label": "object_b"},
        "answer_style": "numerical",
    },
    "spar_depth_prediction_oo_mv": {
        "dge_to_template": {"object1_label": "object_a", "object2_label": "object_b"},
        "answer_style": "numerical",
    },
    "spar_obj_spatial_relation_oo": {
        "dge_to_template": {"target_label": "target_object", "reference_label": "reference_object"},
        "answer_style": "mc",
        "mc_option_fields": [],
    },
    "spar_obj_spatial_relation_oc_mv": {
        "dge_to_template": {"target_label": "target_object", "reference_label": "reference_object"},
        "answer_style": "mc",
        "mc_option_fields": [],
    },
    "spar_obj_spatial_relation_oo_mv": {
        "dge_to_template": {"target_label": "target_object", "reference_label": "reference_object"},
        "answer_style": "mc",
        "mc_option_fields": [],
    },
    "spar_distance_infer_center_oo": {
        "dge_to_template": {
            "anchor_label": "anchor_object",
            "option_labels": ["option_a_object", "option_b_object"],
        },
        "answer_style": "mc",
        "mc_option_fields": ["option_labels"],
    },
    "spar_distance_infer_center_oo_mv": {
        "dge_to_template": {
            "anchor_label": "anchor_object",
            "option_labels": ["option_a_object", "option_b_object"],
        },
        "answer_style": "mc",
        "mc_option_fields": ["option_labels"],
    },
}


def _load_specs() -> Dict[str, Any]:
    with open(SPECS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_SPECS_CACHE: Optional[Dict[str, Any]] = None


def _specs() -> Dict[str, Any]:
    global _SPECS_CACHE
    if _SPECS_CACHE is None:
        _SPECS_CACHE = _load_specs()
    return _SPECS_CACHE


def _fill_template(template: str, var_map: Dict[str, str]) -> str:
    """Fill a {var} template (single-brace, no jinja) with var_map.

    If a template variable is not in var_map, replace it with a generic
    fallback so the literal {var_name} does not leak into the question text.
    """
    import re as _re
    out = template
    # First, fill all known variables
    for k, v in var_map.items():
        out = out.replace("{" + k + "}", str(v))
    # Then, replace any remaining {var_name} placeholders that were not filled
    # (e.g. {object_c} when the Proposer gave fewer candidates than expected)
    def _replace_unfilled(m):
        var = m.group(1)
        # If it looks like a template variable (alphanumeric + underscore),
        # replace with a neutral word rather than leaking the placeholder.
        if _re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", var):
            return "another object"
        return m.group(0)
    out = _re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", _replace_unfilled, out)
    return out


def _build_var_map(task_type: str, params: Dict[str, Any]) -> Dict[str, str]:
    """Map DGE input fields (in params) -> template variable names."""
    cfg = TASK_CONFIG.get(task_type)
    if cfg is None:
        return {}
    var_map: Dict[str, str] = {}
    for dge_field, tmpl_var in cfg["dge_to_template"].items():
        val = params.get(dge_field)
        if val is None:
            continue
        if isinstance(tmpl_var, list):
            # list-valued DGE field (e.g. candidate_labels) -> multiple template vars
            if isinstance(val, list):
                for i, var_name in enumerate(tmpl_var):
                    if i < len(val):
                        var_map[var_name] = str(val[i])
        else:
            var_map[tmpl_var] = str(val)
    return var_map


def render_question(
    task_type: str,
    params: Dict[str, Any],
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, Optional[List[str]], Optional[str]]:
    """Render a GRPO-style question from structured params.

    Returns (question_text, answer_style, options_json_or_None, gt_letter_or_None).
      - For numerical tasks: options=None, gt_letter=None.
      - For mc tasks: options is a list like ["A) label1", "B) label2", ...] and
        gt_letter is the letter of the correct option (caller fills it from DGE GT).
    """
    specs = _specs()
    task_spec = specs.get("tasks", {}).get(task_type, {})
    templates = task_spec.get("question_templates") or []
    if not templates:
        return "", "numerical", None, None
    tmpl = templates[0] if rng is None else rng.choice(templates)
    var_map = _build_var_map(task_type, params)
    question_text = _fill_template(tmpl, var_map)

    cfg = TASK_CONFIG.get(task_type, {})
    answer_style = cfg.get("answer_style", "numerical")
    options: Optional[List[str]] = None
    gt_letter: Optional[str] = None
    if answer_style == "mc":
        mc_fields = cfg.get("mc_option_fields", [])
        opt_labels: List[str] = []
        for field in mc_fields:
            v = params.get(field)
            if isinstance(v, list):
                opt_labels.extend(str(x) for x in v)
            elif isinstance(v, str):
                opt_labels.append(v)
        if opt_labels:
            r = rng or random.Random()
            shuffled = list(opt_labels)
            r.shuffle(shuffled)
            options = [f"{chr(ord('A') + i)}) {lbl}" for i, lbl in enumerate(shuffled)]
            # gt_letter must be set by caller using DGE GT vs the shuffled order.
    return question_text, answer_style, options, gt_letter


def map_params_to_dge_input(task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through: the Proposer already outputs DGE input fields directly."""
    return dict(params)


if __name__ == "__main__":
    # smoke
    for tt, p in [
        ("object_counting", {"target_category": "chair"}),
        ("absolute_distance", {"object1_label": "sofa", "object2_label": "table"}),
        ("object_size", {"object_label": "door"}),
        ("relative_distance", {"target_label": "bed", "candidate_labels": ["chair", "window", "lamp"]}),
    ]:
        q, style, opts, _ = render_question(tt, p)
        print(f"[{tt}] style={style}")
        print(f"  Q: {q}")
        if opts:
            print(f"  opts: {opts}")
        print()
