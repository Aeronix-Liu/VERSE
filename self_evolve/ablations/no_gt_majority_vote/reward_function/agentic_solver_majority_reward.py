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

"""ABLATION #1 -- Self-evolve WITHOUT ground-truth: majority-voting Round-2 reward.

This is the ablation twin of `agentic_solver_reward.py`. It is IDENTICAL in every
respect (gcot multi-turn format check, invalid-question judge branch, maze /
VSI-grounding process diagnostics, GRPO diversity metrics) EXCEPT for the one
signal we are ablating:

    Original (with GT):  r_acc = accuracy( answer_i , engine_GT )
    This ablation:       r_acc = accuracy( answer_i , majority_vote(group) )

i.e. the deterministic engine's ground-truth ANSWER is discarded for the Solver
reward. Instead, within each GRPO group (one `uid` == one verified question,
n=8 rollouts), we estimate a self-consistency pseudo-label by majority voting
over the rollout answers and reward each rollout by agreement with it (TTRL /
self-consistency RL). The engine is still used to VERIFY question validity and to
RENDER the question text / MC options (see the run's design note) -- only the GT
answer is removed from the reward.

Consensus rule per group (valid rollouts only):
  * multiple-choice / maze : mode of the option LETTERS.
  * object_counting        : mode of the integer answers.
  * other numeric tasks    : median of the parsed float answers.
  * otherwise (text)       : mode of the normalized answer strings.
The pseudo-label is then fed through the SAME `_accuracy_for_valid` scorer used
by the original reward, so MC letter matching, partial direction credit, and MRA
for numeric tasks all behave exactly as in production.

Diagnostics added on top of the base metrics (never enter the reward):
  * majority_vote/consensus_ratio    -- fraction of the group agreeing w/ vote.
  * majority_vote/group_valid_size   -- #valid rollouts used to form the vote.
  * majority_vote/pseudo_vs_true_acc -- pseudo-label accuracy against the engine
                                        GT (measures how good self-consistency is;
                                        NOT used for training).

NOTE: The original `agentic_solver_reward.py` and every other production file are
UNTOUCHED. This file lives entirely under `ablations/no_gt_majority_vote/`.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

# --- Make the ORIGINAL reward_function package importable (we reuse it wholesale).
_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
_ORIG_REWARD_DIR = os.path.abspath(
    os.path.join(
        _ABL_DIR,
        "..", "..", "..", "easy_r1", "training", "reward_function",
    )
)
# Allow an explicit override (robust to repo relocation) via env var.
_ORIG_REWARD_DIR = os.environ.get("SPATIALEVO_ORIG_REWARD_DIR", _ORIG_REWARD_DIR)
if _ORIG_REWARD_DIR not in sys.path:
    sys.path.insert(0, _ORIG_REWARD_DIR)

# Reuse EVERYTHING from the production Round-2 reward so behavior stays identical
# outside the ablated signal.
import agentic_solver_reward as base  # noqa: E402
from majority_correctness_reward import (  # noqa: E402
    COUNT_TASK_TYPES,
    _normalize_exact_answer,
    to_float,
)

try:  # numpy is always available in this env; keep a pure-python fallback anyway.
    import numpy as _np

    def _median(vals: list[float]) -> float:
        return float(_np.median(vals))
except Exception:  # pragma: no cover
    def _median(vals: list[float]) -> float:
        s = sorted(vals)
        m = len(s)
        if m == 0:
            return 0.0
        if m % 2 == 1:
            return float(s[m // 2])
        return float((s[m // 2 - 1] + s[m // 2]) / 2.0)


REWARD_NAME = "agentic_solver_majority"
REWARD_TYPE = "batch"


# --------------------------------------------------------------------------
# Majority-vote consensus (the ablated pseudo-label)
# --------------------------------------------------------------------------

def _mode(items: list) -> Optional[Any]:
    """Deterministic mode: highest count, ties broken by sorted key order."""
    if not items:
        return None
    counts = Counter(items)
    top = max(counts.values())
    tied = sorted([k for k, c in counts.items() if c == top], key=lambda x: str(x))
    return tied[0]


def _num_to_str(value: float) -> str:
    """Render a numeric consensus as a clean string for the scorer's to_float()."""
    if float(value).is_integer():
        return str(int(round(value)))
    return repr(float(value))


def _group_consensus(
    answers: list[str],
    *,
    q_type: str,
    answer_style: str,
    mc_options_json: str,
) -> tuple[Optional[str], str]:
    """Return (pseudo_gt, pseudo_gt_letter) estimated by majority vote.

    `pseudo_gt` is a string that plays the role the engine GT normally plays when
    passed to `base._accuracy_for_valid`; `pseudo_gt_letter` is the consensus MC
    letter (empty for numeric/text). Returns (None, "") when no answer parses.
    """
    is_mc = (answer_style == "multiple choice") or (q_type == "maze_navigation")

    if is_mc:
        letters: list[str] = []
        for a in answers:
            letter, _ = base._letter_to_option_label(a, mc_options_json)
            if letter is None:
                s = str(a or "").strip().upper()
                letter = s[0] if (s[:1] and s[0] in "ABCDEFGH") else None
            if letter:
                letters.append(letter)
        consensus_letter = _mode(letters)
        if consensus_letter is None:
            return None, ""
        # Map the consensus letter back to its option label so non-maze MC tasks
        # (which may award partial credit via the label) score consistently.
        pseudo_gt: Optional[str] = None
        try:
            opts = json.loads(mc_options_json) if mc_options_json else None
        except Exception:
            opts = None
        if isinstance(opts, list):
            idx = ord(consensus_letter) - ord("A")
            if 0 <= idx < len(opts):
                pseudo_gt = str(opts[idx])
        if q_type == "maze_navigation":
            # The maze scorer compares the answer letter directly to `gt`.
            return consensus_letter, consensus_letter
        return (pseudo_gt if pseudo_gt is not None else consensus_letter), consensus_letter

    # ---- numeric / text tasks ----
    floats = [to_float(a) for a in answers]
    floats = [f for f in floats if f is not None]

    if q_type in COUNT_TASK_TYPES and floats:
        consensus_int = _mode([int(round(f)) for f in floats])
        return _num_to_str(float(consensus_int)), ""

    # Continuous numeric: use the median when at least half the group parsed.
    if floats and len(floats) >= max(1, (len(answers) + 1) // 2):
        return _num_to_str(_median(floats)), ""

    # Fallback: mode of normalized answer strings (route sequences, exact-match).
    normed = [_normalize_exact_answer(a, q_type) for a in answers if str(a or "").strip()]
    consensus_txt = _mode(normed)
    if consensus_txt is None or consensus_txt == "":
        return None, ""
    return str(consensus_txt), ""


# --------------------------------------------------------------------------
# Batch entry point (mirrors base.compute_score; only the valid-accuracy path
# is swapped for the majority-vote pseudo-label).
# --------------------------------------------------------------------------

def compute_score(reward_inputs: list[dict[str, Any]], **kwargs) -> list[base.RewardScore]:
    judge_url = kwargs.get("invalid_reason_judge_url", kwargs.get("service_url", kwargs.get("url", "https://example.com/v1")))
    judge_timeout = kwargs.get("invalid_reason_judge_timeout", kwargs.get("timeout", 30))
    judge_model = kwargs.get("invalid_reason_judge_model", kwargs.get("model", "gpt-oss-120b-ldm"))
    judge_api_key = kwargs.get("invalid_reason_judge_api_key", kwargs.get("api_key", ""))
    judge_max_workers = base._resolve_invalid_reason_judge_max_workers(kwargs)

    sample_details: list[dict[str, Any]] = []
    invalid_jobs: list[int] = []

    # First pass: identical to base except we DEFER valid-accuracy scoring until
    # the per-uid majority vote is known. We also stash per-sample metadata used
    # to build the vote and (diagnostic-only) the engine GT.
    for idx, reward_input in enumerate(reward_inputs):
        uid = reward_input.get("uid", f"unknown_{idx}")
        gt = reward_input.get("gt", "")  # kept ONLY for the pseudo-vs-true diag.
        q_type = str(reward_input.get("question_type", "") or "")
        is_valid = bool(reward_input.get("is_valid", True))
        sequence_text = str(reward_input.get("agentic_sequence_text", "") or "")
        response_text = str(reward_input.get("agentic_response_text", "") or reward_input.get("response", "") or "")

        if sequence_text:
            conversations = base.parse_dialog(sequence_text)
            fmt, reflection = base.format_reward(conversations)
        else:
            conversations = []
            fmt, reflection = 0.0, 0.0

        answer_text = base._extract_last_answer_tag(response_text) or ""
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
            # ablation bookkeeping:
            "q_type": q_type,
            "gt": gt,
            "answer_text": answer_text,
            "answer_style": str(reward_input.get("answer_style", "") or ""),
            "mc_options_json": str(reward_input.get("mc_options_json", "") or ""),
            "gt_letter": str(reward_input.get("gt_letter", "") or ""),  # engine GT letter (diag only)
            "pseudo_gt": None,
            "pseudo_gt_letter": "",
            "true_acc": None,
        }

        # Engine-verified maze process signal (diagnostic unless weight>0).
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
                    _, movement_list, _, _ = base.parse_bbox_and_movement(assistant_text)
                    detail["maze_fidelity"] = base.maze_trajectory_fidelity(movement_list, ref)

        # Engine-verified VSI grounding process signal (diagnostic unless weight>0).
        if q_type in base.VSI_GROUNDING_TASKS and is_valid:
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
                    bbox_list, _, _, _ = base.parse_bbox_and_movement(assistant_text)
                    detail["vsi_grounding"] = base.vsi_grounding_fidelity(bbox_list, gref)

        sample_details.append(detail)

        if not is_valid:
            # Invalid questions: identical judge-the-explanation branch as base.
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
            if explanation:
                invalid_jobs.append(len(sample_details) - 1)

    # ---- Majority vote per uid (valid rollouts only) ----
    group_valid_idx: dict[Any, list[int]] = defaultdict(list)
    for di, d in enumerate(sample_details):
        if d["is_valid"]:
            group_valid_idx[d["uid"]].append(di)

    consensus_ratios: list[float] = []
    group_valid_sizes: list[float] = []
    for uid, idxs in group_valid_idx.items():
        answers = [sample_details[i]["answer_text"] for i in idxs]
        meta = sample_details[idxs[0]]
        pseudo_gt, pseudo_letter = _group_consensus(
            answers,
            q_type=meta["q_type"],
            answer_style=meta["answer_style"],
            mc_options_json=meta["mc_options_json"],
        )
        agree = 0
        for i in idxs:
            d = sample_details[i]
            d["pseudo_gt"] = pseudo_gt
            d["pseudo_gt_letter"] = pseudo_letter
            if pseudo_gt is None:
                d["r_acc"] = 0.0
            else:
                d["r_acc"] = float(
                    base._accuracy_for_valid(
                        d["answer_text"],
                        pseudo_gt,
                        d["q_type"],
                        mc_options_json=d["mc_options_json"],
                        gt_letter=pseudo_letter,
                        answer_style=d["answer_style"],
                    )
                )
            if d["r_acc"] >= 0.999:
                agree += 1
            # Diagnostic ONLY: how well the pseudo-label matches the engine GT.
            if pseudo_gt is not None and str(d["gt"]).strip():
                d["true_acc"] = float(
                    base._accuracy_for_valid(
                        d["answer_text"],
                        str(d["gt"]),
                        d["q_type"],
                        mc_options_json=d["mc_options_json"],
                        gt_letter=d["gt_letter"],
                        answer_style=d["answer_style"],
                    )
                )
        n_valid = max(1, len(idxs))
        consensus_ratios.append(float(agree) / float(n_valid))
        group_valid_sizes.append(float(len(idxs)))

    # ---- Invalid-reason judge (unchanged from base) ----
    if invalid_jobs:
        def _run_judge(i: int) -> tuple[int, float]:
            score = base.judge_invalid_reason_explanation(
                **sample_details[i]["judge_payload"],
                ak=judge_api_key,
                model=judge_model,
                url=judge_url,
                timeout=judge_timeout,
            )
            return i, float(score)

        workers = min(judge_max_workers, len(invalid_jobs))
        if workers > 1 and len(invalid_jobs) > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_judge, i): i for i in invalid_jobs}
                for future in as_completed(futures):
                    fi = futures[future]
                    try:
                        i, score = future.result()
                        sample_details[i]["invalid_reason_score"] = score
                    except Exception as exc:
                        print(f"Invalid-reason judge failed (idx={fi}): {exc}")
        else:
            for i in invalid_jobs:
                try:
                    _, score = _run_judge(i)
                    sample_details[i]["invalid_reason_score"] = score
                except Exception as exc:
                    print(f"Invalid-reason judge failed (idx={i}): {exc}")

    # ---- Final scoring loop (identical composition to base) ----
    scores: list[base.RewardScore] = []
    fidelity_weight = base._maze_fidelity_weight()
    grounding_weight = base._vsi_grounding_weight()
    W = base.ROUND2_FORMAT_WEIGHT
    for detail in sample_details:
        fmt = float(detail["format_score"])
        r_acc = float(detail["r_acc"])
        content_reward = r_acc if detail["is_valid"] else float(detail["invalid_reason_score"])
        overall = (W * fmt) + ((1.0 - W) * content_reward)
        maze_fid = detail.get("maze_fidelity")
        vsi_g = detail.get("vsi_grounding")
        if isinstance(maze_fid, dict) and fidelity_weight > 0.0 and detail["is_valid"]:
            overall = (
                (W * fmt)
                + ((1.0 - W - fidelity_weight) * content_reward)
                + (fidelity_weight * float(maze_fid["fidelity"]))
            )
        elif isinstance(vsi_g, dict) and grounding_weight > 0.0 and detail["is_valid"]:
            overall = (
                (W * fmt)
                + ((1.0 - W - grounding_weight) * content_reward)
                + (grounding_weight * float(vsi_g["grounding"]))
            )
        score_item = {
            "overall": overall,
            "format": fmt,
            "accuracy": r_acc,  # NOTE: now agreement-with-majority, not vs GT.
            "answer_correctness": r_acc,
            "reflection": float(detail["reflection"]),
            "invalid_reason_score": float(detail["invalid_reason_score"]),
            "valid_case": 1.0 if detail["is_valid"] else 0.0,
            "invalid_case": 0.0 if detail["is_valid"] else 1.0,
        }
        if detail.get("true_acc") is not None:
            score_item["majority_vote/pseudo_vs_true_acc"] = float(detail["true_acc"])
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

    # ---- Batch-level diagnostics (attached to scores[0], same idiom as base) ----
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
        if scores and consensus_ratios:
            scores[0]["majority_vote/consensus_ratio"] = float(sum(consensus_ratios) / len(consensus_ratios))
            scores[0]["majority_vote/group_valid_size"] = float(sum(group_valid_sizes) / len(group_valid_sizes))
    except Exception:
        pass

    return scores
