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

"""ABLATION #1 unified reward -- self-evolve WITHOUT GT (majority voting).

Twin of the production `unified_reward.py`. Round-1 (Proposer question quality)
is dispatched to the UNCHANGED production reward; Round-2 (Solver) is dispatched
to the ablation majority-voting reward (`agentic_solver_majority_reward`) instead
of the GT-based `agentic_solver_reward`. No production file is modified.
"""

import os
import sys
from typing import Optional, TypedDict

# This file's own directory (holds agentic_solver_majority_reward).
_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
if _ABL_DIR not in sys.path:
    sys.path.insert(0, _ABL_DIR)

# The ORIGINAL reward_function package (holds question_quality_reward etc.).
_ORIG_REWARD_DIR = os.path.abspath(
    os.path.join(_ABL_DIR, "..", "..", "..", "easy_r1", "training", "reward_function")
)
_ORIG_REWARD_DIR = os.environ.get("SPATIALEVO_ORIG_REWARD_DIR", _ORIG_REWARD_DIR)
if _ORIG_REWARD_DIR not in sys.path:
    sys.path.insert(0, _ORIG_REWARD_DIR)

from question_quality_reward import compute_score as question_quality_compute_score  # noqa: E402
from agentic_solver_majority_reward import compute_score as agentic_solver_majority_compute_score  # noqa: E402

REWARD_NAME = "unified_reward_majority"
REWARD_TYPE = "batch"


class RewardInput(TypedDict, total=False):
    response: str
    response_length: int
    ground_truth: str
    multi_modal_data: Optional[dict]
    uid: Optional[str]
    round: Optional[int]


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


def compute_score(
    reward_inputs: list[RewardInput],
    question_quality_kwargs: Optional[dict] = None,
    majority_correctness_kwargs: Optional[dict] = None,
    agentic_solver: bool = True,  # kept for config compatibility; ignored (always majority in R2).
    **kwargs,
) -> list[RewardScore]:
    if not reward_inputs:
        raise ValueError("Input list is empty")

    round_values = [inp.get("round", 1) for inp in reward_inputs]
    unique_rounds = set(round_values)
    if len(unique_rounds) == 0:
        raise ValueError("Cannot determine the round value because the input list is empty")
    if len(unique_rounds) > 1:
        raise ValueError(
            f"Input contains mixed round values: {unique_rounds}. "
            f"All items in one batch must share the same round."
        )

    round_value = unique_rounds.pop()
    if round_value not in (1, 2):
        raise ValueError(f"round must be 1 or 2, but got {round_value}.")

    if round_value == 1:
        # Proposer reward is UNCHANGED from production.
        scores = question_quality_compute_score(reward_inputs, **(question_quality_kwargs or {}))
    else:
        # Round-2 Solver: majority-voting (no-GT) reward.
        scores = agentic_solver_majority_compute_score(
            reward_inputs, **(majority_correctness_kwargs or {})
        )

    if len(scores) != len(reward_inputs):
        raise RuntimeError(
            f"Returned score count ({len(scores)}) does not match input count ({len(reward_inputs)})"
        )
    for i, score in enumerate(scores):
        assert isinstance(score, dict), f"Score must be a dict, but got {type(score)}"
        assert "overall" in score, f"Score {i} is missing the overall field"
        assert isinstance(score["overall"], (int, float)), f"Score {i} overall must be numeric"
        assert "format" in score, f"Score {i} is missing the format field"
        assert "accuracy" in score, f"Score {i} is missing the accuracy field"

    return scores
