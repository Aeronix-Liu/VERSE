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
"""Shared helpers for the ViLaSR-style agentic (think->draw->observe) Solver.

These constants/functions are used by BOTH the trainer (Round-2 prompt
construction in `_build_answer_batch`) and the rollout worker (multi-turn
generation). Keeping them in one place guarantees the image processing and the
prompt format stay token-aligned between the two sides.
"""

from typing import Any, Union

from PIL.Image import Image as ImageObject
from qwen_vl_utils import fetch_image

# ViLaSR RL system prompt (verbatim from ViLaSR verl/utils/dataset_multimodal.py).
# `{width}` / `{height}` are filled with the FIRST image's dimensions.
VILASR_SYSTEM_PROMPT = """### Guidance:
You are a spatial reasoning assistant with access to two powerful visualization tools.
Your task is to break down complex spatial problems and iteratively refine your solution through visualization feedback.

### Available tools:
You can use the following two tools to visualize. After each tool usage, you must wait for and analyze the visualization feedback before proceeding.

1. **Object Mapper**
- Purpose: Identifies and maps key items in the space
- Input format: JSON
```json
[{{
    "index": i, # Image index
    "bbox_2d": [x1, y1, x2, y2],
    "label": "object name/description"
}}]
```
- Output: Generates bounding boxes for visual inspection of the i-th image

2. **Path Tracer**
- Purpose: Plots movement or connections between points
- Input format: JSON
```json
[{{
    "index": i, # Image index
    "start_point_2d": [x1, y1],
    "end_point_2d": [x2, y2],
    "label": "trace_description"
}}]
```
- Output: Generates visual paths for verification of the i-th image

### Required Output Format:
For each reasoning step, you must structure your response as follows:
<think> [Your detailed reasoning process] </think> Action: [Object Mapper/Path Tracer]
```json
[JSON format coordinates]
```

After your reasoning and iteratively refine your solution through visualization feedback, you should arrive at a final answer and structure your response as follows:
<think> [Your detailed reasoning process] </think> Action: Answer
<answer> [Your final answer] </answer>

### Please NOTE the following reasoning techniques:
1. Initial Analysis
   - Break down the spatial problem
   - Plan your approach

2. Iterative Reasoning for Each Step
   - Choose appropriate tool
   - Provide absolute coordinates in JSON format (The top-left corner of the image is (0, 0) and the bottom-right corner is ({width}, {height}))
   - Observe the visualization output
   - Reflect on the visualization:
     * Is the placement/path accurate?
     * Does it align with your reasoning?
     * What adjustments are needed?
   - Backtrack and Adjust:
     * If errors found, backtrack to previous step to modify actions or decisions as needed"""

VILASR_PROMPT_TEMPLATE = """
### Question:
{question}

Begin your reasoning. After each tool use, critically evaluate the visualization and adjust if needed:
"""

# Answer-type templates (verbatim ViLaSR). Keyed by lowercase question_type.
VILASR_TYPE_TEMPLATE = {
    "multiple choice": "\nAnswer with the option's letter from the given choices directly.",
    "numerical": "\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).",
    "regression": "\nPlease answer the question using a single word or phrase (e.g., 42 or 3.14).",
    "free-form": "",
}

# Per-turn feedback prompts (verbatim ViLaSR rollout).
AGENTIC_INTERMEDIATE_PROMPT = (
    "The index of the given image is {current_image_idx} (width: {width}, height: {height}). "
    "Continue your reasoning. After each tool use, critically evaluate the visualization and adjust if needed:"
)
AGENTIC_FINAL_PROMPT = (
    "The index of the given image is {current_image_idx} (width: {width}, height: {height}). "
    "Then, you can not invoke the Object Mapper or Path Tracer tool. "
    "Please answer the initial question and structure your response as required:"
)
AGENTIC_INTERMEDIATE_TEMPLATE = """<|im_end|>
<|im_start|>user
{pad}
{prompt}
<|im_end|>
<|im_start|>assistant
"""

# Hard cap on images per trajectory (original frames + tool-edited feedback images).
AGENTIC_MAX_IMAGES = 42


def process_image_agentic(
    image: Union[str, ImageObject, dict[str, Any]],
    min_pixels: int,
    max_pixels: int,
) -> ImageObject:
    """Deterministically load+resize an image for the agentic pipeline.

    Uses qwen_vl_utils.fetch_image (smart_resize), which snaps dimensions to
    multiples of 28 so that the HF processor performs NO further internal
    resizing. This keeps (a) drawing coordinates aligned with what the model
    sees and (b) the image token count identical between the trainer's prompt
    tokenization and the rollout's retokenization.
    """
    if isinstance(image, dict):
        payload = dict(image)
    else:
        payload = {"image": image}
    payload.setdefault("min_pixels", min_pixels)
    payload.setdefault("max_pixels", max_pixels)
    return fetch_image(payload)
