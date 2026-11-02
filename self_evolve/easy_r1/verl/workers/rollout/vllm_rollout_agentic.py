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
"""Agentic (ViLaSR think->draw->observe) multi-turn rollout.

Ported from ViLaSR's `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_video.py`
and adapted to the VERSE easy_r1 stack:

- Subclasses the stock single-turn `vLLMRollout` (identical engine setup: dummy
  load format, sleep mode, external launcher) and dispatches on
  `prompts.meta_info["rollout_mode"] == "agentic"`.
- `multi_modal_data` arrives as `{"images": [paths]}` (VERSE convention);
  images are materialized to 28-divisible PIL via `process_image_agentic` — the
  SAME function the trainer used to tokenize the Round-2 prompt.
- Position ids follow the VERSE 4-row scheme (text row + 3 vision mRoPE rows).
- Returns extra fields consumed downstream:
  * batch  "multi_turn_mask"       — loss mask over assistant tokens only
  * non_tensor "multi_modal_inputs" — processor features aligned with the final sequence
  * non_tensor "agentic_response_text" / "agentic_sequence_text" — for the reward fn
"""

import copy
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.agentic import (
    AGENTIC_FINAL_PROMPT,
    AGENTIC_INTERMEDIATE_PROMPT,
    AGENTIC_INTERMEDIATE_TEMPLATE,
    AGENTIC_MAX_IMAGES,
    process_image_agentic,
)
from ...utils.edit_image_video import (
    merge_bbox_movement,
    parse_bbox_and_movement,
    plot_bounding_boxes,
    plot_movement,
)
from .vllm_rollout_spmd import _repeat_interleave, vLLMRollout


@dataclass
class ProcessData:
    index: int
    response: str
    response_ids: List
    mm_data: Dict
    bbox_list_origin: Dict
    movement_list_origin: Dict
    finish_reason: str
    is_finished: bool
    grid_size: Any


def process_single_response(data: ProcessData):
    """Parse one turn's tool call(s), draw them, and return the edited image(s)."""
    if data.is_finished is True:
        return {
            "index": data.index,
            "response": data.response,
            "response_ids": data.response_ids,
            "finish_reason": data.finish_reason,
            "is_finished": data.is_finished,
            "processed_image_idx": [None],
        }
    try:
        bbox_list_new, movement_list_new = parse_bbox_and_movement(data.response)
        current_image_index = len(data.mm_data["image"])
        image_index_list, image_list = [], []
        bbox_list, movement_list = data.bbox_list_origin, data.movement_list_origin
        finish_reason = None
        allindex = {}
        try:
            for tmp_bbox_list in bbox_list_new:
                tmp_bbox_list = copy.deepcopy(tmp_bbox_list)
                if tmp_bbox_list["index"] in allindex:
                    if "bbox_list" in allindex[tmp_bbox_list["index"]]:
                        allindex[tmp_bbox_list["index"]]["bbox_list"].append(tmp_bbox_list)
                    else:
                        allindex[tmp_bbox_list["index"]]["bbox_list"] = [tmp_bbox_list]
                else:
                    allindex[tmp_bbox_list["index"]] = {"bbox_list": [tmp_bbox_list], "movement_list": []}
            for tmp_movement_list in movement_list_new:
                tmp_movement_list = copy.deepcopy(tmp_movement_list)
                if tmp_movement_list["index"] in allindex:
                    if "movement_list" in allindex[tmp_movement_list["index"]]:
                        allindex[tmp_movement_list["index"]]["movement_list"].append(tmp_movement_list)
                    else:
                        allindex[tmp_movement_list["index"]]["movement_list"] = [tmp_movement_list]
                else:
                    allindex[tmp_movement_list["index"]] = {"bbox_list": [], "movement_list": [tmp_movement_list]}
        except Exception:
            traceback.print_exc()
            print("bbox_list_new, movement_list_new: ", bbox_list_new, movement_list_new)
            finish_reason = "ToolGenError"

        if len(allindex) == 0:
            finish_reason = "ToolError"
        elif len(data.mm_data["image"]) >= AGENTIC_MAX_IMAGES + 1:
            finish_reason = "TooManyImages"

        if finish_reason is not None:
            return {
                "index": data.index,
                "processed_image_idx": [None],
                "image": [data.mm_data["image"][0].copy()],
                "response": data.response,
                "response_ids": data.response_ids,
                "finish_reason": finish_reason,
                "bbox_list": bbox_list,
                "movement_list": movement_list,
                "is_finished": True,
            }
        for cnt, tmp_index in enumerate(allindex):
            bbox_list_new, movement_list_new = allindex[tmp_index]["bbox_list"], allindex[tmp_index]["movement_list"]
            image_index_new = current_image_index + cnt
            image_index, bbox_list, movement_list = merge_bbox_movement(
                bbox_list_origin=data.bbox_list_origin,
                movement_list_origin=data.movement_list_origin,
                bbox_list_new=bbox_list_new,
                movement_list_new=movement_list_new,
                image_index_new=image_index_new,
            )
            image_index_list.append(image_index)
            if image_index == -1:
                return {
                    "index": data.index,
                    "processed_image_idx": [None],
                    "image": [data.mm_data["image"][0].copy()],
                    "response": data.response,
                    "response_ids": data.response_ids,
                    "finish_reason": "ToolError",
                    "bbox_list": bbox_list,
                    "movement_list": movement_list,
                    "is_finished": True,
                }
            try:
                image = data.mm_data["image"][image_index].copy()
                assert isinstance(image, Image.Image)
                input_width, input_height = image.size
                plot_bounding_boxes(image, bbox_list[image_index_new], input_height=input_height, input_width=input_width)
                plot_movement(image, movement_list[image_index_new], input_height=input_height, input_width=input_width)
                image_list.append(image)
            except IndexError as e:
                print(f"Image index error: {str(e)}")
                return {
                    "index": data.index,
                    "processed_image_idx": [None],
                    "image": [data.mm_data["image"][0].copy()],
                    "response": data.response,
                    "response_ids": data.response_ids,
                    "finish_reason": "IndexError",
                    "bbox_list": bbox_list,
                    "movement_list": movement_list,
                    "is_finished": True,
                }

        return {
            "index": data.index,
            "processed_image_idx": image_index_list,
            "image": image_list,
            "response": data.response,
            "response_ids": data.response_ids,
            "finish_reason": data.finish_reason,
            "bbox_list": bbox_list,
            "movement_list": movement_list,
            "is_finished": data.is_finished,
        }
    except Exception as e:
        print(f"Error processing response {data.index}: {str(e)}")
        traceback.print_exc()
        return None


class AgenticVLLMRollout(vLLMRollout):
    """Single-turn vLLMRollout + ViLaSR agentic multi-turn mode."""

    # ------------------------------------------------------------------ helpers

    def _get_multi_turn_mask(self, response_tokens: torch.Tensor) -> torch.Tensor:
        """Mask that keeps ONLY assistant-generated tokens of a multi-turn response."""
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        pad_id = self.tokenizer.pad_token_id
        newline_id = 198

        attention_mask = torch.zeros_like(response_tokens)
        current_pos = 0
        in_assistant_response = True  # the response starts inside an assistant turn
        n = len(response_tokens)
        while current_pos < n:
            if response_tokens[current_pos] == im_end_id:
                in_assistant_response = False
                current_pos += 1
                continue
            if (
                current_pos + 2 < n
                and response_tokens[current_pos] == im_start_id
                and response_tokens[current_pos + 1] == assistant_id
                and response_tokens[current_pos + 2] == newline_id
            ):
                in_assistant_response = True
                current_pos += 3
                continue
            if in_assistant_response and response_tokens[current_pos] != pad_id:
                attention_mask[current_pos] = 1
            current_pos += 1
        return attention_mask

    def _get_prompts_and_indices(self, samples_info, num_llm_calls_available_list):
        prompts, multi_modal_data, indices = [], [], []
        for index, (info, num_calls) in enumerate(zip(samples_info, num_llm_calls_available_list)):
            if not info["stop"] and len(info["multi_modal_data"]["image"]) <= AGENTIC_MAX_IMAGES and num_calls >= 0:
                prompts.append(info["sequence"])
                multi_modal_data.append(info["multi_modal_data"])
                indices.append(info["index"])
        return prompts, multi_modal_data, indices

    # ------------------------------------------------------- multi-turn generate

    def _multi_turn_generate(self, vllm_inputs, sampling_params, use_tqdm=False):
        sampling_params = copy.deepcopy(sampling_params)
        new_vllm_inputs = []
        for single_vllm_input in vllm_inputs:
            prompt = self.tokenizer.decode(single_vllm_input["prompt_token_ids"], skip_special_tokens=False)
            new_vllm_inputs.extend(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": copy.deepcopy(single_vllm_input["multi_modal_data"]),
                        "grid_size": single_vllm_input["grid_size"],
                    }
                    for _ in range(sampling_params.n)
                ]
            )
        sampling_params.n = 1
        sampling_params.detokenize = True
        sampling_params.max_tokens = self.config.single_turn_response_length

        samples_info = []
        for index, item in enumerate(new_vllm_inputs):
            origin_image = item["multi_modal_data"]["image"][0]
            width, height = origin_image.size
            assert width % 28 == 0 and height % 28 == 0, (
                f"Image dimensions ({width}, {height}) must be divisible by 28."
            )
            sample_info = {
                "prompt": item["prompt"],
                "sequence": item["prompt"],
                "multi_modal_data": item["multi_modal_data"],
                "response": "",
                "stop": False,
                "finish_reason": None,
                "processed_image_idx": [],
                "index": index,
                "execution_pass": 0,
                "bbox_list": {img_idx: [] for img_idx in range(len(item["multi_modal_data"]["image"]))},
                "movement_list": {img_idx: [] for img_idx in range(len(item["multi_modal_data"]["image"]))},
                "grid_size": item["grid_size"],
                "total_response_tokens": 0,
            }
            samples_info.append(sample_info)

        num_llm_calls_available = copy.deepcopy(self.config.num_llm_calls_available) - 1
        num_llm_calls_available_list = [
            num_llm_calls_available + 10 if sample["grid_size"] is not None else num_llm_calls_available
            for sample in samples_info
        ]

        while any(num_calls >= 0 for num_calls in num_llm_calls_available_list):
            input_prompts, multi_modal_data, indices = self._get_prompts_and_indices(
                samples_info, num_llm_calls_available_list
            )
            for i in range(len(num_llm_calls_available_list)):
                if num_llm_calls_available_list[i] >= 0:
                    num_llm_calls_available_list[i] -= 1
            input_prompts = [
                {
                    "prompt_token_ids": self.tokenizer.encode(prompt, add_special_tokens=False)[
                        : self.config.prompt_length + self.config.response_length
                    ],
                    "multi_modal_data": mm_data,
                }
                for prompt, mm_data in zip(input_prompts, multi_modal_data)
            ]
            if self.rank == 0:
                print(
                    f"[agentic] calls_left={num_llm_calls_available_list} "
                    f"prompt_toks={[len(x['prompt_token_ids']) for x in input_prompts]} "
                    f"images={[len(x['multi_modal_data']['image']) for x in input_prompts]}"
                )
            outputs = self.inference_engine.generate(
                prompts=input_prompts, sampling_params=sampling_params, use_tqdm=use_tqdm
            )
            sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
            responses = [x.outputs[0].text for x in sorted_outputs]
            output_token_ids = [x.outputs[0].token_ids for x in sorted_outputs]
            finish_reason = [x.outputs[0].finish_reason for x in sorted_outputs]
            stop_reason = [x.outputs[0].stop_reason for x in sorted_outputs]

            for i, index in enumerate(indices):
                new_tokens = len(output_token_ids[i])
                if samples_info[index]["total_response_tokens"] + new_tokens >= self.config.response_length:
                    finish_reason[i] = "max_tokens"
                    stop_reason[i] = None

            if all(num_calls == -1 for num_calls in num_llm_calls_available_list):  # last turn
                for i, index in enumerate(indices):
                    samples_info[index]["response"] += responses[i]
                    samples_info[index]["sequence"] += responses[i]
                    samples_info[index]["stop"] = True
                    samples_info[index]["finish_reason"] = finish_reason[i]
                break

            def _is_finished(finish_reason, stop_reason, response):
                if finish_reason == "stop" and stop_reason is None and "<answer>" in response and "</answer>" in response:
                    return True
                if finish_reason == "length":
                    return True
                if finish_reason == "rule":
                    return True
                if finish_reason == "max_tokens":
                    return True
                return False

            is_finished = [_is_finished(finish_reason[i], stop_reason[i], responses[i]) for i in range(len(finish_reason))]
            if all(is_finished):
                for i, index in enumerate(indices):
                    samples_info[index]["response"] += responses[i]
                    samples_info[index]["sequence"] += responses[i]
                    samples_info[index]["stop"] = True
                    samples_info[index]["finish_reason"] = finish_reason[i]
                break

            process_data_list = [
                ProcessData(
                    index=index,
                    response=responses[i],
                    response_ids=output_token_ids[i],
                    mm_data=samples_info[index]["multi_modal_data"],
                    bbox_list_origin=samples_info[index]["bbox_list"],
                    movement_list_origin=samples_info[index]["movement_list"],
                    finish_reason=finish_reason[i],
                    is_finished=is_finished[i],
                    grid_size=samples_info[index]["grid_size"],
                )
                for i, index in enumerate(indices)
            ]
            with ThreadPoolExecutor(max_workers=1) as executor:
                results = list(executor.map(process_single_response, process_data_list))

            for result in results:
                if result is None:
                    continue
                index = result["index"]
                samples_info[index]["response"] += result["response"]
                samples_info[index]["stop"] = result["is_finished"]
                samples_info[index]["finish_reason"] = result["finish_reason"]
                samples_info[index]["processed_image_idx"].extend(result["processed_image_idx"])

                if result["is_finished"] is False:
                    current_image_count = len(samples_info[index]["multi_modal_data"]["image"])
                    current_image_idx = current_image_count + 1
                    current_tokens = samples_info[index]["total_response_tokens"]
                    response_tokens = len(result["response_ids"])
                    if len(result["image"]) > 1:
                        total_required_tokens = response_tokens
                        pad_prompt = ""
                        images_to_add = []
                        for tmp_image_idx, tmp_image in enumerate(result["image"]):
                            width, height = tmp_image.size
                            if current_image_count + tmp_image_idx + 1 >= AGENTIC_MAX_IMAGES:
                                current_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + AGENTIC_FINAL_PROMPT.format(
                                    current_image_idx=current_image_idx + tmp_image_idx, width=width, height=height
                                )
                                pad_prompt += current_prompt
                                images_to_add.append(tmp_image)
                                break
                            if tmp_image_idx <= len(result["image"]) - 2:
                                current_prompt = (
                                    f"<|vision_start|><|image_pad|><|vision_end|>The index of the given image is "
                                    f"{current_image_idx + tmp_image_idx} (width: {width}, height: {height}).\n"
                                )
                            else:
                                prompt_template = (
                                    AGENTIC_INTERMEDIATE_PROMPT
                                    if num_llm_calls_available_list[index] > 0
                                    else AGENTIC_FINAL_PROMPT
                                )
                                current_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + prompt_template.format(
                                    current_image_idx=current_image_idx + tmp_image_idx, width=width, height=height
                                )
                            pad_prompt += current_prompt
                            images_to_add.append(tmp_image)

                        template_str = AGENTIC_INTERMEDIATE_TEMPLATE.format(prompt="", pad=pad_prompt)
                        template_tokens = len(self.tokenizer.encode(template_str, add_special_tokens=False))
                        image_tokens = 256 * len(images_to_add)
                        total_required_tokens += template_tokens + image_tokens

                        if current_tokens + total_required_tokens >= self.config.response_length:
                            samples_info[index]["sequence"] += result["response"]
                            samples_info[index]["total_response_tokens"] += response_tokens
                            samples_info[index]["stop"] = True
                            samples_info[index]["finish_reason"] = "max_tokens"
                        else:
                            samples_info[index]["sequence"] += result["response"] + template_str
                            samples_info[index]["multi_modal_data"]["image"].extend(images_to_add)
                            samples_info[index]["total_response_tokens"] = current_tokens + total_required_tokens
                    else:
                        width, height = result["image"][0].size
                        if current_image_count + 1 >= AGENTIC_MAX_IMAGES:
                            prompt = AGENTIC_FINAL_PROMPT.format(
                                current_image_idx=current_image_idx, width=width, height=height
                            )
                        else:
                            prompt = (
                                AGENTIC_INTERMEDIATE_PROMPT
                                if num_llm_calls_available_list[index] > 0
                                else AGENTIC_FINAL_PROMPT
                            ).format(current_image_idx=current_image_idx, width=width, height=height)
                        new_template = AGENTIC_INTERMEDIATE_TEMPLATE.format(
                            prompt=prompt, pad="<|vision_start|><|image_pad|><|vision_end|>"
                        )
                        template_tokens = len(self.tokenizer.encode(new_template, add_special_tokens=False))
                        image_tokens = 256
                        potential_total_tokens = current_tokens + response_tokens + template_tokens + image_tokens
                        if potential_total_tokens >= self.config.response_length:
                            samples_info[index]["sequence"] += result["response"]
                            samples_info[index]["total_response_tokens"] += response_tokens
                            samples_info[index]["stop"] = True
                            samples_info[index]["finish_reason"] = "max_tokens"
                        else:
                            samples_info[index]["sequence"] += result["response"] + new_template
                            samples_info[index]["multi_modal_data"]["image"].append(result["image"][0])
                            samples_info[index]["total_response_tokens"] = potential_total_tokens
                    samples_info[index]["bbox_list"] = result["bbox_list"]
                    samples_info[index]["movement_list"] = result["movement_list"]
                else:
                    samples_info[index]["sequence"] += result["response"]

        for i in range(len(samples_info)):
            if samples_info[i]["finish_reason"] != "length":
                samples_info[i]["sequence"] += self.tokenizer.eos_token
                samples_info[i]["response"] += self.tokenizer.eos_token

        responses, sequences, image_inputs = [], [], []
        for sample_info in samples_info:
            responses.append(sample_info["response"])
            sequences.append(sample_info["sequence"])
            image_inputs.append(sample_info["multi_modal_data"]["image"])
        return responses, sequences, image_inputs

    # -------------------------------------------------------- dispatch entrypoint

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if prompts.meta_info.get("rollout_mode") != "agentic":
            return super().generate_sequences(prompts)
        return self._generate_sequences_agentic(prompts)

    @torch.no_grad()
    def _generate_sequences_agentic(self, prompts: DataProto) -> DataProto:
        from ...models.transformers.qwen2_vl import get_rope_index

        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length), left-padded
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]  # (bs, 4, prompt_length)
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data")
        if "grid_size" in non_tensor_batch:
            batch_grid_size = non_tensor_batch.pop("grid_size")
        else:
            batch_grid_size = np.array([None] * batch_size, dtype=object)

        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        min_pixels = prompts.meta_info["min_pixels"]
        max_pixels = prompts.meta_info["max_pixels"]

        vllm_inputs = []
        for raw_prompt_ids, multi_modal_data, grid_size in zip(
            batch_raw_prompt_ids, batch_multi_modal_data, batch_grid_size
        ):
            images = []
            if isinstance(multi_modal_data, dict) and "images" in multi_modal_data:
                for image in multi_modal_data["images"]:
                    images.append(process_image_agentic(image, min_pixels, max_pixels))
            vllm_inputs.append(
                {
                    "prompt_token_ids": list(raw_prompt_ids),
                    "multi_modal_data": {"image": images},
                    "grid_size": None if grid_size is None or (isinstance(grid_size, float) and np.isnan(grid_size)) else grid_size,
                }
            )

        with self.update_sampling_params(**prompts.meta_info):
            responses, sequences, image_inputs = self._multi_turn_generate(
                vllm_inputs=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False
            )
            n = self.sampling_params.n
            if n > 1:
                batch_size = batch_size * n
                input_ids = _repeat_interleave(input_ids, n)
                attention_mask = _repeat_interleave(attention_mask, n)
                position_ids = _repeat_interleave(position_ids, n)
                batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, n)
                batch_grid_size = _repeat_interleave(batch_grid_size, n)

        # Full multi-turn token ids (prompt + all turns), used downstream in logs.
        non_tensor_batch["raw_prompt_ids"] = np.array(
            [
                self.tokenizer.encode(sequence, add_special_tokens=False)[
                    : self.config.prompt_length + self.config.response_length
                ]
                for sequence in sequences
            ],
            dtype=object,
        )

        valid_prompt_len = torch.sum(attention_mask, dim=-1)
        response_ids = []
        response_mask = []
        response_position_ids = []
        model_inputs = []
        multi_turn_mask = []
        for idx, prompt_len in enumerate(valid_prompt_len):
            prompt_len = int(prompt_len)
            inputs = self.processor(
                text=sequences[idx],
                images=image_inputs[idx],
                add_special_tokens=False,
                return_tensors="pt",
            )
            full_input_ids = inputs["input_ids"][0]
            full_attention_mask = inputs["attention_mask"][0]
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=full_input_ids,
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                second_per_grid_ts=inputs.get("second_per_grid_ts", None),
                attention_mask=full_attention_mask,
            )  # (3, full_len)
            text_position_ids = torch.arange(len(full_input_ids)).unsqueeze(0)  # (1, full_len)
            full_position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, full_len)

            assert torch.sum(input_ids[idx][-prompt_len:].cpu() == full_input_ids[:prompt_len].cpu()) == prompt_len, (
                f"Input IDs mismatch at batch index {idx}"
            )
            n_vision_start = int(torch.sum(full_input_ids[: prompt_len + self.config.response_length] == 151652))
            assert n_vision_start == len(image_inputs[idx]), (
                f"Number of <|vision_start|> {n_vision_start} != Image Inputs {len(image_inputs[idx])}"
            )

            response_ids.append(full_input_ids[prompt_len : prompt_len + self.config.response_length])
            response_mask.append(full_attention_mask[prompt_len : prompt_len + self.config.response_length])
            pad_position_ids = VF.pad_sequence_to_length(
                full_position_ids[:, prompt_len : prompt_len + self.config.response_length],
                max_seq_len=self.config.response_length,
                pad_token_id=0,
                left_pad=False,
            ).to(input_ids.device)  # (4, response_length)
            response_position_ids.append(pad_position_ids)
            tmp_multi_turn_mask = self._get_multi_turn_mask(
                full_input_ids[prompt_len : prompt_len + self.config.response_length]
            )
            multi_turn_mask.append(tmp_multi_turn_mask)

            inputs.pop("input_ids")
            inputs.pop("attention_mask")
            model_inputs.append(dict(inputs))

        response_ids = VF.pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)
        non_tensor_batch["multi_modal_inputs"] = np.array(model_inputs, dtype=object)
        # Keep the ORIGINAL image paths (repeated n times) for logging/metrics;
        # the actor consumes `multi_modal_inputs`, never re-processes these.
        non_tensor_batch["multi_modal_data"] = batch_multi_modal_data
        non_tensor_batch["grid_size"] = batch_grid_size
        non_tensor_batch["agentic_response_text"] = np.array(responses, dtype=object)
        non_tensor_batch["agentic_sequence_text"] = np.array(sequences, dtype=object)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_position_ids = torch.stack(response_position_ids, dim=0).to(input_ids.device)  # (bs, 4, resp_len)
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.pad_2d_list_to_length(response_mask, 0, max_length=self.config.response_length).to(
            input_ids.device
        )
        multi_turn_mask = VF.pad_2d_list_to_length(multi_turn_mask, 0, max_length=self.config.response_length).to(
            input_ids.device
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        if self.rank == 0:
            valid_lengths = torch.sum(attention_mask, dim=1)
            print(
                f"[agentic] seq={tuple(sequence_ids.size())} "
                f"valid_len max/min/avg: {int(valid_lengths.max())}/{int(valid_lengths.min())}/"
                f"{float(valid_lengths.float().mean()):.1f}"
            )

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
                "multi_turn_mask": multi_turn_mask,
            },
            batch_size=batch_size,
        )
        for key, value in non_tensor_batch.items():
            if not isinstance(value, np.ndarray):
                non_tensor_batch[key] = np.array(value, dtype=object)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
