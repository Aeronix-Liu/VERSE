#!/usr/bin/env python3
"""Single-turn direct eval (paper Table 1: Tool=No, Reasoning=No) for ALL 5 ViLaSR
benchmarks: vsi_bench, maze, SpatialEval_spatialreal, spar_bench, mmsi_bench.

No ViLaSR system prompt, no drawing / multi-turn agentic loop.
Prompts follow Appendix C.1 (direct zero-shot answer): the raw question + a
short C.1 suffix that asks for a direct letter / number. Per-shard output is
written to ``{output_dir}/{split}_{all}/results.jsonl`` (resumable).
"""
import argparse
import json
import os

from qwen_vl_utils import fetch_image, process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

# Paper Appendix C.1 / Figure 7 (direct, w/o reasoning)
DIRECT_SUFFIX = {
    "multiple choice": "\nAnswer with the option's letter from the given choices directly.",
    "numerical": "\nPlease answer with a single numerical value (e.g., 42 or 3.14).",
    "regression": "\nPlease answer with a single numerical value (e.g., 42 or 3.14).",
    "free-form": "",
    "vci": "",
}

BSZ = 8

# Datasets that present a single image (no per-image index caption needed).
SINGLE_IMAGE_DATASETS = {"maze", "SpatialEval_spatialreal"}


def _suffix_for(sample):
    ptype = (sample.get("problem_type") or "free-form").lower()
    return DIRECT_SUFFIX.get(ptype, "")


def _options_text(sample):
    """Return the options block to append after the question, or '' if options
    are already inline in the question text (maze / mmsi / SpatialEval)."""
    if sample.get("problem_type") == "multiple choice" and sample.get("options"):
        return "\n" + "\n".join(sample["options"])
    return ""


def _spar_post_prompt(sample):
    """spar_bench position_matching / camera_motion_infer normalization hint."""
    if sample.get("original_question_type") in ("position_matching", "camera_motion_infer"):
        return "\nThe values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
    return ""


def build_messages(sample, args):
    ds = args.dataset
    q = sample["question"]
    # Strip the redundant single-answer instructions that spar embeds (mirrors
    # the agentic infer.py cleanup) so the C.1 suffix is the only answer directive.
    q = q.replace("Your answer can only include one of options A, B, C or D.", "")
    q = q.replace("Answer using a single number and nothing else.", "")
    q = q + _options_text(sample)
    q = q + _spar_post_prompt(sample)
    suffix = _suffix_for(sample)

    frames = sample["image_path"][: args.max_frames]
    n = len(frames)
    content = []

    if ds in SINGLE_IMAGE_DATASETS:
        # Single image; no per-image index caption (matches paper direct protocol).
        content.append(
            {
                "type": "image",
                "image": os.path.join(args.image_folder, frames[0]),
                "max_pixels": args.max_pixels,
            }
        )
        content.append({"type": "text", "text": f"{q}{suffix}"})
    elif ds == "vsi_bench":
        text = (
            f"These are frames from a video, numbered from 1 to {n} in sequence. "
            f"That is, the index of each image is 1, 2, 3, ..., {n}.\n\n"
            f"{q}{suffix}"
        )
        for i, rel in enumerate(frames):
            content.append(
                {
                    "type": "image",
                    "image": os.path.join(args.image_folder, rel),
                    "max_pixels": args.max_pixels,
                }
            )
            content.append({"type": "text", "text": f"The index of the given image is {i + 1}.\n"})
        content.append({"type": "text", "text": text})
    elif ds in ("spar_bench", "spar_bench_tiny", "mmsi_bench"):
        if sample.get("data_type") == "single_view":
            content.append(
                {
                    "type": "image",
                    "image": os.path.join(args.image_folder, frames[0]),
                    "max_pixels": args.max_pixels,
                }
            )
            content.append({"type": "text", "text": f"{q}{suffix}"})
        else:  # multi_view
            for i, rel in enumerate(frames):
                content.append(
                    {
                        "type": "image",
                        "image": os.path.join(args.image_folder, rel),
                        "max_pixels": args.max_pixels,
                    }
                )
                content.append({"type": "text", "text": f"The index of the given image is {i + 1}.\n"})
            content.append({"type": "text", "text": f"{q}{suffix}"})
    else:
        raise ValueError(f"UNKNOWN args.dataset: {ds}")

    return [{"role": "user", "content": content}]


def eval_direct(args):
    print(f"Loading {args.model_path}")
    tp = max(1, len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        limit_mm_per_prompt={"image": 62, "video": 10},
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        mm_processor_cache_gb=0,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    with open(args.input_file, encoding="utf-8") as f:
        data = json.load(f) if args.input_file.endswith(".json") else [json.loads(l) for l in f]
    st = (len(data) * args.split) // args.all
    ed = (len(data) * (args.split + 1)) // args.all
    data = data[st:ed]
    print(f"Shard {args.split}/{args.all}: {len(data)} samples ({st}:{ed})")

    if args.all > 1:
        out_dir = os.path.join(args.output_dir, f"{args.split}_{args.all}")
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.jsonl")

    start_idx = 0
    if os.path.exists(out_path):
        with open(out_path) as fin:
            start_idx = sum(1 for _ in fin)
    mode = "a" if start_idx else "w"
    print(f"Output: {out_path}  resume_from={start_idx}")

    messages = [build_messages(x, args) for x in data]
    with open(out_path, mode, encoding="utf-8") as fout:
        for i in tqdm(range(start_idx, len(messages), BSZ), desc="direct batches"):
            batch_data = data[i : i + BSZ]
            batch_msgs = messages[i : i + BSZ]
            prompts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in batch_msgs
            ]
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                batch_msgs, return_video_kwargs=True
            )
            image_idx = 0
            llm_inputs = []
            for msg, prompt in zip(batch_msgs, prompts):
                imgs = []
                n_img = sum(
                    1
                    for turn in msg
                    for c in (turn["content"] if isinstance(turn["content"], list) else [])
                    if isinstance(c, dict) and c.get("type") == "image"
                )
                for _ in range(n_img):
                    imgs.append(image_inputs[image_idx])
                    image_idx += 1
                llm_inputs.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": imgs},
                    }
                )

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            for sample, out in zip(batch_data, outputs):
                text = out.outputs[0].text
                row = sample.copy()
                row["model_output"] = text
                row["model_id"] = args.model_path
                row["eval_protocol"] = "direct"
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--input-file", required=True)
    p.add_argument("--image-folder", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default="vsi_bench")
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--all", type=int, default=1)
    args = p.parse_args()
    eval_direct(args)
