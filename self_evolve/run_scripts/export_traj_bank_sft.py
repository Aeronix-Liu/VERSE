#!/usr/bin/env python3
"""Export the verified trajectory bank to LLaMA-Factory sharegpt SFT data.

The bank (`<ckpt>/logs/trajectory_bank.jsonl`, written when
SPATIALEVO_TRAJ_BANK=1) stores every valid + format-correct + high-accuracy
solver rollout (maze additionally gated on engine-verified trajectory
fidelity). This script converts those rollouts into the same sharegpt format
as the ViLaSR cold-start/RRS data, so they can be distilled back into the
policy via rejection-sampling SFT (the cheap form of train-time-search:
search = sample n rollouts -> verify -> keep the best -> imitate).

Tool feedback images are NOT stored in the bank; they are deterministically
re-rendered here by replaying each assistant turn's tool JSON with the exact
rollout renderer (merge_bbox_movement + plot_bounding_boxes/plot_movement) on
images processed with the same min/max-pixels as training.

Usage:
  cd ViLaSR-self-evolve && ../envs/Xinran_VERSE/bin/python \
    run_scripts/export_traj_bank_sft.py \
    --bank ../checkpoints/<EXP>/logs/trajectory_bank.jsonl \
    --out-dir ../ViLaSR/ViLaSR-data/traj_bank_<EXP> [--register]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EASY_R1 = REPO_ROOT / "easy_r1"
XINRAN_ROOT = REPO_ROOT.parent
VILASR_DATA = XINRAN_ROOT / "ViLaSR" / "ViLaSR-data"
sys.path.insert(0, str(EASY_R1))
sys.path.insert(0, str(REPO_ROOT))

from verl.utils.agentic import process_image_agentic  # noqa: E402
from verl.utils.edit_image_video import (  # noqa: E402
    merge_bbox_movement,
    parse_bbox_and_movement,
    plot_bounding_boxes,
    plot_movement,
)

IMAGE_PAD = "<|vision_start|><|image_pad|><|vision_end|>"


def split_turns(sequence_text: str) -> list[dict[str, str]]:
    """Split the serialized rollout into role turns, keeping <image> markers.

    Mirrors agentic_solver_reward.parse_dialog but replaces every image pad
    with the sharegpt `<image>` marker instead of stripping it.
    """
    text = sequence_text.replace(IMAGE_PAD + "\n", "<image>").replace(IMAGE_PAD, "<image>")
    segments = [s for s in re.split(r"<\|im_start\|>|<\|im_end\|>", text) if s.strip()]
    turns = []
    for seg in segments:
        if seg.startswith("system"):
            # Keep the "system\n" prefix — the original RRS data does the same.
            turns.append({"role": "system", "content": seg})
        elif seg.startswith("user"):
            turns.append({"role": "user", "content": seg.replace("user\n", "", 1)})
        elif seg.startswith("assistant"):
            turns.append({"role": "assistant", "content": seg.replace("assistant\n", "", 1)})
    return turns


def replay_tool_images(turns: list[dict], input_images: list) -> list | None:
    """Re-render tool feedback images by replaying assistant tool calls.

    Returns the ordered list of tool-generated PIL images, or None when the
    replay cannot reproduce the expected image count (row is then skipped).
    """
    mm_images = list(input_images)
    bbox_dict: dict = {}
    movement_dict: dict = {}
    tool_images = []
    assistant_turns = [t for t in turns if t["role"] == "assistant"]
    for turn in assistant_turns[:-1]:  # final turn is Action: Answer
        bbox_new, movement_new = parse_bbox_and_movement(turn["content"])
        allindex: dict = defaultdict(lambda: {"bbox_list": [], "movement_list": []})
        for item in bbox_new:
            allindex[item["index"]]["bbox_list"].append(copy.deepcopy(item))
        for item in movement_new:
            allindex[item["index"]]["movement_list"].append(copy.deepcopy(item))
        if not allindex:
            return None
        current_image_index = len(mm_images)
        for cnt, tmp_index in enumerate(allindex):
            image_index_new = current_image_index + cnt
            image_index, bbox_dict, movement_dict = merge_bbox_movement(
                bbox_list_origin=bbox_dict,
                movement_list_origin=movement_dict,
                bbox_list_new=allindex[tmp_index]["bbox_list"],
                movement_list_new=allindex[tmp_index]["movement_list"],
                image_index_new=image_index_new,
            )
            if image_index < 0 or image_index >= len(mm_images):
                return None
            image = mm_images[image_index].copy()
            width, height = image.size
            plot_bounding_boxes(image, bbox_dict[image_index_new], input_width=width, input_height=height)
            plot_movement(image, movement_dict[image_index_new], input_width=width, input_height=height)
            mm_images.append(image)
            tool_images.append(image)
    return tool_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True, help="trajectory_bank.jsonl path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-pixels", type=int, default=100352)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--min-acc", type=float, default=0.8)
    parser.add_argument("--min-fidelity", type=float, default=None,
                        help="extra maze fidelity filter (default: keep bank's own gating)")
    parser.add_argument("--task-types", default="", help="comma-separated whitelist; empty = all")
    parser.add_argument("--dedup", action="store_true",
                        help="keep only the best rollout per (scene_id, question)")
    parser.add_argument("--dataset-name", default="traj_bank")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_whitelist = {t.strip() for t in args.task_types.split(",") if t.strip()}

    rows_in = [json.loads(l) for l in open(args.bank)]
    print(f"bank rows: {len(rows_in)}")

    # Optional dedup: keep the highest (fidelity, accuracy) rollout per question.
    if args.dedup:
        best: dict = {}
        for row in rows_in:
            key = (row.get("scene_id"), row.get("question"))
            score = (row.get("maze_traj_fidelity") or 0.0, row.get("accuracy") or 0.0)
            if key not in best or score > best[key][0]:
                best[key] = (score, row)
        rows_in = [v[1] for v in best.values()]
        print(f"after dedup: {len(rows_in)}")

    exported, skipped = [], 0
    for k, row in enumerate(rows_in):
        if task_whitelist and row.get("task_type") not in task_whitelist:
            continue
        if float(row.get("accuracy") or 0.0) < args.min_acc:
            continue
        fid = row.get("maze_traj_fidelity")
        if args.min_fidelity is not None and fid is not None and float(fid) < args.min_fidelity:
            continue
        seq = row.get("sequence_text") or ""
        image_paths = row.get("images") or []
        if not seq or not image_paths:
            skipped += 1
            continue
        turns = split_turns(seq)
        if not turns or turns[-1]["role"] != "assistant" or "<answer>" not in turns[-1]["content"]:
            skipped += 1
            continue
        try:
            input_images = [
                process_image_agentic(p, args.min_pixels, args.max_pixels) for p in image_paths
            ]
            tool_images = replay_tool_images(turns, input_images)
        except Exception as exc:
            print(f"  [skip] row {k}: replay failed: {exc}")
            skipped += 1
            continue
        if tool_images is None:
            skipped += 1
            continue
        all_images = input_images + tool_images
        n_markers = sum(t["content"].count("<image>") for t in turns)
        if n_markers != len(all_images):
            print(f"  [skip] row {k}: {n_markers} <image> markers vs {len(all_images)} images")
            skipped += 1
            continue

        row_id = f"step{row.get('step', 0)}_{k}"
        img_dir = out_dir / "images" / row_id
        img_dir.mkdir(parents=True, exist_ok=True)
        rel_paths = []
        for j, img in enumerate(all_images):
            p = img_dir / f"image_{j}.png"
            img.save(p)
            try:
                rel_paths.append(f"./{out_dir.relative_to(VILASR_DATA)}/images/{row_id}/image_{j}.png")
            except ValueError:
                rel_paths.append(str(p))

        answer = row.get("gt_letter") or row.get("gt") or ""
        exported.append(
            {
                "source": "traj_bank",
                "split": "train",
                "question_id": row_id,
                "images": rel_paths,
                "question": row.get("question", ""),
                "answer": answer,
                "question_type": row.get("answer_style") or row.get("task_type", ""),
                "task_type": row.get("task_type", ""),
                "accuracy": row.get("accuracy"),
                "maze_traj_fidelity": fid,
                "conversations": turns,
                "score": 1.0,
            }
        )

    out_json = out_dir / f"{args.dataset_name}.json"
    out_json.write_text(json.dumps(exported, ensure_ascii=False, indent=1))
    print(f"exported {len(exported)} rows ({skipped} skipped) -> {out_json}")

    if args.register:
        info_path = VILASR_DATA / "dataset_info.json"
        info = json.loads(info_path.read_text())
        try:
            rel_file = f"{out_dir.relative_to(VILASR_DATA)}/{args.dataset_name}.json"
        except ValueError:
            rel_file = str(out_json)
        info[args.dataset_name] = {
            "file_name": rel_file,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
        print(f"Registered `{args.dataset_name}` in {info_path}")


if __name__ == "__main__":
    main()
