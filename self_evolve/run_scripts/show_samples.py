#!/usr/bin/env python3
"""Display sampled training interactions: maze image, proposer question, solver rollout.

Logs are read from two files under <ckpt>/logs/:
  - io_log.jsonl         : Round-1 Questioner I/O (one entry per sample)
  - round2_valid_qa.jsonl: Round-2 Solver I/O grouped by question (multiple rollouts per entry)
  - round2_invalid_qa.jsonl: Round-2 invalid questions + solver explanations

Sampling frequency is controlled by SPATIALEVO_SAMPLE_LOG_FREQ (default 20).

Usage:
    python show_samples.py                         # show all sampled steps (round2)
    python show_samples.py --step 20               # show only step 20
    python show_samples.py --step 1 --n 3          # show first 3 questions of step 1
    python show_samples.py --round1                # show round1 (questioner) instead
    python show_samples.py --invalid               # show invalid round2 questions
    python show_samples.py --list-steps            # list available steps
"""
import argparse
import json
import os
import re
import textwrap

DEFAULT_CKPT = "/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr_maze_rr"


def load_jsonl(path: str):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_final_answer(response: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # gcot format: Answer: X
    m = re.search(r"answer\s*[:：]\s*([ABCD])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "?"


def format_text(text: str, indent: str = "      ", width: int = 200) -> str:
    lines = text.split("\n")
    out = []
    for line in lines:
        if len(line) > width:
            wrapped = textwrap.wrap(line, width=width, initial_indent="", subsequent_indent=indent + "  ")
            for w in wrapped:
                out.append(indent + w)
        else:
            out.append(indent + line if line else "")
    return "\n".join(out)


def display_round2_sample(entry: dict, idx: int, max_rollouts: int | None = None):
    step = entry.get("step", "?")
    question = entry.get("question", "")
    gt = entry.get("gt", "?")
    images = entry.get("images", [])
    responses = entry.get("responses", [])
    accuracies = entry.get("accuracies", [])
    dup_weight = entry.get("duplicate_weight", 1)

    # Extract action sequence
    action_seq = ""
    if "Full Action Sequence:" in question:
        action_seq = question.split("Full Action Sequence:")[-1].strip()

    image_path = images[0] if images else ""

    print(f"\n{'='*100}")
    print(f"  Question {idx} | Step {step} | GT: {gt} | Rollouts: {len(responses)} | DupWeight: {dup_weight}")
    print(f"{'='*100}")

    print(f"\n  [1] Maze Image:")
    print(f"      {image_path}")

    print(f"\n  [2] Proposer's Question (round-robin assigned):")
    if action_seq:
        print(f"      Action Sequence: {action_seq[:200]}")
    for line in question.split("\n"):
        print(f"      {line}")

    print(f"\n  [3] Solver's Agentic Rollouts ({len(responses)} rollouts):")
    n = len(responses) if max_rollouts is None else min(max_rollouts, len(responses))
    for r_idx in range(n):
        resp = responses[r_idx]
        acc = accuracies[r_idx] if r_idx < len(accuracies) else "?"
        pred = extract_final_answer(resp)
        correct = "✓" if acc == 1.0 else "✗" if acc == 0.0 else "?"
        print(f"\n      --- Rollout {r_idx}: acc={acc} pred={pred} {correct} | {len(resp)} chars ---")
        formatted = format_text(resp, indent="        ")
        print(formatted)


def display_round1_sample(entry: dict, idx: int):
    step = entry.get("step", "?")
    reward = entry.get("reward", "?")
    is_valid = entry.get("is_valid", "?")
    gt = entry.get("gt", "?")
    question = entry.get("question", "")
    response = entry.get("response", "")
    image = entry.get("image", "")

    image_path = ""
    if isinstance(image, list) and image:
        image_path = image[0] if isinstance(image[0], str) else str(image[0])
    elif isinstance(image, str):
        image_path = image

    pred = extract_final_answer(response)

    print(f"\n{'='*100}")
    print(f"  Sample {idx} | Step {step} | Reward: {reward} | Valid: {is_valid} | GT: {gt} | Pred: {pred}")
    print(f"{'='*100}")

    print(f"\n  [1] Maze Image:")
    print(f"      {image_path}")

    print(f"\n  [2] Constructed Question:")
    for line in question.split("\n"):
        print(f"      {line}")

    print(f"\n  [3] Questioner's Raw Response:")
    formatted = format_text(response, indent="      ")
    print(formatted)


def main():
    parser = argparse.ArgumentParser(description="Display sampled training interactions")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Checkpoint directory")
    parser.add_argument("--step", type=int, default=None, help="Filter by step number")
    parser.add_argument("--n", type=int, default=None, help="Show only first N entries")
    parser.add_argument("--round1", action="store_true", help="Show Round-1 (Questioner) instead of Round-2")
    parser.add_argument("--invalid", action="store_true", help="Show invalid Round-2 questions")
    parser.add_argument("--rollouts", type=int, default=None, help="Max rollouts to display per question")
    parser.add_argument("--list-steps", action="store_true", help="List available steps and exit")
    args = parser.parse_args()

    log_dir = os.path.join(args.ckpt, "logs")

    if args.list_steps:
        r1 = load_jsonl(os.path.join(log_dir, "io_log.jsonl"))
        r2 = load_jsonl(os.path.join(log_dir, "round2_valid_qa.jsonl"))
        r2i = load_jsonl(os.path.join(log_dir, "round2_invalid_qa.jsonl"))
        print(f"Round-1 (io_log.jsonl): {len(r1)} samples, steps: {sorted(set(s.get('step') for s in r1))}")
        print(f"Round-2 valid (round2_valid_qa.jsonl): {len(r2)} entries, steps: {sorted(set(s.get('step') for s in r2))}")
        print(f"Round-2 invalid (round2_invalid_qa.jsonl): {len(r2i)} entries, steps: {sorted(set(s.get('step') for s in r2i))}")
        return

    if args.round1:
        # Show Round-1 Questioner I/O
        log_path = os.path.join(log_dir, "io_log.jsonl")
        samples = load_jsonl(log_path)
        if args.step is not None:
            samples = [s for s in samples if s.get("step") == args.step]
        if not samples:
            print(f"No Round-1 samples found at {log_path}")
            return
        steps = sorted(set(s.get("step") for s in samples))
        print(f"\nFound {len(samples)} Round-1 samples across steps: {steps}")
        shown = 0
        for step in steps:
            step_samples = [s for s in samples if s.get("step") == step]
            if args.n is not None:
                step_samples = step_samples[:args.n]
            print(f"\n{'#'*100}")
            print(f"## Step {step} (Round-1 Questioner, {len(step_samples)} samples)")
            print(f"{'#'*100}")
            for i, s in enumerate(step_samples):
                display_round1_sample(s, i)
                shown += 1
        print(f"\nTotal displayed: {shown} samples")
        return

    # Show Round-2 Solver I/O (default)
    if args.invalid:
        log_path = os.path.join(log_dir, "round2_invalid_qa.jsonl")
        label = "Round-2 Invalid"
    else:
        log_path = os.path.join(log_dir, "round2_valid_qa.jsonl")
        label = "Round-2 Valid"

    samples = load_jsonl(log_path)
    if args.step is not None:
        samples = [s for s in samples if s.get("step") == args.step]
    if not samples:
        print(f"No {label} samples found at {log_path}")
        return
    steps = sorted(set(s.get("step") for s in samples))
    print(f"\nFound {len(samples)} {label} entries across steps: {steps}")
    shown = 0
    for step in steps:
        step_samples = [s for s in samples if s.get("step") == step]
        if args.n is not None:
            step_samples = step_samples[:args.n]
        print(f"\n{'#'*100}")
        print(f"## Step {step} ({label}, {len(step_samples)} questions)")
        print(f"{'#'*100}")
        for i, s in enumerate(step_samples):
            display_round2_sample(s, i, max_rollouts=args.rollouts)
            shown += 1
    print(f"\n{'='*100}")
    print(f"Total displayed: {shown} questions")


if __name__ == "__main__":
    main()
