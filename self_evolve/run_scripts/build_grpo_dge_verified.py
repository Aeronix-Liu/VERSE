#!/usr/bin/env python3
"""Build the DGE-verified GRPO subset for Phase-1 baseline Round-2 training.

Reads the full DGE x GRPO compatibility results (run_scripts/dge_grpo_compat.json,
produced by test_dge_grpo_compat.py --all) and the original GRPO RL data, and emits
the rows that DGE validated AND whose computed ground truth agrees with the GRPO
answer. These rows can be fed directly to Round-2 (skipping the Round-1 proposer)
as a static-GRPO-on-DGE-verified-subset baseline.

Output: run_scripts/grpo_dge_verified.jsonl
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMPAT_JSON = os.path.join(HERE, "dge_grpo_compat.json")
GRPO_DATA = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR/ViLaSR-data/rl/vilasr_rl_data.jsonl"
OUT_JSONL = os.path.join(HERE, "grpo_dge_verified.jsonl")

# Task types that are reliable enough for the baseline (see compat report).
# High-confidence (>=95% agree): absolute_distance, object_counting, object_size
# Medium-confidence: relative_distance (70%), relative_direction_hard (35%), room_size (20%)
# All agreeing rows are included — the agree filter ensures DGE & GRPO match.
USABLE_TYPES = {
    "object_counting",
    "object_size",
    "absolute_distance",
    "relative_distance",
    "relative_direction_hard",
    "room_size",
}


def main():
    compat = json.load(open(COMPAT_JSON))
    print(f"[load] compat rows: {len(compat)}")

    # Index GRPO rows by question_id for joining (compat rows only kept question[:200]).
    grpo_by_id = {}
    with open(GRPO_DATA) as f:
        for line in f:
            r = json.loads(line)
            grpo_by_id[r["question_id"]] = r
    print(f"[load] grpo rows: {len(grpo_by_id)}")

    kept = 0
    skipped_join = 0
    by_type = {}
    with open(OUT_JSONL, "w") as out:
        for c in compat:
            if not c.get("is_valid") or not c.get("agree"):
                continue
            tt = c.get("task_type")
            if tt not in USABLE_TYPES:
                continue
            qid = None
            # compat rows don't carry question_id; join by matching the (truncated)
            # question prefix + scene + answer. Fall back to scanning.
            q_prefix = c["question"]
            scene = c.get("scene_id")
            grpo_ans = str(c.get("grpo_answer", "")).strip()
            cand = None
            for g in grpo_by_id.values():
                if g.get("source") not in ("SR-91k", "GPT4Scene"):
                    continue
                if g["question"][:200] == q_prefix and str(g.get("answer", "")).strip() == grpo_ans:
                    cand = g
                    break
            if cand is None:
                skipped_join += 1
                continue
            row = dict(cand)
            row["dge_task_type"] = tt
            row["dge_answer"] = c.get("dge_answer")
            row["dge_prefill"] = c.get("prefill")
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            by_type[tt] = by_type.get(tt, 0) + 1

    print(f"\n[done] wrote {kept} verified rows to {OUT_JSONL}")
    print(f"  by task_type: {by_type}")
    if skipped_join:
        print(f"  (skipped {skipped_join} rows that could not be joined back to GRPO data)")


if __name__ == "__main__":
    main()
