"""CPU unit tests for VGS-GRPO train-time search selection (tts_search.py).

Run:
    python3 run_scripts/test_tts_search.py
"""

import json
import os
import sys

_TRAINER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "easy_r1", "verl", "trainer")
)
if _TRAINER_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_DIR)

import tts_search as T  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  - {name}")
    else:
        FAIL += 1
        print(f"  FAIL- {name}")


def test_mixed_group_keeps_both_classes():
    # One group of 16: 6 correct, 10 wrong. keep=8.
    uids = ["g"] * 16
    accs = [1.0] * 6 + [0.0] * 10
    fids = [0.0] * 16
    valids = [True] * 16
    kept, m = T.select_indices(uids=uids, accs=accs, fids=fids, is_valids=valids, keep=8, seed=1)
    check("mixed: kept exactly 8", len(kept) == 8)
    n_pos = sum(1 for i in kept if accs[i] >= 0.5)
    n_neg = 8 - n_pos
    check("mixed: has >=1 positive", n_pos >= 1)
    check("mixed: has >=1 negative (contrast preserved)", n_neg >= 1)
    check("mixed: positives capped (<=6)", n_pos <= 6)
    check("mixed: fallback not triggered", m["zero_var_fallback"] == 0.0)


def test_all_correct_maze_prefers_fidelity():
    # All correct, fidelity varies -> should keep the 8 most faithful.
    uids = ["g"] * 12
    accs = [1.0] * 12
    fids = [i / 11.0 for i in range(12)]  # 0.0 .. 1.0
    valids = [True] * 12
    kept, m = T.select_indices(
        uids=uids, accs=accs, fids=fids, is_valids=valids, keep=8, fidelity_weight=0.5, seed=2
    )
    check("all-correct: kept 8", len(kept) == 8)
    kept_fids = sorted(fids[i] for i in kept)
    top8 = sorted(fids)[-8:]
    check("all-correct: kept the 8 highest-fidelity", kept_fids == top8)
    check("all-correct: no fallback (fidelity gives spread)", m["zero_var_fallback"] == 0.0)


def test_identical_group_triggers_fallback():
    uids = ["g"] * 16
    accs = [1.0] * 16
    fids = [0.0] * 16  # identical acc & fid -> zero variance
    valids = [True] * 16
    kept, m = T.select_indices(uids=uids, accs=accs, fids=fids, is_valids=valids, keep=8, seed=3)
    check("identical: kept 8", len(kept) == 8)
    check("identical: fallback triggered", m["zero_var_fallback"] == 1.0)


def test_invalid_group_keeps_first_n():
    uids = ["g"] * 16
    accs = [0.0] * 16
    fids = [0.0] * 16
    valids = [False] * 16
    kept, m = T.select_indices(uids=uids, accs=accs, fids=fids, is_valids=valids, keep=8, seed=4)
    check("invalid: kept 8", len(kept) == 8)
    check("invalid: kept the first 8 (no engine acc)", kept == list(range(8)))


def test_small_group_kept_whole():
    uids = ["g"] * 8
    accs = [1.0, 0.0] * 4
    fids = [0.0] * 8
    valids = [True] * 8
    kept, m = T.select_indices(uids=uids, accs=accs, fids=fids, is_valids=valids, keep=8, seed=5)
    check("small: group<=keep kept whole", sorted(kept) == list(range(8)))
    check("small: not counted as searched", m["groups_searched"] == 0.0)


def test_multiple_groups_and_metrics():
    # 2 groups of 16; group A mixed, group B all wrong.
    uids = ["A"] * 16 + ["B"] * 16
    accs = [1.0] * 5 + [0.0] * 11 + [0.0] * 16
    fids = [0.0] * 32
    valids = [True] * 32
    kept, m = T.select_indices(uids=uids, accs=accs, fids=fids, is_valids=valids, keep=8, seed=6)
    check("multi: kept 16 (8+8)", len(kept) == 16)
    check("multi: 2 groups searched", m["groups_searched"] == 2.0)
    # kept indices for group B must all come from B's index range [16,32)
    b_kept = [i for i in kept if i >= 16]
    check("multi: group B kept 8 from its own range", len(b_kept) == 8)


def test_score_rollout_vsi_mc():
    acc, fid = T.score_rollout(
        task_type="relative_distance",
        sequence_text="",
        response_text="<think>...</think>\n<answer>B</answer>",
        gt="chair",
        mc_options_json=json.dumps(["door", "chair", "table", "bed"]),
        gt_letter="B",
        answer_style="multiple choice",
        maze_ref_json="",
        is_valid=True,
    )
    check("score: VSI correct-letter -> acc==1", acc == 1.0)
    check("score: VSI fidelity 0 (non-maze)", fid == 0.0)

    acc2, _ = T.score_rollout(
        task_type="relative_distance",
        sequence_text="",
        response_text="<answer>A</answer>",
        gt="chair",
        mc_options_json=json.dumps(["door", "chair", "table", "bed"]),
        gt_letter="B",
        answer_style="multiple choice",
        is_valid=True,
    )
    check("score: VSI wrong-letter -> acc<1", acc2 < 1.0)

    acc3, fid3 = T.score_rollout(
        task_type="maze_navigation",
        sequence_text="",
        response_text="<answer>C</answer>",
        gt="C",
        mc_options_json="",
        gt_letter="",
        answer_style="",
        maze_ref_json="",  # no ref -> fidelity path skipped
        is_valid=True,
    )
    check("score: maze letter match -> acc==1", acc3 == 1.0)
    check("score: maze no-ref -> fid 0", fid3 == 0.0)


def test_score_rollout_maze_with_ref_runs():
    ref = {
        "grid_size": 4,
        "start": [1, 0],
        "moves": ["down", "down", "right"],
        "gt_letter": "B",
        "image_width": 476,
        "image_height": 476,
        "margins": [0.05, 0.05, 0.05, 0.05],
    }
    seq = (
        "<|im_start|>assistant\n<think>trace</think> Action: ```json\n"
        "[{\"start_point_2d\": [60, 180], \"end_point_2d\": [60, 400], \"label\": \"path\"}]\n"
        "```<|im_end|>"
        "<|im_start|>assistant\n<think>done</think> Action: <answer>B</answer><|im_end|>"
    )
    acc, fid = T.score_rollout(
        task_type="maze_navigation",
        sequence_text=seq,
        response_text="<answer>B</answer>",
        gt="B",
        maze_ref_json=json.dumps(ref),
        is_valid=True,
    )
    check("score: maze-with-ref acc==1", acc == 1.0)
    check("score: maze-with-ref fidelity is a float in [0,1]", isinstance(fid, float) and 0.0 <= fid <= 1.0)


if __name__ == "__main__":
    test_mixed_group_keeps_both_classes()
    test_all_correct_maze_prefers_fidelity()
    test_identical_group_triggers_fallback()
    test_invalid_group_keeps_first_n()
    test_small_group_kept_whole()
    test_multiple_groups_and_metrics()
    test_score_rollout_vsi_mc()
    test_score_rollout_maze_with_ref_runs()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
