#!/usr/bin/env python3
"""CPU unit tests for ABLATION #1 (self-evolve without GT, majority-vote reward).

Run:  python3 ablations/no_gt_majority_vote/test_majority_reward.py
All tests are pure-CPU (no GPU, no network) and must pass before launching.
"""

import os
import sys

_ABL_DIR = os.path.dirname(os.path.abspath(__file__))
_RF_DIR = os.path.join(_ABL_DIR, "reward_function")
for _p in (_RF_DIR, _ABL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agentic_solver_majority_reward as M  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {extra}")


def _mc_answer(letter):
    return {"agentic_response_text": f"<answer>{letter}</answer>"}


def _mk(uid, q_type, answer_style, ans_text, gt, mc_options_json="", is_valid=True):
    d = {
        "uid": uid,
        "question_type": q_type,
        "answer_style": answer_style,
        "mc_options_json": mc_options_json,
        "gt": gt,
        "is_valid": is_valid,
        "round": 2,
        "agentic_response_text": f"<answer>{ans_text}</answer>",
    }
    return d


# --------------------------------------------------------------------------
print("== 1. _group_consensus unit ==")

# MC: mode of letters (A wins 4:2:1:1); gt deliberately different (B).
gt_letter, letter = M._group_consensus(
    ["A", "A", "A", "A", "B", "B", "C", "D"],
    q_type="relative_distance",
    answer_style="multiple choice",
    mc_options_json='["w","x","y","z"]',
)
check("MC consensus letter == A", letter == "A", f"got {letter}")
check("MC consensus gt maps to option 'w'", gt_letter == "w", f"got {gt_letter}")

# maze: consensus letter returned as gt directly.
g, l = M._group_consensus(["A", "A", "B"], q_type="maze_navigation", answer_style="", mc_options_json="")
check("maze consensus gt==letter==A", g == "A" and l == "A", f"got {g},{l}")

# counting: integer mode (3 appears 3x).
g, _ = M._group_consensus(["3", "3", "3", "5", "4"], q_type="object_counting", answer_style="numerical", mc_options_json="")
check("count consensus == '3'", g == "3", f"got {g}")

# numeric continuous: median of [1.9,2.0,2.1,5.0] == 2.05.
g, _ = M._group_consensus(["1.9", "2.0", "2.1", "5.0"], q_type="absolute_distance", answer_style="numerical", mc_options_json="")
check("dist consensus median == 2.05", abs(float(g) - 2.05) < 1e-6, f"got {g}")

# no parseable answers -> (None, "")
g, l = M._group_consensus(["", "  "], q_type="absolute_distance", answer_style="numerical", mc_options_json="")
check("empty group -> None", g is None, f"got {g}")


# --------------------------------------------------------------------------
print("== 2. compute_score: MC majority reward, GT-independent ==")

opts = '["left","right","closer","farther"]'
group = []
for a in ["A", "A", "A", "A", "B", "B", "C", "D"]:
    group.append(_mk("q1", "relative_distance", "multiple choice", a, gt="right", mc_options_json=opts))
scores = M.compute_score(group)
accs = [s["accuracy"] for s in scores]
# First 4 (A) are majority -> 1.0; rest -> 0.0.
check("MC majority: 4x A get acc 1.0", accs[:4] == [1.0, 1.0, 1.0, 1.0], f"got {accs[:4]}")
check("MC majority: minority get acc 0.0", accs[4:] == [0.0, 0.0, 0.0, 0.0], f"got {accs[4:]}")

# Now flip the ENGINE GT to 'left' (letter A) -- reward must be IDENTICAL (no GT).
group2 = []
for a in ["A", "A", "A", "A", "B", "B", "C", "D"]:
    group2.append(_mk("q1", "relative_distance", "multiple choice", a, gt="left", mc_options_json=opts))
scores2 = M.compute_score(group2)
accs2 = [s["accuracy"] for s in scores2]
check("reward is GT-independent (accs unchanged)", accs2 == accs, f"{accs2} vs {accs}")

# overall = 0.1*fmt(0) + 0.9*acc  (sequence_text empty -> fmt 0)
check("overall == 0.9*acc for majority A", abs(scores[0]["overall"] - 0.9) < 1e-9, f"got {scores[0]['overall']}")


# --------------------------------------------------------------------------
print("== 3. compute_score: counting (graded) ==")

cgroup = []
for a in ["3", "3", "3", "5", "4"]:
    cgroup.append(_mk("qc", "object_counting", "numerical", a, gt="7"))
cs = M.compute_score(cgroup)
cacc = [round(s["accuracy"], 3) for s in cs]
# consensus 3: '3'->1.0, '5'->abs2->0.1, '4'->abs1->0.3
check("count acc [1,1,1,0.1,0.3]", cacc == [1.0, 1.0, 1.0, 0.1, 0.3], f"got {cacc}")


# --------------------------------------------------------------------------
print("== 4. compute_score: maze majority ==")

mgroup = []
for a in ["A", "A", "A", "B", "C"]:
    mgroup.append(_mk("qm", "maze_navigation", "", a, gt="D", mc_options_json='["A","B","C","D"]'))
ms = M.compute_score(mgroup)
macc = [s["accuracy"] for s in ms]
check("maze majority A -> [1,1,1,0,0]", macc == [1.0, 1.0, 1.0, 0.0, 0.0], f"got {macc}")


# --------------------------------------------------------------------------
print("== 5. compute_score: diagnostics present ==")
# consensus_ratio for q1 group = 4/8 = 0.5; pseudo_vs_true_acc computed per-sample.
diag_group = []
for a in ["A", "A", "A", "A", "B", "B", "C", "D"]:
    g = _mk("qd", "relative_distance", "multiple choice", a, gt="left", mc_options_json=opts)
    g["gt_letter"] = "A"  # engine GT letter (only used by the diagnostic, never the reward)
    diag_group.append(g)
ds = M.compute_score(diag_group)
cr = ds[0].get("majority_vote/consensus_ratio")
check("consensus_ratio == 0.5", cr is not None and abs(cr - 0.5) < 1e-9, f"got {cr}")
# gt='left' == letter A, so majority (A) rollouts are TRUE-correct: pseudo_vs_true_acc==1 for them.
tva = [s.get("majority_vote/pseudo_vs_true_acc") for s in ds]
check("pseudo_vs_true_acc present on A rollouts", tva[0] == 1.0, f"got {tva[0]}")


# --------------------------------------------------------------------------
print("== 6. invalid questions still routed to judge branch (score 0 w/o network) ==")
inv = [_mk("qi", "object_counting", "numerical", "", gt="3", is_valid=False)]
inv[0]["agentic_response_text"] = ""  # no explanation -> no judge job, stays 0
isc = M.compute_score(inv)
check("invalid -> valid_case 0", isc[0]["valid_case"] == 0.0, f"got {isc[0]['valid_case']}")


# --------------------------------------------------------------------------
print("== 7. unified dispatcher routes round=2 to majority ==")
import unified_reward_majority as U  # noqa: E402

u_scores = U.compute_score(group, majority_correctness_kwargs={})
check("unified round=2 matches majority reward", [s["accuracy"] for s in u_scores] == accs, "mismatch")


print(f"\n==== {PASS} passed, {FAIL} failed ====")
sys.exit(1 if FAIL else 0)
