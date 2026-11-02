#!/usr/bin/env python3
"""Test DGE compatibility with original GRPO RL data (v2: pre-fill extraction).

Parses GRPO question text to extract object names, pre-fills them into the
DGE input to bypass the vlm_tool, then runs WorldSimulator.validate_and_answer.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve")
from src.simulator.world_simulator import WorldSimulator

GRPO_DATA = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR/ViLaSR-data/rl/vilasr_rl_data.jsonl"
SCANNET_ROOT = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/ScanNetData_local/train"

# --- 1. Load equipped scenes ---
equipped = set()
for d in os.listdir(SCANNET_ROOT):
    if d.startswith("scene") and os.path.isdir(os.path.join(SCANNET_ROOT, d)):
        equipped.add(d)
print(f"[setup] Equipped ScanNet scenes: {len(equipped)}")

# --- 2. Question parser: map + pre-fill ---
def map_and_parse(question: str, source: str, options):
    """Return (task_type, prefill_dict) or (None, None)."""
    q = question.lower().strip()
    opts = options or []

    # object_counting: "How many book(s) are in this room?" / "How many trash can(s) are in this room?"
    m = re.search(r"how many (.+?)\(s\)\s+(?:are |is )?(?:in this room|are there|are in)", q)
    if m:
        return "object_counting", {"target_category": m.group(1).strip()}

    # absolute_distance: "direct distance between the X and the Y (in meters)"
    m = re.search(r"distance between the (.+?) and the (.+?)(?:\s*\(in meters\)|\?|$)", q)
    if m:
        o1 = m.group(1).strip().rstrip("?.,")
        o2 = m.group(2).strip().rstrip("?.,")
        return "absolute_distance", {"object1_label": o1, "object2_label": o2}

    # object_size: "length of the longest dimension ... of the X, measured in centimeters"
    # Must capture the object name right before "measured in", not "longest dimension"
    m = re.search(r"dimension.*?of the (.+?),? measured in (?:centimeters|meters)", q)
    if m:
        return "object_size", {"object_label": m.group(1).strip().rstrip("?.,")}

    # relative_distance: "which of the following objects is the closest to the X?"
    # options: ['A. box', 'B. curtain', ...]
    m = re.search(r"closest to the (.+?)\?", q)
    if m:
        target = m.group(1).strip().rstrip("?.,")
        cands = []
        for o in opts:
            # strip "A. " / "A) " prefix
            c = re.sub(r"^[A-Z][.)]\s*", "", str(o)).strip()
            if c and c.lower() != target:
                cands.append(c)
        if cands:
            return "relative_distance", {"target_label": target, "candidate_labels": cands}

    # relative_direction_hard: "If I am standing by the X and facing the Y, is the Z to ..."
    # Three variants:
    #   easy  (2-dir): "to the left or the right of the Y?"
    #   medium (3-dir): "to my left, right, or back?"
    #   hard  (8-dir): "to my front-left, front-right, back-left, or back-right?"
    m = re.search(
        r"if i am standing by the (.+?) and facing the (.+?), is the (.+?) to ",
        q,
    )
    if m:
        positioning = m.group(1).strip().rstrip("?.,")
        orienting = m.group(2).strip().rstrip("?.,")
        querying = m.group(3).strip().rstrip("?.,")
        # Determine difficulty from the option count / question phrasing
        if "front-left" in q or "back-right" in q:
            difficulty = "hard"
        elif "to my left, right, or back" in q or "left, right, or back" in q:
            difficulty = "medium"
        else:
            difficulty = "easy"
        return "relative_direction_hard", {
            "positioning_label": positioning,
            "orienting_label": orienting,
            "querying_label": querying,
            "difficulty": difficulty,
        }

    # room_size: "What is the size of this room (in square meters)?"
    if re.search(r"size of (this|the) room", q):
        return "room_size", {}

    return None, None


def extract_scene_id(image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    for p in image_paths:
        m = re.search(r"(scene\d{4}_\d{2})", p)
        if m:
            return m.group(1)
    return None


def extract_frame_id(image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    for p in image_paths:
        m = re.search(r"(\d+)\.(jpg|png)$", p)
        if m:
            return int(m.group(1))
    return 0


def normalize_answer(val):
    s = str(val).strip().upper()
    m = re.match(r"^([A-Z])[\)\.\s]", s)
    if m:
        return m.group(1)
    if len(s) == 1 and s.isalpha():
        return s
    try:
        return str(round(float(s), 1))
    except (ValueError, TypeError):
        return s


def compare_answers(grpo_ans, dge_ans, task_type, options):
    """Compare GRPO answer with DGE answer, handling format differences.

    For MC tasks, DGE may return the object label while GRPO returns the letter.
    For numerical tasks, DGE may append units (e.g. "41.18 cm") — compare with tolerance.
    """
    na = normalize_answer(grpo_ans)
    nd = normalize_answer(dge_ans)
    if na == nd:
        return True
    # For relative_distance: DGE returns object name, GRPO returns letter
    if task_type == "relative_distance" and options:
        dge_upper = str(dge_ans).strip().lower()
        for i, opt in enumerate(options):
            opt_text = re.sub(r"^[A-Z][.)]\s*", "", str(opt)).strip().lower()
            if opt_text == dge_upper:
                letter = chr(ord("A") + i)
                if na == letter:
                    return True
    # For relative_direction_hard: DGE returns direction string ("left", "front-left", etc.),
    # GRPO returns letter (A/B/C/D). Match DGE direction to option text.
    if task_type == "relative_direction_hard" and options:
        dge_dir = str(dge_ans).strip().lower()
        for i, opt in enumerate(options):
            opt_text = re.sub(r"^[A-Z][.)]\s*", "", str(opt)).strip().lower()
            if opt_text == dge_dir:
                letter = chr(ord("A") + i)
                if na == letter:
                    return True
    # For room_size: DGE returns "3.03 square meters", GRPO returns "3.0" — numerical compare below handles this
    # For numerical tasks: extract numbers and compare with tolerance
    try:
        grpo_num = float(re.search(r"[\d.]+", str(grpo_ans)).group())
        dge_num = float(re.search(r"[\d.]+", str(dge_ans)).group())
        # agreement if within 10% or within 2 units
        if abs(grpo_num - dge_num) <= max(2.0, 0.1 * max(grpo_num, dge_num)):
            return True
    except (AttributeError, ValueError):
        pass
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=200, help="max rows per category")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    # --- 3. Filter + parse GRPO rows ---
    matched = []
    cat_counts = Counter()
    parse_fail = 0
    for line in open(GRPO_DATA):
        r = json.loads(line)
        src = r.get("source", "")
        if src not in ("SR-91k", "GPT4Scene"):
            continue
        scene_id = extract_scene_id(r.get("image_path", []))
        if not scene_id or scene_id not in equipped:
            continue
        tt, prefill = map_and_parse(r.get("question", ""), src, r.get("options"))
        if tt is None:
            continue
        if prefill is None:
            parse_fail += 1
            continue
        cat_counts[tt] += 1
        matched.append((scene_id, tt, prefill, r))

    print(f"[filter] DGE-mappable + parsed GRPO rows: {len(matched)} (parse_fail={parse_fail})")
    print(f"  by task_type: {dict(cat_counts)}")

    if not args.all:
        by_cat = defaultdict(list)
        for row in matched:
            by_cat[row[1]].append(row)
        sampled = []
        for cat, rows in by_cat.items():
            sampled.extend(rows[:args.max])
        matched = sampled
        print(f"[sample] Testing {len(matched)} rows (max {args.max}/category)")

    # --- 4. Run DGE ---
    print("[dge] Initializing WorldSimulator (enable_vlm=False, pre-fill mode)...")
    ws = WorldSimulator(scannet_root=SCANNET_ROOT, enable_vlm=False)

    results = []
    valid_count = 0
    agree_count = 0
    for i, (scene_id, task_type, prefill, r) in enumerate(matched):
        if i % 50 == 0:
            print(f"  [dge] {i}/{len(matched)} (valid={valid_count} agree={agree_count})")
        q = r.get("question", "")
        grpo_ans = r.get("answer", "")
        frame_id = extract_frame_id(r.get("image_path", []))

        input_data = {
            "task_type": task_type,
            "question": q,
            "scene_id": scene_id,
            "frame_id": frame_id,
        }
        input_data.update(prefill)

        try:
            resp = ws.validate_and_answer(input_data)
        except Exception as e:
            resp = {"is_valid": False, "error": str(e), "answer": None}

        is_valid = resp.get("is_valid", False)
        dge_ans = resp.get("answer", None)
        error = resp.get("error", "")

        agree = False
        if is_valid and dge_ans is not None:
            valid_count += 1
            agree = compare_answers(grpo_ans, dge_ans, task_type, r.get("options"))
            if agree:
                agree_count += 1

        results.append({
            "scene_id": scene_id,
            "task_type": task_type,
            "source": r.get("source"),
            "question": q[:200],
            "grpo_answer": str(grpo_ans),
            "dge_answer": str(dge_ans) if dge_ans is not None else None,
            "is_valid": is_valid,
            "agree": agree,
            "prefill": prefill,
            "error": error[:150] if not is_valid else "",
        })

    # --- 5. Report ---
    total = len(results)
    valid = sum(1 for r in results if r["is_valid"])
    agree = sum(1 for r in results if r["agree"])
    print(f"\n{'='*70}")
    print(f"RESULTS: tested={total}  valid={valid} ({100*valid/total:.1f}%)  "
          f"agree={agree} ({100*agree/valid if valid else 0:.1f}% of valid)")

    by_tt = defaultdict(lambda: {"tested": 0, "valid": 0, "agree": 0})
    for r in results:
        by_tt[r["task_type"]]["tested"] += 1
        if r["is_valid"]:
            by_tt[r["task_type"]]["valid"] += 1
        if r["agree"]:
            by_tt[r["task_type"]]["agree"] += 1
    print(f"\nPer task_type:")
    print(f"  {'task_type':30s} {'tested':>7s} {'valid':>6s} {'agree':>6s} {'valid%':>7s} {'agree%':>7s}")
    for tt, c in sorted(by_tt.items()):
        vp = 100 * c["valid"] / c["tested"] if c["tested"] else 0
        ap = 100 * c["agree"] / c["valid"] if c["valid"] else 0
        print(f"  {tt:30s} {c['tested']:7d} {c['valid']:6d} {c['agree']:6d} {vp:6.1f}% {ap:6.1f}%")

    err_counts = Counter()
    for r in results:
        if not r["is_valid"]:
            err_counts[r["error"][:80]] += 1
    if err_counts:
        print(f"\nTop invalid reasons:")
        for err, cnt in err_counts.most_common(10):
            print(f"  [{cnt:4d}] {err}")

    # Sample valid agreements + disagreements
    print(f"\nSample valid agreements:")
    for r in [x for x in results if x["agree"]][:5]:
        print(f"  [{r['task_type']}] GRPO={r['grpo_answer']} DGE={r['dge_answer']} | {r['question'][:100]}")
    print(f"\nSample valid disagreements:")
    for r in [x for x in results if x["is_valid"] and not x["agree"]][:8]:
        print(f"  [{r['task_type']}] GRPO={r['grpo_answer']} DGE={r['dge_answer']} | {r['question'][:100]}")
        print(f"    prefill={r.get('prefill')}")

    out_path = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/run_scripts/dge_grpo_compat.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
