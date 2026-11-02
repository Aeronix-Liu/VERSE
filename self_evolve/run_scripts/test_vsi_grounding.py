#!/usr/bin/env python3
"""CPU unit tests for the VSI grounding process reward (2026-08-20).

Covers:
  1. _bbox_iou geometry.
  2. vsi_grounding_fidelity scorer on synthetic refs (perfect / partial / none).
  3. _vsi_grounding_weight env clamping.
  4. Real-data ref build (label extraction via DGE_TASK_CONFIG + real ScanNet
     per-frame metadata + processed-frame scaling) fed through the scorer.
  5. End-to-end compute_score reward shaping for a valid scene rollout.

Run: python3 run_scripts/test_vsi_grounding.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_DIR = os.path.join(REPO_ROOT, "easy_r1", "training", "reward_function")
for p in (REPO_ROOT, REWARD_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import agentic_solver_reward as G  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name, cond, extra=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name} {extra}")


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------
def test_iou():
    print("test_iou")
    check("identical box IoU=1", approx(G._bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0))
    check("disjoint box IoU=0", approx(G._bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0))
    # half overlap: A=[0,0,10,10]=100, B=[5,0,15,10]=100, inter=[5,0,10,10]=50, union=150
    check("half-overlap IoU=1/3", approx(G._bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]), 50.0 / 150.0))
    check("degenerate box IoU=0", approx(G._bbox_iou([0, 0, 0, 0], [0, 0, 10, 10]), 0.0))
    check("swapped coords handled", approx(G._bbox_iou([10, 10, 0, 0], [0, 0, 10, 10]), 1.0))


def test_scorer_synthetic():
    print("test_scorer_synthetic")
    ref = {
        "relevant_labels": ["chair", "table"],
        "gt_by_label": {"chair": [[0, 0, 10, 10]], "table": [[20, 20, 40, 40]]},
    }
    # Perfect grounding of both labels.
    boxes = [
        {"index": 0, "bbox_2d": [0, 0, 10, 10], "label": "chair"},
        {"index": 1, "bbox_2d": [20, 20, 40, 40], "label": "table"},
    ]
    r = G.vsi_grounding_fidelity(boxes, ref)
    check("perfect grounding=1.0", approx(r["grounding"], 1.0), r)
    check("perfect hit_rate=1.0", approx(r["grounding_hit_rate"], 1.0))
    check("perfect coverage=1.0", approx(r["coverage"], 1.0))

    # Only chair drawn (perfectly) -> grounding 0.5, coverage 0.5, hit 0.5.
    r = G.vsi_grounding_fidelity([{"index": 0, "bbox_2d": [0, 0, 10, 10], "label": "chair"}], ref)
    check("one-of-two grounding=0.5", approx(r["grounding"], 0.5), r)
    check("one-of-two coverage=0.5", approx(r["coverage"], 0.5))
    check("one-of-two hit_rate=0.5", approx(r["grounding_hit_rate"], 0.5))
    check("mean_matched_iou=1.0 (only drawn label perfect)", approx(r["mean_matched_iou"], 1.0))

    # No boxes -> all zero.
    r = G.vsi_grounding_fidelity([], ref)
    check("empty grounding=0", approx(r["grounding"], 0.0))
    check("empty coverage=0", approx(r["coverage"], 0.0))

    # Wrong location -> low IoU -> grounding low, coverage still counts the draw.
    r = G.vsi_grounding_fidelity(
        [{"index": 0, "bbox_2d": [500, 500, 510, 510], "label": "chair"},
         {"index": 0, "bbox_2d": [500, 500, 510, 510], "label": "table"}],
        ref,
    )
    check("wrong-loc grounding~0", r["grounding"] < 0.05, r)
    check("wrong-loc coverage=1.0", approx(r["coverage"], 1.0))
    check("wrong-loc hit_rate=0", approx(r["grounding_hit_rate"], 0.0))

    # Plural / contains-match: GT 'chair', drawn 'chairs'.
    r = G.vsi_grounding_fidelity([{"index": 0, "bbox_2d": [0, 0, 10, 10], "label": "chairs"}],
                                 {"relevant_labels": ["chair"], "gt_by_label": {"chair": [[0, 0, 10, 10]]}})
    check("plural contains-match grounding=1.0", approx(r["grounding"], 1.0), r)

    # Multi-instance: best IoU over instances is taken.
    r = G.vsi_grounding_fidelity(
        [{"index": 0, "bbox_2d": [100, 100, 110, 110], "label": "chair"}],
        {"relevant_labels": ["chair"], "gt_by_label": {"chair": [[0, 0, 10, 10], [100, 100, 110, 110]]}},
    )
    check("multi-instance picks best=1.0", approx(r["grounding"], 1.0), r)

    # Empty relevant_labels -> zeros, no crash.
    r = G.vsi_grounding_fidelity(boxes, {"relevant_labels": [], "gt_by_label": {}})
    check("no relevant labels grounding=0", approx(r["grounding"], 0.0))


def test_weight_env():
    print("test_weight_env")
    old = os.environ.get("SPATIALEVO_VSI_GROUNDING_WEIGHT")
    try:
        os.environ["SPATIALEVO_VSI_GROUNDING_WEIGHT"] = "0"
        check("weight default 0", approx(G._vsi_grounding_weight(), 0.0))
        os.environ["SPATIALEVO_VSI_GROUNDING_WEIGHT"] = "0.2"
        check("weight 0.2", approx(G._vsi_grounding_weight(), 0.2))
        os.environ["SPATIALEVO_VSI_GROUNDING_WEIGHT"] = "5.0"
        check("weight clamped to 0.8", approx(G._vsi_grounding_weight(), 0.8))
        os.environ["SPATIALEVO_VSI_GROUNDING_WEIGHT"] = "junk"
        check("weight junk -> 0", approx(G._vsi_grounding_weight(), 0.0))
    finally:
        if old is None:
            os.environ.pop("SPATIALEVO_VSI_GROUNDING_WEIGHT", None)
        else:
            os.environ["SPATIALEVO_VSI_GROUNDING_WEIGHT"] = old


def _build_ref_core(task_type, parsed_params, image_paths):
    """Standalone replica of RayTrainer._build_vsi_grounding_refs core, using the
    same real functions (DGE_TASK_CONFIG label extraction + real metadata +
    process_image_agentic scaling). Returns the ref dict (or None)."""
    from collections import defaultdict
    from PIL import Image as _PILImage
    from src.simulator.structured_question_renderer import TASK_CONFIG as DGE_TASK_CONFIG
    from easy_r1.verl.utils.agentic import process_image_agentic

    cfg = DGE_TASK_CONFIG.get(task_type)
    if not cfg:
        return None
    labels = []
    for f in cfg.get("dge_to_template", {}).keys():
        v = parsed_params.get(f)
        if isinstance(v, list):
            labels.extend(str(x).strip() for x in v if str(x).strip())
        elif v is not None and str(v).strip():
            labels.append(str(v).strip())
    seen, uniq = set(), []
    for l in labels:
        if l.lower() not in seen:
            seen.add(l.lower()); uniq.append(l)
    labels = uniq
    if not labels:
        return None
    label_keys = {l.lower() for l in labels}
    MIN_PX, MAX_PX = 200704, 351232  # only affects processed size; scaling is ratio-based
    gt_by_label = defaultdict(list)
    for p in image_paths:
        meta_path = os.path.splitext(str(p).replace("/train/", "/metadata/"))[0] + ".json"
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            fm = json.load(f)
        with _PILImage.open(p) as im:
            ow, oh = im.size
        proc = process_image_agentic(str(p), MIN_PX, MAX_PX)
        pw, ph = proc.size
        sx, sy = pw / max(1, ow), ph / max(1, oh)
        for obj in (fm.get("objects") or []):
            lbl = str(obj.get("label", "")).strip().lower()
            if not lbl or not any(lbl == k or lbl in k or k in lbl for k in label_keys):
                continue
            if float(obj.get("visibility", 1.0)) < 0.1:
                continue
            b = obj.get("2d_bbox")
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                continue
            gt_by_label[lbl].append([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy])
    if not gt_by_label:
        return None
    return {"relevant_labels": labels, "gt_by_label": {k: v for k, v in gt_by_label.items()}}


def test_real_data_ref():
    print("test_real_data_ref")
    # Find a scene sample with >=3 frames from the training jsonl.
    jsonl = os.path.join(REPO_ROOT, "easy_r1", "dataset", "full_selfevolve_train_v3.jsonl")
    image_paths = None
    for line in open(jsonl):
        r = json.loads(line)
        imgs = r.get("images") or []
        if len(imgs) >= 3 and all(os.path.exists(p) for p in imgs[:1]):
            image_paths = imgs
            break
    if not image_paths:
        check("found a scene sample", False, "no scene sample with existing frames")
        return

    # Pick a label that actually appears in the first frame's metadata.
    meta0 = os.path.splitext(str(image_paths[0]).replace("/train/", "/metadata/"))[0] + ".json"
    if not os.path.exists(meta0):
        check("first frame metadata exists", False, meta0)
        return
    objs = json.load(open(meta0)).get("objects") or []
    vis = [o for o in objs if float(o.get("visibility", 1.0)) >= 0.1 and o.get("2d_bbox")]
    if not vis:
        check("first frame has a visible labeled object", False)
        return
    target = str(vis[0]["label"]).strip().lower()

    ref = _build_ref_core("object_counting", {"target_category": target}, image_paths)
    check("real ref built", ref is not None and bool(ref.get("gt_by_label")), ref)
    if not ref:
        return
    check("real ref has target label", any(target == k or target in k or k in target for k in ref["gt_by_label"]))

    # Draw a box equal to one scaled GT instance of the target -> grounding ~1.0.
    gt_boxes = None
    for k, v in ref["gt_by_label"].items():
        if target == k or target in k or k in target:
            gt_boxes = v
            break
    check("target has >=1 GT box", gt_boxes and len(gt_boxes) >= 1)
    if gt_boxes:
        drawn = [{"index": 0, "bbox_2d": list(gt_boxes[0]), "label": target}]
        r = G.vsi_grounding_fidelity(drawn, ref)
        check("real GT-matching box grounding>=0.99", r["grounding"] >= 0.99, r)
        # Bogus far-away box -> grounding ~0.
        r0 = G.vsi_grounding_fidelity([{"index": 0, "bbox_2d": [9998, 9998, 9999, 9999], "label": target}], ref)
        check("real bogus box grounding~0", r0["grounding"] < 0.05, r0)


def test_compute_score_shaping():
    print("test_compute_score_shaping")
    ref = {"relevant_labels": ["chair"], "gt_by_label": {"chair": [[0, 0, 10, 10]]}}
    seq = (
        "<|im_start|>assistant\n<think>look</think> Action:\n```json\n"
        "[{\"index\": 0, \"bbox_2d\": [0, 0, 10, 10], \"label\": \"chair\"}]\n```\n<|im_end|>"
        "<|im_start|>assistant\n<think>done</think> Action: answer <answer>3</answer><|im_end|>"
    )
    ri = {
        "uid": "u1",
        "question_type": "object_counting",
        "is_valid": True,
        "gt": "3",
        "agentic_sequence_text": seq,
        "agentic_response_text": "<answer>3</answer>",
        "vsi_grounding_ref_json": json.dumps(ref),
        "mc_options_json": "",
        "gt_letter": "",
        "answer_style": "numerical",
    }
    old = os.environ.get("SPATIALEVO_VSI_GROUNDING_WEIGHT")
    try:
        # Weight on: grounding metric present + overall follows shaping formula.
        os.environ["SPATIALEVO_VSI_GROUNDING_WEIGHT"] = "0.2"
        s = G.compute_score([dict(ri)])[0]
        check("score has vsi_grounding", "vsi_grounding" in s, s)
        # Standalone grounding for the same drawn box.
        exp_g = G.vsi_grounding_fidelity(
            [{"index": 0, "bbox_2d": [0, 0, 10, 10], "label": "chair"}], ref
        )["grounding"]
        check("vsi_grounding matches standalone", approx(s.get("vsi_grounding", -1), exp_g), (s.get("vsi_grounding"), exp_g))
        expected_overall = 0.1 * s["format"] + (0.9 - 0.2) * s["accuracy"] + 0.2 * s["vsi_grounding"]
        check("overall follows w=0.2 shaping", approx(s["overall"], expected_overall), (s["overall"], expected_overall))

        # Weight off (diagnostic): grounding still reported, overall is pure outcome.
        os.environ["SPATIALEVO_VSI_GROUNDING_WEIGHT"] = "0"
        s0 = G.compute_score([dict(ri)])[0]
        check("w=0 still reports grounding", "vsi_grounding" in s0)
        expected0 = 0.1 * s0["format"] + 0.9 * s0["accuracy"]
        check("w=0 overall = pure outcome", approx(s0["overall"], expected0), (s0["overall"], expected0))
    finally:
        if old is None:
            os.environ.pop("SPATIALEVO_VSI_GROUNDING_WEIGHT", None)
        else:
            os.environ["SPATIALEVO_VSI_GROUNDING_WEIGHT"] = old


if __name__ == "__main__":
    test_iou()
    test_scorer_synthetic()
    test_weight_env()
    try:
        test_real_data_ref()
    except Exception as e:
        _FAIL += 1
        print(f"  [FAIL] test_real_data_ref raised: {e}")
    try:
        test_compute_score_shaping()
    except Exception as e:
        _FAIL += 1
        print(f"  [FAIL] test_compute_score_shaping raised: {e}")
    print(f"\n==== {_PASS} passed, {_FAIL} failed ====")
    sys.exit(1 if _FAIL else 0)
