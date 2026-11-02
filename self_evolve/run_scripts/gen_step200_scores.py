#!/usr/bin/env python3
"""Generate scores.json for the imitate_proposer_step200 eval results.

Reads scores.jsonl from each benchmark folder and aggregates into a scores.json
matching the format of benchmark_result/ViLaSR/<model>/scores.json.

VSI-Bench and SPAR-Bench use the OFFICIAL category-mean overall (mirrors
ViLaSR/utils/cal_metric_{vsibench,sparbench}.py). MAZE / MMSI / SpatialEval
remain sample-weighted (no official per-category aggregator).
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# 8 canonical VSI-Bench categories, official display order
VSI_CATS = [
    "object_counting", "object_abs_distance", "object_size_estimation",
    "room_size_estimation", "object_rel_distance", "object_rel_direction",
    "route_planning", "obj_appearance_order",
]

# SPAR-Bench category grouping (20 types) mirroring cal_metric_sparbench.py.
SPAR_LOW = {
    "depth_prediction_oc", "depth_prediction_oc_mv",
    "depth_prediction_oo", "depth_prediction_oo_mv",
    "distance_prediction_oc", "distance_prediction_oc_mv",
    "distance_prediction_oo", "distance_prediction_oo_mv",
}
SPAR_MED = {"position_matching", "camera_motion_infer", "view_change_infer"}
SPAR_HIGH = {
    "distance_infer_center_oo", "distance_infer_center_oo_mv",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo", "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc", "spatial_imagination_oc_mv",
    "spatial_imagination_oo", "spatial_imagination_oo_mv",
}
SPAR_CATS_ORDER = (
    "depth_prediction_oc", "depth_prediction_oc_mv",
    "depth_prediction_oo", "depth_prediction_oo_mv",
    "distance_prediction_oc", "distance_prediction_oc_mv",
    "distance_prediction_oo", "distance_prediction_oo_mv",
    "position_matching", "camera_motion_infer", "view_change_infer",
    "distance_infer_center_oo", "distance_infer_center_oo_mv",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo", "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc", "spatial_imagination_oc_mv",
    "spatial_imagination_oo", "spatial_imagination_oo_mv",
)
SPAR_GROUPS = {"Low": SPAR_LOW, "Medium": SPAR_MED, "High": SPAR_HIGH}


def _spar_group(qtype: str) -> str:
    for name, s in SPAR_GROUPS.items():
        if qtype in s:
            return name
    return "Other"


def _merge_rel_direction(qtype: str) -> str:
    return re.sub(r"^(object_rel_direction)_(easy|medium|hard)$", r"\1", qtype)


def compute_vsi_bench_official(rows):
    merged = defaultdict(list)
    raw = defaultdict(list)
    for r in rows:
        s = r.get("score")
        if s is None:
            continue
        raw_t = r["original_question_type"]
        merged[_merge_rel_direction(raw_t)].append(float(s))
        raw[raw_t].append(float(s))
    per_cat = {t: sum(v) / len(v) * 100 for t, v in merged.items()}
    per_cat_n = {t: len(v) for t, v in merged.items()}
    all_scores = [s for v in merged.values() for s in v]
    sample_wtd = (sum(all_scores) / max(1, len(all_scores))) * 100
    valid_means = [per_cat[c] for c in VSI_CATS if c in per_cat]
    overall = sum(valid_means) / len(valid_means) if valid_means else 0.0
    entry = {
        "available": True,
        "overall": round(overall, 4),
        "overall_method": "VSI-Bench official: mean of 8 category scores (rel_direction merged from easy+medium+hard, sample-weighted internal)",
        "overall_sample_weighted": round(sample_wtd, 4),
        "num_scored": len(rows),
        "num_shards": 4,
        "num_categories_scored": len(valid_means),
    }
    for c in VSI_CATS:
        if c in per_cat:
            entry[c] = round(per_cat[c], 4)
            entry[f"{c}__n"] = per_cat_n[c]
    entry["per_subtype"] = {
        t: {"score": round(sum(v) / len(v) * 100, 4), "n": len(v)}
        for t, v in sorted(raw.items())
    }
    return entry

OUT_ROOT = Path("/data/pengtianshuo/self_evolve/Xinran/benchmark_result/vilasr_self_evolve_imitate_porposer_step200")

# (key_in_scores_json, folder_name, type_field)
BENCHES = [
    ("vsi_bench", "VSI-Bench", "original_question_type"),
    ("maze", "MAZE", None),
    ("SpatialEval_spatialreal", "SpatialEval-Real", None),
    ("spar_bench", "SPAR-Bench", "original_question_type"),
    ("mmsi_bench", "MMSI-Bench", "original_question_type"),
]


def load_scores(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_spar_bench_official(rows):
    """SPAR-Bench official: mean of 20 category scores (equal weight per type)."""
    per_type = defaultdict(list)
    for r in rows:
        s = r.get("score")
        if s is None:
            continue
        per_type[r["original_question_type"]].append(float(s))
    per_type_mean = {t: sum(v) / len(v) * 100 for t, v in per_type.items()}
    per_type_n = {t: len(v) for t, v in per_type.items()}
    included_means = [per_type_mean[t] for t in per_type_mean if _spar_group(t) != "Other"]
    overall = sum(included_means) / len(included_means) if included_means else 0.0
    all_scores = [s for v in per_type.values() for s in v]
    sample_wtd = (sum(all_scores) / max(1, len(all_scores))) * 100

    group_stats = {}
    for group_name, type_set in SPAR_GROUPS.items():
        merged = [v for t, vs in per_type.items() if t in type_set for v in vs]
        n = len(merged)
        group_stats[group_name] = {
            "count": n,
            "sample_weighted_mean": round(sum(merged) / n * 100, 4) if n else 0.0,
        }

    entry = {
        "available": True,
        "overall": round(overall, 4),
        "overall_method": "SPAR-Bench official: mean of 20 category scores (Low+Medium+High groups, equal weight per type)",
        "overall_sample_weighted": round(sample_wtd, 4),
        "num_scored": len(rows),
        "num_shards": 4,
        "num_categories_scored": len(included_means),
        "group_stats": group_stats,
    }
    for t in SPAR_CATS_ORDER:
        if t in per_type_mean:
            entry[t] = round(per_type_mean[t], 4)
            entry[f"{t}__n"] = per_type_n[t]
    for t, m in sorted(per_type_mean.items()):
        if t not in SPAR_CATS_ORDER:
            entry[t] = round(m, 4)
            entry[f"{t}__n"] = per_type_n[t]
    return entry


def compute_bench(rows, type_field, bench_key=None):
    if not rows:
        return {"available": False}
    if bench_key == "vsi_bench":
        return compute_vsi_bench_official(rows)
    if bench_key == "spar_bench":
        return compute_spar_bench_official(rows)
    n = len(rows)
    # overall = mean score * 100  (sample-weighted; used for non-VSI benches)
    scores = [r.get("score", 0) for r in rows if r.get("score") is not None]
    overall = round(sum(scores) / len(scores) * 100, 4) if scores else 0.0
    entry = {
        "available": True,
        "overall": overall,
        "num_scored": n,
        "num_shards": 4,
    }
    # per-type breakdown
    if type_field:
        by_type = {}
        for r in rows:
            t = r.get(type_field, "unknown")
            s = r.get("score", 0)
            by_type.setdefault(t, []).append(s)
        for t, ss in sorted(by_type.items()):
            mean_s = round(sum(ss) / len(ss) * 100, 4) if ss else 0.0
            # VSI uses MRA suffix naming; SPAR/MMSI use plain type name
            if type_field == "original_question_type" and t and "vsi" in str(rows[0].get("data_source", "")).lower():
                pass  # keep plain
            entry[t] = mean_s
    return entry


def main():
    for label in ["selfevolve_step200", "selfevolve_step150"]:
        model_dir = OUT_ROOT / label
        if not model_dir.exists():
            print(f"skip {label}: dir not found")
            continue
        out = {"checkpoint": label, "benchmarks": {}}
        for bkey, bfolder, type_field in BENCHES:
            sj = model_dir / bfolder / "scores.jsonl"
            if not sj.exists():
                out["benchmarks"][bkey] = {"available": False}
                continue
            rows = load_scores(sj)
            out["benchmarks"][bkey] = compute_bench(rows, type_field, bkey)
        # overall mean over available
        ovs = [v["overall"] for v in out["benchmarks"].values()
               if v.get("available") and v.get("overall") is not None]
        out["overall_mean"] = round(sum(ovs) / len(ovs), 2) if ovs else None
        out["num_benchmarks_evaluated"] = len(ovs)

        outpath = model_dir / "scores.json"
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"=== {label} ===")
        print(f"  benchmarks evaluated: {out['num_benchmarks_evaluated']}/5")
        print(f"  overall_mean: {out['overall_mean']}")
        for bkey, v in out["benchmarks"].items():
            if v.get("available"):
                print(f"  {bkey}: {v['overall']:.2f}  ({v['num_scored']} scored)")
            else:
                print(f"  {bkey}: NOT available")
        print(f"  saved: {outpath}")
        print()


if __name__ == "__main__":
    main()
