#!/usr/bin/env python3
"""Regenerate VSI-Bench and SPAR-Bench sections of scores.json using the
OFFICIAL algorithms from the ViLaSR repo.

VSI-Bench  <- mirrors `ViLaSR/utils/cal_metric_vsibench.py`:
  * bucket by `original_question_type`;
  * merge `object_rel_direction_{easy,medium,hard}` -> `object_rel_direction`
    (sample-weighted internal concatenation, then mean);
  * overall = arithmetic mean of the 8 category means (equal weight per type).

SPAR-Bench <- mirrors `ViLaSR/utils/cal_metric_sparbench.py`:
  * 20 task_types grouped into Low/Medium/High buckets (see SPAR_LOW/MED/HIGH
    below);
  * overall = arithmetic mean of the 20 category means (equal weight per type;
    Low/Med/High groups are only used for display, NOT weighted separately).

MAZE / MMSI-Bench / SpatialEval-Real have no official per-category aggregator
in the ViLaSR repo — they are sample-weighted (single-category benches). This
script leaves them untouched.

For each affected scores.json:
  * only the `vsi_bench` and/or `spar_bench` blocks are rewritten;
  * the previous sample-weighted overall is preserved under
    `overall_sample_weighted`;
  * `overall_method` records the algorithm used;
  * top-level `overall_mean` is recomputed (over available benches).

Idempotent: rerunning it yields the same values.
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

TARGETS = [
    "/data/pengtianshuo/self_evolve/Xinran/benchmark_result/vilasr_self_evolve_imitate_porposer",
    "/data/pengtianshuo/self_evolve/Xinran/benchmark_result/vilasr_self_evolve_imitate_porposer_step200",
    "/data/pengtianshuo/self_evolve/Xinran/benchmark_result/vilasr-self-evolve-imitate-0808",
]

# 8 canonical VSI-Bench category names (as they appear in each row's
# `original_question_type` field, AFTER merging the rel_direction difficulty
# variants).
VSI_CATS = [
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
]

# SPAR-Bench category grouping (from cal_metric_sparbench.py; 20 types total).
SPAR_LOW = {
    "depth_prediction_oc", "depth_prediction_oc_mv",
    "depth_prediction_oo", "depth_prediction_oo_mv",
    "distance_prediction_oc", "distance_prediction_oc_mv",
    "distance_prediction_oo", "distance_prediction_oo_mv",
}
SPAR_MED = {
    "position_matching", "camera_motion_infer", "view_change_infer",
}
SPAR_HIGH = {
    "distance_infer_center_oo", "distance_infer_center_oo_mv",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo", "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc", "spatial_imagination_oc_mv",
    "spatial_imagination_oo", "spatial_imagination_oo_mv",
}
SPAR_CATS_ORDER = (
    # Low
    "depth_prediction_oc", "depth_prediction_oc_mv",
    "depth_prediction_oo", "depth_prediction_oo_mv",
    "distance_prediction_oc", "distance_prediction_oc_mv",
    "distance_prediction_oo", "distance_prediction_oo_mv",
    # Medium
    "position_matching", "camera_motion_infer", "view_change_infer",
    # High
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
    """object_rel_direction_{easy,medium,hard} -> object_rel_direction."""
    return re.sub(r"^(object_rel_direction)_(easy|medium|hard)$", r"\1", qtype)


def load_scored_rows(path: Path):
    """Load rows from a benchmark's scores.jsonl (each row has a `score`)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("score") is None:
                continue
            rows.append(r)
    return rows


# Alias kept for backward compat with the previous VSI-only entry point.
load_vsi_rows = load_scored_rows


def compute_official_vsi(rows: list[dict]):
    """Return (per_cat_mean_pct, overall_pct, per_cat_n, per_subcat_mean_pct,
    per_subcat_n, sample_wtd_pct). Sample-weighted is computed here from
    raw rows so it stays correct even if the fix script is re-run against an
    already-fixed scores.json (i.e. it does NOT depend on `existing.overall`)."""
    merged_scores = defaultdict(list)     # merged 8 categories
    raw_scores = defaultdict(list)        # per subtype (keeps _easy/_med/_hard)

    for r in rows:
        raw_t = r["original_question_type"]
        merged_t = _merge_rel_direction(raw_t)
        s = float(r["score"])
        merged_scores[merged_t].append(s)
        raw_scores[raw_t].append(s)

    per_cat_mean = {t: sum(v) / len(v) * 100 for t, v in merged_scores.items()}
    per_cat_n = {t: len(v) for t, v in merged_scores.items()}
    per_subcat_mean = {t: sum(v) / len(v) * 100 for t, v in raw_scores.items()}
    per_subcat_n = {t: len(v) for t, v in raw_scores.items()}

    valid_means = [per_cat_mean[c] for c in VSI_CATS if c in per_cat_mean]
    overall = sum(valid_means) / len(valid_means) if valid_means else 0.0

    all_scores = [s for lst in merged_scores.values() for s in lst]
    sample_wtd = (sum(all_scores) / len(all_scores) * 100) if all_scores else 0.0
    return per_cat_mean, overall, per_cat_n, per_subcat_mean, per_subcat_n, sample_wtd


def compute_official_spar(rows: list[dict]):
    """Return (per_type_mean_pct, overall_pct, per_type_n, group_stats, sample_wtd_pct).

    group_stats: dict of {"Low"/"Medium"/"High": {"n": int, "mean_pct": float}}
    computed as sample-weighted within each group (matches the display in
    cal_metric_sparbench.py). The overall is the arithmetic mean of the 20
    per-type means (equal weight per type), NOT weighted by group size.
    """
    per_type = defaultdict(list)
    for r in rows:
        per_type[r["original_question_type"]].append(float(r["score"]))

    per_type_mean = {t: sum(v) / len(v) * 100 for t, v in per_type.items()}
    per_type_n = {t: len(v) for t, v in per_type.items()}

    # Only types listed in Low/Med/High are counted for overall (matches
    # cal_metric_sparbench.py which builds `all_type_scores` inside those groups).
    included_types = [t for t in per_type_mean if _spar_group(t) != "Other"]
    included_means = [per_type_mean[t] for t in included_types]
    overall = sum(included_means) / len(included_means) if included_means else 0.0

    all_scores = [s for lst in per_type.values() for s in lst]
    sample_wtd = (sum(all_scores) / len(all_scores) * 100) if all_scores else 0.0

    group_stats = {}
    for group_name, type_set in SPAR_GROUPS.items():
        merged = [v for t, vs in per_type.items() if t in type_set for v in vs]
        n = len(merged)
        group_stats[group_name] = {
            "n": n,
            "mean_pct": round(sum(merged) / n * 100, 4) if n else 0.0,
        }
    return per_type_mean, overall, per_type_n, group_stats, sample_wtd


def rebuild_spar_block(rows, existing_block: dict) -> dict:
    per_type, overall, per_type_n, group_stats, sample_wtd = compute_official_spar(rows)
    new_block = {
        "available": True,
        "overall": round(overall, 4),
        "overall_method": "SPAR-Bench official: mean of 20 category scores (Low+Medium+High groups, equal weight per type)",
        "overall_sample_weighted": round(sample_wtd, 4),
        "num_scored": len(rows),
        "num_shards": existing_block.get("num_shards", 4),
        "num_categories_scored": len([t for t in per_type if _spar_group(t) != "Other"]),
        "group_stats": {
            g: {"count": s["n"], "sample_weighted_mean": s["mean_pct"]}
            for g, s in group_stats.items()
        },
    }
    # Emit categories in canonical display order (Low -> Medium -> High).
    for t in SPAR_CATS_ORDER:
        if t in per_type:
            new_block[t] = round(per_type[t], 4)
            new_block[f"{t}__n"] = per_type_n[t]
    # Any leftover (shouldn't happen but be safe).
    for t, m in sorted(per_type.items()):
        if t not in SPAR_CATS_ORDER:
            new_block[t] = round(m, 4)
            new_block[f"{t}__n"] = per_type_n[t]
    return new_block


def rebuild_vsi_block(rows, existing_block: dict) -> dict:
    per_cat, overall, per_cat_n, per_sub, per_sub_n, sample_wtd = compute_official_vsi(rows)

    # Build the new block: bookkeeping first, then per-category means (both
    # merged 8 and the raw _easy/_medium/_hard split for transparency).
    new_block: dict = {
        "available": True,
        "overall": round(overall, 4),
        "overall_method": "VSI-Bench official: mean of 8 category scores (rel_direction merged from easy+medium+hard, sample-weighted internal)",
        "overall_sample_weighted": round(sample_wtd, 4),
        "num_scored": len(rows),
        "num_shards": existing_block.get("num_shards", 4),
        "num_categories_scored": len([c for c in VSI_CATS if c in per_cat]),
    }
    # 8 merged categories, ordered
    for c in VSI_CATS:
        if c in per_cat:
            new_block[c] = round(per_cat[c], 4)
            new_block[f"{c}__n"] = per_cat_n[c]

    # Original per-subtype breakdown (including _easy/_medium/_hard split)
    # kept under a nested dict so downstream tools that read raw keys still
    # find them if needed.
    new_block["per_subtype"] = {
        t: {"score": round(v, 4), "n": per_sub_n[t]}
        for t, v in sorted(per_sub.items())
    }
    return new_block


def update_scores_json(scores_path: Path) -> dict | None:
    """Rewrite vsi_bench and (if present) spar_bench sections using the
    official method; return summary of changes."""
    with open(scores_path) as f:
        d = json.load(f)

    summary: dict = {"scores_json": str(scores_path), "changed": []}

    # VSI
    vsi_scores = scores_path.parent / "VSI-Bench" / "scores.jsonl"
    if vsi_scores.exists():
        rows = load_scored_rows(vsi_scores)
        existing = d.get("benchmarks", {}).get("vsi_bench", {}) or {}
        new_vsi = rebuild_vsi_block(rows, existing)
        d.setdefault("benchmarks", {})["vsi_bench"] = new_vsi
        summary["changed"].append({
            "bench": "vsi_bench",
            "old_overall": existing.get("overall"),
            "new_overall": new_vsi["overall"],
        })

    # SPAR
    spar_scores = scores_path.parent / "SPAR-Bench" / "scores.jsonl"
    if spar_scores.exists():
        rows = load_scored_rows(spar_scores)
        existing = d.get("benchmarks", {}).get("spar_bench", {}) or {}
        new_spar = rebuild_spar_block(rows, existing)
        d.setdefault("benchmarks", {})["spar_bench"] = new_spar
        summary["changed"].append({
            "bench": "spar_bench",
            "old_overall": existing.get("overall"),
            "new_overall": new_spar["overall"],
        })

    if not summary["changed"]:
        return None

    # Recompute top-level overall_mean if it exists (mean over available bench
    # overalls).
    if "overall_mean" in d:
        vals = [
            v["overall"] for k, v in d["benchmarks"].items()
            if isinstance(v, dict) and v.get("available") and isinstance(v.get("overall"), (int, float))
        ]
        d["overall_mean"] = round(sum(vals) / len(vals), 2) if vals else None
        d["num_benchmarks_evaluated"] = len(vals)
    summary["new_overall_mean"] = d.get("overall_mean")

    with open(scores_path, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    return summary


def find_scores_json_files(base: Path):
    return sorted(base.glob("selfevolve_step*/scores.json"))


def main():
    summaries = []
    for base_str in TARGETS:
        base = Path(base_str)
        if not base.exists():
            print(f"[skip] {base}: directory not found")
            continue
        for sj in find_scores_json_files(base):
            info = update_scores_json(sj)
            if info is None:
                print(f"[skip] {sj}: no VSI-Bench/scores.jsonl or SPAR-Bench/scores.jsonl found")
                continue
            summaries.append(info)
            print(f"[ok]   {info['scores_json']}")
            for c in info["changed"]:
                old = c["old_overall"]
                delta = (c["new_overall"] - old) if isinstance(old, (int, float)) else None
                print(f"         {c['bench']:<10}  old={old}  new={c['new_overall']}  "
                      f"Δ={delta if delta is None else round(delta, 4)}")
            print(f"         new overall_mean = {info.get('new_overall_mean')}")

    print("\n=== SUMMARY ===")
    print(f"{'file':<80} {'bench':<12} {'old':>8} {'new':>8} {'delta':>8}")
    for s in summaries:
        short = s["scores_json"].replace(
            "/data/pengtianshuo/self_evolve/Xinran/benchmark_result/", ""
        )
        for c in s["changed"]:
            old_v = c["old_overall"]
            old_s = f"{old_v:.2f}" if isinstance(old_v, (int, float)) else "   -   "
            delta = c["new_overall"] - old_v if isinstance(old_v, (int, float)) else None
            delta_s = f"{delta:+.2f}" if isinstance(delta, (int, float)) else "   -   "
            print(f"{short:<80} {c['bench']:<12} {old_s:>8} {c['new_overall']:>8.2f} {delta_s:>8}")


if __name__ == "__main__":
    main()
