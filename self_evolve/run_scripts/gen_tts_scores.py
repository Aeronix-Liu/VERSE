#!/usr/bin/env python3
"""Generate scores.json for the TTS-search run, matching the format of
benchmark_result/vilasr-self-evolve-imitate-0811-no-spar/selfevolve_step200/scores.json.

Reuses the official VSI (8-cat mean) / SPAR (20-type mean) aggregators from
gen_imitate0811_scores.py. MAZE stays sample-weighted (official for MAZE).

Usage:
    python3 gen_tts_scores.py [OUT_ROOT] [label1 label2 ...]
Defaults: OUT_ROOT = benchmark_result/vilasr-se-tts-search-50step, labels = selfevolve_step50
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_imitate0811_scores import BENCHES, compute_bench, load_scores  # noqa: E402

DEFAULT_ROOT = "/data/pengtianshuo/self_evolve/Xinran/benchmark_result/vilasr-se-tts-search-50step"


def main():
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_ROOT)
    labels = sys.argv[2:] if len(sys.argv) > 2 else ["selfevolve_step50"]
    for label in labels:
        model_dir = out_root / label
        if not model_dir.exists():
            print(f"skip {label}: dir not found ({model_dir})")
            continue
        out = {"checkpoint": label, "benchmarks": {}}
        for bkey, bfolder, tf in BENCHES:
            sj = model_dir / bfolder / "scores.jsonl"
            out["benchmarks"][bkey] = compute_bench(load_scores(sj), tf, bkey) if sj.exists() else {"available": False}
        ovs = [v["overall"] for v in out["benchmarks"].values()
               if v.get("available") and v.get("overall") is not None]
        out["overall_mean"] = round(sum(ovs) / len(ovs), 2) if ovs else None
        out["num_benchmarks_evaluated"] = len(ovs)

        outpath = model_dir / "scores.json"
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"=== {label} ===")
        for bkey, v in out["benchmarks"].items():
            if v.get("available"):
                print(f"  {bkey:12s} {v['overall']:6.2f}  (n={v['num_scored']})")
            else:
                print(f"  {bkey:12s} NOT AVAILABLE")
        print(f"  overall_mean: {out['overall_mean']}  ({out['num_benchmarks_evaluated']}/3)")
        print(f"  saved: {outpath}\n")


if __name__ == "__main__":
    main()
