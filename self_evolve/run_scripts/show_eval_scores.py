#!/usr/bin/env python3
"""Summarize all eval results for vilasr_self_evolve_immitate_porposer.

Usage:
    python3 show_eval_scores.py              # all steps, all benchmarks
    python3 show_eval_scores.py 100           # only step 100
    python3 show_eval_scores.py 100 75       # steps 100 and 75
"""
import json, os, sys
from collections import defaultdict

ROOT = "/data/pengtianshuo/self_evolve/Xinran/benchmark_result/vilasr_self_evolve_immitate_porposer"
BENCHES = ["VSI-Bench", "SPAR-Bench", "MAZE", "SpatialEval-Real", "MMSI-Bench"]

steps = sys.argv[1:] if len(sys.argv) > 1 else sorted(
    d.replace("selfevolve_step", "")
    for d in os.listdir(ROOT)
    if d.startswith("selfevolve_step")
)

print(f"{'Step':>6s}  {'Benchmark':20s}  {'Avg':>6s}  {'N':>5s}")
print("-" * 45)

for step in sorted(steps, key=lambda x: int(x)):
    row = {}
    for bench in BENCHES:
        f = os.path.join(ROOT, f"selfevolve_step{step}", bench, "scores.jsonl")
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            row[bench] = None
            continue
        scores = []
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    scores.append(d.get("score", 0))
        if scores:
            row[bench] = sum(scores) / len(scores)
        else:
            row[bench] = None

    for bench in BENCHES:
        v = row[bench]
        if v is not None:
            n_file = os.path.join(ROOT, f"selfevolve_step{step}", bench, "results.jsonl")
            n = sum(1 for _ in open(n_file)) if os.path.exists(n_file) else 0
            print(f"{step:>6s}  {bench:20s}  {v:6.4f}  {n:5d}")
        else:
            print(f"{step:>6s}  {bench:20s}  {'--':>6s}  {'--':>5s}")
    print()
