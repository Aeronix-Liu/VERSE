#!/usr/bin/env python3
"""Plot training curve: eval scores at different checkpoints across 5 benchmarks."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/data/pengtianshuo/self_evolve/Xinran/benchmark_result/vilasr_self_evolve_immitate_porposer"
BENCHES = ["VSI-Bench", "SPAR-Bench", "MAZE", "SpatialEval-Real", "MMSI-Bench"]
BENCH_LABELS = ["VSI-Bench", "SPAR-Bench", "MAZE", "SpatialEval-Real", "MMSI-Bench"]
BENCH_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
BENCH_MARKERS = ["o", "s", "^", "D", "v"]

OUTDIR = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/training_curve/2026-08-03_full_se"

# Collect data
steps = sorted(
    int(d.replace("selfevolve_step", ""))
    for d in os.listdir(ROOT)
    if d.startswith("selfevolve_step")
)

data = {bench: {"steps": [], "scores": [], "counts": []} for bench in BENCHES}

for step in steps:
    for bench in BENCHES:
        f = os.path.join(ROOT, f"selfevolve_step{step}", bench, "scores.jsonl")
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        scores = []
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    scores.append(d.get("score", 0))
        if scores:
            data[bench]["steps"].append(step)
            data[bench]["scores"].append(sum(scores) / len(scores))
            data[bench]["counts"].append(len(scores))

# ---- Plot 1: All benchmarks on one chart ----
fig, ax = plt.subplots(figsize=(10, 6))
for i, bench in enumerate(BENCHES):
    if data[bench]["steps"]:
        ax.plot(data[bench]["steps"], [s * 100 for s in data[bench]["scores"]],
                marker=BENCH_MARKERS[i], color=BENCH_COLORS[i], linewidth=2,
                markersize=8, label=bench)

ax.set_xlabel("Training Step", fontsize=13)
ax.set_ylabel("Score (%)", fontsize=13)
ax.set_title("Self-Evolve Training Curve (vilasr_self_evolve_immitate_porposer)", fontsize=14)
ax.set_xticks(steps)
ax.set_ylim(0, 100)
ax.set_xlim(15, 110)
ax.grid(True, alpha=0.3)
ax.legend(loc="center right", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "training_curve_all.png"), dpi=150)
plt.close()

# ---- Plot 2: Separate subplots ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, bench in enumerate(BENCHES):
    ax = axes[i]
    if data[bench]["steps"]:
        ax.plot(data[bench]["steps"], [s * 100 for s in data[bench]["scores"]],
                marker=BENCH_MARKERS[i], color=BENCH_COLORS[i], linewidth=2,
                markersize=8)
        ax.set_title(bench, fontsize=12, fontweight="bold")
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_xticks(steps)
        ax.set_ylim(0, max(100, max(s * 100 for s in data[bench]["scores"]) * 1.15))
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title(f"{bench} (pending)", fontsize=12, color="gray")
        ax.set_xticks(steps)
        ax.grid(True, alpha=0.2)

# Hide the 6th subplot
axes[5].axis("off")

# Add summary table in the 6th subplot area
col_labels = ["Step"] + BENCHES
row_data = []
for step in steps:
    row = [step]
    for bench in BENCHES:
        if step in data[bench]["steps"]:
            idx = data[bench]["steps"].index(step)
            row.append(f"{data[bench]['scores'][idx]*100:.1f}")
        else:
            row.append("--")
    row_data.append(row)

table = axes[5].table(cellText=row_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

fig.suptitle("Self-Evolve Training Curve — Per Benchmark", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "training_curve_subplots.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- Print summary ----
print("=== Training Curve Data ===")
print(f"{'Step':>6s}  " + "  ".join(f"{b:>14s}" for b in BENCHES))
for step in steps:
    row = f"{step:>6d}  "
    for bench in BENCHES:
        if step in data[bench]["steps"]:
            idx = data[bench]["steps"].index(step)
            row += f"  {data[bench]['scores'][idx]*100:>13.2f}%"
        else:
            row += f"  {'--':>14s}"
    print(row)

print(f"\nSaved to: {OUTDIR}")
print(f"  - training_curve_all.png")
print(f"  - training_curve_subplots.png")
