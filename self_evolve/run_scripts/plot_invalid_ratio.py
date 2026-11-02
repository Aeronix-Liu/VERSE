#!/usr/bin/env python3
"""Plot invalid_ratio vs step for the full 200-step imitate-self-evolve run."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OLD_LOG = "/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr_self_evolve_imitate_porposer/experiment_log.jsonl"
NEW_LOG = "/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr_self_evolve_imitate_porposer_step200/experiment_log.jsonl"
OUTDIR = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/training_curve/2026-08-06_imitate_step200"
RESTART_STEP = 100

os.makedirs(OUTDIR, exist_ok=True)


def load_log(path, step_offset=0):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if step_offset:
                    r = dict(r)
                    r["step"] = r["step"] + step_offset
                records.append(r)
    return records


all_records = load_log(OLD_LOG, 0) + load_log(NEW_LOG, RESTART_STEP)
steps = [r["step"] for r in all_records]

# Extract invalid_ratio and task type per step
invalid = []
task_types = []
for r in all_records:
    invalid.append(r.get("round2", {}).get("invalid_ratio", float('nan')))
    sc = r.get("round1_scheduler", {}).get("selected_count", {})
    task_types.append(list(sc.keys())[0] if sc else "unknown")
invalid = np.array(invalid, dtype=float)
task_types = np.array(task_types)


def moving_avg(vals, window=10):
    vals = np.array(vals, dtype=float)
    out = np.full_like(vals, np.nan)
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        seg = vals[lo:i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 0:
            out[i] = seg.mean()
    return out


def family(tt):
    if tt == "maze_navigation":
        return "Maze", "#E91E63"
    if tt.startswith("spar_"):
        return "SPAR", "#FF9800"
    return "VSI", "#2196F3"


# ---- Single focused plot ----
fig, ax = plt.subplots(figsize=(14, 6))

# Color scatter by task family
colors = [family(tt)[1] for tt in task_types]
is_maze = np.array([tt == "maze_navigation" for tt in task_types])

ax.scatter(np.array(steps)[~is_maze], invalid[~is_maze],
           c=[c for c, m in zip(colors, is_maze) if not m],
           s=25, alpha=0.6, edgecolors='none', zorder=2, label="_non-maze")
ax.scatter(np.array(steps)[is_maze], invalid[is_maze],
           c="#E91E63", s=25, alpha=0.6, edgecolors='none', zorder=2, marker='s',
           label="_maze")

# Overall MA line
ax.plot(steps, moving_avg(invalid, 10), linewidth=2.5, color='black',
        label="Overall (MA10)", zorder=3)

# Non-maze MA line (exclude maze steps which are always 0)
non_maze_mask = ~is_maze
nm_steps = np.array(steps)[non_maze_mask]
nm_invalid = invalid[non_maze_mask]
ax.plot(nm_steps, moving_avg(nm_invalid, 10), linewidth=2.5, color='#F44336',
        linestyle='--', label="Non-maze (MA10)", zorder=3)

# Restart line
ax.axvline(x=RESTART_STEP, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
ax.text(RESTART_STEP + 1, ax.get_ylim()[1] * 0.97, ' restart',
        fontsize=9, color='red', va='top', alpha=0.8)

# Legend
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor="#2196F3", markersize=8, label='VSI (per step)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor="#FF9800", markersize=8, label='SPAR (per step)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor="#E91E63", markersize=8, label='Maze (per step, always 0)'),
    Line2D([0], [0], color='black', linewidth=2.5, label='Overall (MA10)'),
    Line2D([0], [0], color='#F44336', linewidth=2.5, linestyle='--', label='Non-maze only (MA10)'),
]
ax.legend(handles=legend_elems, fontsize=10, loc='upper right')

ax.set_title("Round 2 Invalid Ratio over Training (200 steps)", fontsize=14, fontweight="bold")
ax.set_xlabel("Overall Step", fontsize=12)
ax.set_ylabel("Invalid Ratio", fontsize=12)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

# Annotate mean values
nm_mean_1st = np.nanmean(invalid[:RESTART_STEP][is_maze[:RESTART_STEP] == False]) if (~is_maze[:RESTART_STEP]).any() else float('nan')
nm_mean_2nd = np.nanmean(invalid[RESTART_STEP:][is_maze[RESTART_STEP:] == False]) if (~is_maze[RESTART_STEP:]).any() else float('nan')
ax.text(0.02, 0.97, f"Non-maze mean invalid:\n  Steps 1-100: {nm_mean_1st:.1%}\n  Steps 101-200: {nm_mean_2nd:.1%}",
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

fig.tight_layout()
outpath = os.path.join(OUTDIR, "invalid_ratio.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {outpath}")
print(f"Non-maze mean invalid  Steps 1-100:  {nm_mean_1st:.1%}")
print(f"Non-maze mean invalid  Steps 101-200: {nm_mean_2nd:.1%}")
