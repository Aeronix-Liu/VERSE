#!/usr/bin/env python3
"""Plot Round-2 Solver reward with smoothed fit, matching mix_training style."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOGFILE = "/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr_self_evolve_immitate_porposer/experiment_log.jsonl"
OUTDIR = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/training_curve/2026-08-03_full_se"

os.makedirs(OUTDIR, exist_ok=True)

# Load all steps
records = []
with open(LOGFILE) as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

steps = np.array([r["step"] for r in records])

def get_metric(path):
    vals = []
    for r in records:
        v = r
        for k in path:
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                v = None
                break
        if isinstance(v, dict):
            v = v.get("mean", v.get("uniq_ge2_ratio", None))
        vals.append(float(v) if v is not None else float('nan'))
    return np.array(vals, dtype=float)

def smooth(vals, window=10):
    """Moving average smoothing."""
    if len(vals) < window:
        return vals
    cumsum = np.nancumsum(vals)
    result = np.empty_like(vals)
    result[:window] = cumsum[:window] / np.arange(1, window + 1)
    result[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return result

# Metrics to plot (matching mix_training/round2_reward.png)
metrics = [
    (["reward_round2", "format"], "format", "#E53935"),
    (["reward_round2", "overall"], "overall", "#FB8C00"),
    (["reward_round2", "accuracy"], "accuracy", "#8D6E63"),
    (["reward_round2", "answer_correctness"], "answer_correctness", "#757575"),
]

fig, ax = plt.subplots(figsize=(10, 6))

for path, label, color in metrics:
    raw = get_metric(path)
    smoothed = smooth(raw, window=10)
    # Raw (thin, translucent)
    ax.plot(steps, raw, color=color, alpha=0.25, linewidth=0.8)
    # Smoothed (bold)
    ax.plot(steps, smoothed, color=color, linewidth=2.5, label=label)

ax.set_xlabel("Step", fontsize=13)
ax.set_ylabel("Reward", fontsize=13)
ax.set_title("Round-2 Solver reward", fontsize=14, fontweight="bold")
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, max(steps) + 2)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=11)

fig.tight_layout()
outpath = os.path.join(OUTDIR, "round2_reward.png")
fig.savefig(outpath, dpi=150)
plt.close()
print(f"Saved: {outpath}")

# Also make Round-1 reward plot
metrics_r1 = [
    (["reward_round1", "quality_format"], "quality_format", "#E53935"),
    (["reward_round1", "quality"], "quality", "#FB8C00"),
    (["reward_round1", "quality_question_quality"], "question_quality", "#8D6E63"),
    (["reward_round1", "quality_accuracy"], "quality_accuracy", "#757575"),
]

fig, ax = plt.subplots(figsize=(10, 6))
for path, label, color in metrics_r1:
    raw = get_metric(path)
    smoothed = smooth(raw, window=10)
    ax.plot(steps, raw, color=color, alpha=0.25, linewidth=0.8)
    ax.plot(steps, smoothed, color=color, linewidth=2.5, label=label)

ax.set_xlabel("Step", fontsize=13)
ax.set_ylabel("Reward", fontsize=13)
ax.set_title("Round-1 Proposer reward", fontsize=14, fontweight="bold")
ax.set_ylim(-1.0, 1.1)
ax.set_xlim(0, max(steps) + 2)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=11)
fig.tight_layout()
outpath_r1 = os.path.join(OUTDIR, "round1_reward.png")
fig.savefig(outpath_r1, dpi=150)
plt.close()
print(f"Saved: {outpath_r1}")
