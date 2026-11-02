#!/usr/bin/env python3
"""Plot training curves for the full 200-step imitate-self-evolve run.

Combines:
  - Old run (steps 1-100): checkpoints/vilasr_self_evolve_imitate_porposer/experiment_log.jsonl
  - New run (steps 1-100 -> overall 101-200): checkpoints/vilasr_self_evolve_imitate_porposer_step200/experiment_log.jsonl

Adds a vertical line at step 100 to mark the restart point.
Output: training_curve/2026-08-06_imitate_step200/
"""
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


old_records = load_log(OLD_LOG, step_offset=0)
new_records = load_log(NEW_LOG, step_offset=RESTART_STEP)
all_records = old_records + new_records
steps = [r["step"] for r in all_records]
print(f"Old: {len(old_records)} steps ({old_records[0]['step']}-{old_records[-1]['step']})")
print(f"New: {len(new_records)} steps ({new_records[0]['step']}-{new_records[-1]['step']})")
print(f"Combined: {len(all_records)} steps ({steps[0]}-{steps[-1]})")


def extract(records, key_path):
    vals = []
    for r in records:
        v = r
        for k in key_path:
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                v = None
                break
        vals.append(v if v is not None else float('nan'))
    return np.array(vals, dtype=float)


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


def add_restart_line(ax):
    ax.axvline(x=RESTART_STEP, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ax.text(RESTART_STEP + 1, ymax - (ymax - ymin) * 0.05, ' restart',
            fontsize=8, color='red', va='top', alpha=0.7)


# ---- Plot 1: Reward & Loss (2x2) ----
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

ax = axes[0, 0]
for metric, label, color in [
    (["reward_round2", "overall"], "Overall", "#2196F3"),
    (["reward_round2", "accuracy"], "Accuracy", "#4CAF50"),
    (["reward_round2", "format"], "Format", "#9C27B0"),
    (["reward_round2", "diversity", "uniq_ge2_ratio"], "Diversity", "#F44336"),
]:
    vals = extract(all_records, metric)
    ax.plot(steps, vals, marker='o', markersize=2.5, linewidth=1, label=label, color=color, alpha=0.4)
    ax.plot(steps, moving_avg(vals, 10), linewidth=2, color=color, label=f"{label} (MA10)")
ax.set_title("Round 2 Reward (Solver)", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.set_ylabel("Reward")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8); add_restart_line(ax)

ax = axes[0, 1]
for metric, label, color in [
    (["reward_round1", "quality"], "Quality", "#2196F3"),
    (["reward_round1", "quality_format"], "Quality Format", "#FF9800"),
]:
    vals = extract(all_records, metric)
    ax.plot(steps, vals, marker='o', markersize=2.5, linewidth=1, label=label, color=color, alpha=0.4)
    ax.plot(steps, moving_avg(vals, 10), linewidth=2, color=color, label=f"{label} (MA10)")
ax.set_title("Round 1 Reward (Proposer)", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.set_ylabel("Reward")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8); add_restart_line(ax)

ax = axes[1, 0]
for metric, label, color, ls in [
    (["actor", "pg_loss_round1"], "PG Loss R1", "#2196F3", "-"),
    (["actor", "pg_loss_round2"], "PG Loss R2", "#FF9800", "-"),
    (["actor", "entropy_loss_round1"], "Entropy R1", "#4CAF50", "--"),
    (["actor", "entropy_loss_round2"], "Entropy R2", "#F44336", "--"),
]:
    vals = extract(all_records, metric)
    ax.plot(steps, vals, marker='o', markersize=2.5, linewidth=1, label=label, color=color, linestyle=ls, alpha=0.4)
    ax.plot(steps, moving_avg(vals, 10), linewidth=2, color=color, linestyle=ls, label=f"{label} (MA10)")
ax.set_title("Actor Loss", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.set_ylabel("Loss")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8); add_restart_line(ax)

ax = axes[1, 1]
for metric, label, color, ls in [
    (["actor", "kl_loss_round1"], "KL Loss R1", "#2196F3", "-"),
    (["actor", "kl_loss_round2"], "KL Loss R2", "#FF9800", "-"),
    (["actor", "grad_norm_round1"], "Grad Norm R1", "#4CAF50", "--"),
    (["actor", "grad_norm_round2"], "Grad Norm R2", "#F44336", "--"),
]:
    vals = extract(all_records, metric)
    ax.plot(steps, vals, marker='o', markersize=2.5, linewidth=1, label=label, color=color, linestyle=ls, alpha=0.4)
    ax.plot(steps, moving_avg(vals, 10), linewidth=2, color=color, linestyle=ls, label=f"{label} (MA10)")
ax.set_title("KL Loss & Grad Norm", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.set_ylabel("Value")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8); add_restart_line(ax)

fig.suptitle("Self-Evolve Training Curves — 200 Steps (imitate_proposer)\n"
             "Steps 1-100: base=rrs-744 | Steps 101-200: base=step-100 (restart at red line)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "training_reward_loss.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: training_reward_loss.png")

# ---- Plot 2: Critic & Quality (1x3) ----
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

ax = axes[0]
for metric, label, color in [
    (["critic", "score", "mean"], "Critic Score", "#2196F3"),
    (["critic", "rewards", "mean"], "Critic Rewards", "#FF9800"),
    (["critic", "advantages", "mean"], "Advantages", "#F44336"),
]:
    vals = extract(all_records, metric)
    ax.plot(steps, vals, marker='o', markersize=2.5, linewidth=1, label=label, color=color, alpha=0.4)
    ax.plot(steps, moving_avg(vals, 10), linewidth=2, color=color, label=f"{label} (MA10)")
ax.set_title("Critic Metrics", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8); add_restart_line(ax)

ax = axes[1]
for metric, label, color in [
    (["response_length", "mean"], "Response Mean", "#2196F3"),
    (["response_length", "max"], "Response Max", "#FF9800"),
    (["response_length", "min"], "Response Min", "#4CAF50"),
]:
    vals = extract(all_records, metric)
    ax.plot(steps, vals, marker='o', markersize=2.5, linewidth=1, label=label, color=color, alpha=0.4)
    ax.plot(steps, moving_avg(vals, 10), linewidth=2, color=color, label=f"{label} (MA10)")
ax.set_title("Response Length", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.set_ylabel("Tokens")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8); add_restart_line(ax)

ax = axes[2]
for metric, label, color in [
    (["round2", "invalid_ratio"], "Invalid Ratio", "#F44336"),
    (["round2", "dedup_removed_question_ratio"], "Dedup Removed", "#FF9800"),
]:
    vals = extract(all_records, metric)
    ax.plot(steps, vals, marker='o', markersize=2.5, linewidth=1, label=label, color=color, alpha=0.4)
    ax.plot(steps, moving_avg(vals, 10), linewidth=2, color=color, label=f"{label} (MA10)")
ax.set_title("Round 2 Quality Metrics", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.set_ylabel("Ratio")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8); add_restart_line(ax)

fig.suptitle("Self-Evolve Training Curves — Critic & Quality (200 steps)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "training_critic_quality.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: training_critic_quality.png")

# ---- Plot 3: Round 1 vs Round 2 reward (smoothed) ----
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

ax = axes[0]
vals = extract(all_records, ["reward_round2", "overall"])
ax.plot(steps, vals, marker='o', markersize=2.5, linewidth=0.8, color="#2196F3", alpha=0.3, label="R2 Overall (raw)")
ax.plot(steps, moving_avg(vals, 10), linewidth=2.5, color="#2196F3", label="R2 Overall (MA10)")
vals = extract(all_records, ["reward_round2", "accuracy"])
ax.plot(steps, moving_avg(vals, 10), linewidth=2.5, color="#4CAF50", label="R2 Accuracy (MA10)")
ax.set_title("Round 2 Solver Reward (smoothed)", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.set_ylabel("Reward")
ax.grid(True, alpha=0.3); ax.legend(fontsize=9); add_restart_line(ax)

ax = axes[1]
vals = extract(all_records, ["reward_round1", "quality"])
ax.plot(steps, vals, marker='o', markersize=2.5, linewidth=0.8, color="#FF9800", alpha=0.3, label="R1 Quality (raw)")
ax.plot(steps, moving_avg(vals, 10), linewidth=2.5, color="#FF9800", label="R1 Quality (MA10)")
vals = extract(all_records, ["reward_round1", "quality_format"])
ax.plot(steps, moving_avg(vals, 10), linewidth=2.5, color="#9C27B0", label="R1 Format (MA10)")
ax.set_title("Round 1 Proposer Reward (smoothed)", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.set_ylabel("Reward")
ax.grid(True, alpha=0.3); ax.legend(fontsize=9); add_restart_line(ax)

fig.suptitle("Self-Evolve — Round 1 vs Round 2 Reward (200 steps)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "round_rewards.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: round_rewards.png")

# ---- Plot 4: Per-task-family R2 reward ----
fig, ax = plt.subplots(figsize=(16, 6))
task_types = []
for r in all_records:
    sc = r.get("round1_scheduler", {}).get("selected_count", {})
    task_types.append(list(sc.keys())[0] if sc else "unknown")

r2_overall = extract(all_records, ["reward_round2", "overall"])


def family(tt):
    if tt == "maze_navigation":
        return "Maze", "#E91E63"
    if tt.startswith("spar_"):
        return "SPAR", "#FF9800"
    if tt in ("object_counting", "absolute_distance", "object_size", "relative_distance",
              "relative_direction_hard", "room_size", "route_planning",
              "single_image_relative_direction", "distance_cam_obj",
              "depth_order_obj_obj"):
        return "VSI", "#2196F3"
    return "Other", "#9E9E9E"


colors_pts = [family(tt)[1] for tt in task_types]
ax.scatter(steps, r2_overall, c=colors_pts, s=20, alpha=0.6, edgecolors='none')
ax.plot(steps, moving_avg(r2_overall, 10), linewidth=2.5, color='black', label="Overall (MA10)")
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor="#E91E63", markersize=8, label='Maze'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor="#FF9800", markersize=8, label='SPAR'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor="#2196F3", markersize=8, label='VSI'),
    Line2D([0], [0], color='black', linewidth=2, label='Overall (MA10)'),
]
ax.legend(handles=legend_elems, fontsize=9, loc='upper right')
ax.set_title("Round 2 Reward by Task Family (200 steps)", fontsize=13, fontweight="bold")
ax.set_xlabel("Overall Step"); ax.set_ylabel("R2 Overall Reward")
ax.grid(True, alpha=0.3); add_restart_line(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "reward_by_task_family.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: reward_by_task_family.png")

# ---- Summary table ----
print("\n=== Key Metrics Summary (every 10 steps) ===")
hdr = f"{'Step':>5s} {'R2_Ovl':>7s} {'R2_Acc':>7s} {'R2_Fmt':>7s} {'R1_Q':>7s} {'KL_R2':>7s} {'GrR2':>7s} {'Inv%':>6s} {'RespLen':>7s} {'TaskType'}"
print(hdr)
for r in all_records:
    s = r["step"]
    if s % 10 != 0 and s != 1 and s != 200:
        continue
    r2 = r.get("reward_round2", {})
    a = r.get("actor", {})
    sc = r.get("round1_scheduler", {}).get("selected_count", {})
    tt = list(sc.keys())[0] if sc else "?"
    print(f"{s:>5d} {r2.get('overall',0):>7.3f} {r2.get('accuracy',0):>7.3f} "
          f"{r2.get('format',0):>7.3f} {r.get('reward_round1',{}).get('quality',0):>7.3f} "
          f"{a.get('kl_loss_round2',0):>7.4f} {a.get('grad_norm_round2',0):>7.3f} "
          f"{r.get('round2',{}).get('invalid_ratio',0)*100:>5.1f} "
          f"{r.get('response_length',{}).get('mean',0):>7.0f} {tt}")

print(f"\nAll plots saved to: {OUTDIR}")
