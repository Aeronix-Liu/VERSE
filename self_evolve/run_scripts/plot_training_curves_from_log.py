#!/usr/bin/env python3
"""Plot training curves from experiment_log.jsonl: reward, loss, kl, grad_norm, etc."""
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

steps = [r["step"] for r in records]
print(f"Loaded {len(records)} steps (step {steps[0]} to {steps[-1]})")

# Extract metrics
def extract(key_path):
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

# ---- Plot 1: Reward curves ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Round 2 reward (main reward)
ax = axes[0, 0]
for metric, label, color in [
    (["reward_round2", "overall"], "Overall", "#2196F3"),
    (["reward_round2", "accuracy"], "Accuracy", "#4CAF50"),
    (["reward_round2", "answer_correctness"], "Answer Correctness", "#FF9800"),
    (["reward_round2", "format"], "Format", "#9C27B0"),
    (["reward_round2", "diversity", "uniq_ge2_ratio"], "Diversity", "#F44336"),
]:
    vals = extract(metric)
    ax.plot(steps, vals, marker='o', markersize=3, linewidth=1.5, label=label, color=color)
ax.set_title("Round 2 Reward (Solver)", fontsize=13, fontweight="bold")
ax.set_xlabel("Step")
ax.set_ylabel("Reward")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Round 1 reward (proposer)
ax = axes[0, 1]
for metric, label, color in [
    (["reward_round1", "quality"], "Quality", "#2196F3"),
    (["reward_round1", "quality_accuracy"], "Quality Accuracy", "#4CAF50"),
    (["reward_round1", "quality_format"], "Quality Format", "#FF9800"),
    (["reward_round1", "quality_question_quality"], "Question Quality", "#9C27B0"),
]:
    vals = extract(metric)
    ax.plot(steps, vals, marker='o', markersize=3, linewidth=1.5, label=label, color=color)
ax.set_title("Round 1 Reward (Proposer)", fontsize=13, fontweight="bold")
ax.set_xlabel("Step")
ax.set_ylabel("Reward")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Actor loss
ax = axes[1, 0]
for metric, label, color in [
    (["actor", "pg_loss_round1"], "PG Loss R1", "#2196F3"),
    (["actor", "pg_loss_round2"], "PG Loss R2", "#FF9800"),
]:
    vals = extract(metric)
    ax.plot(steps, vals, marker='o', markersize=3, linewidth=1.5, label=label, color=color)
# Also entropy loss
for metric, label, color in [
    (["actor", "entropy_loss_round1"], "Entropy R1", "#4CAF50"),
    (["actor", "entropy_loss_round2"], "Entropy R2", "#F44336"),
]:
    vals = extract(metric)
    ax.plot(steps, vals, marker='s', markersize=3, linewidth=1.5, label=label, color=color, linestyle='--')
ax.set_title("Actor Loss", fontsize=13, fontweight="bold")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# KL loss & grad norm
ax = axes[1, 1]
for metric, label, color, ls in [
    (["actor", "kl_loss_round1"], "KL Loss R1", "#2196F3", "-"),
    (["actor", "kl_loss_round2"], "KL Loss R2", "#FF9800", "-"),
    (["actor", "grad_norm_round1"], "Grad Norm R1", "#4CAF50", "--"),
    (["actor", "grad_norm_round2"], "Grad Norm R2", "#F44336", "--"),
]:
    vals = extract(metric)
    ax.plot(steps, vals, marker='o', markersize=3, linewidth=1.5, label=label, color=color, linestyle=ls)
ax.set_title("KL Loss & Grad Norm", fontsize=13, fontweight="bold")
ax.set_xlabel("Step")
ax.set_ylabel("Value")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

fig.suptitle("Self-Evolve Training Curves (vilasr_self_evolve_immitate_porposer)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "training_reward_loss.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- Plot 2: Critic & response length ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Critic scores
ax = axes[0]
for metric, label, color in [
    (["critic", "score", "mean"], "Critic Score", "#2196F3"),
    (["critic", "rewards", "mean"], "Critic Rewards", "#FF9800"),
    (["critic", "returns", "mean"], "Critic Returns", "#4CAF50"),
    (["critic", "advantages", "mean"], "Advantages", "#F44336"),
]:
    vals = extract(metric)
    ax.plot(steps, vals, marker='o', markersize=3, linewidth=1.5, label=label, color=color)
ax.set_title("Critic Metrics", fontsize=13, fontweight="bold")
ax.set_xlabel("Step")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Response length
ax = axes[1]
for metric, label, color in [
    (["response_length", "mean"], "Response Mean", "#2196F3"),
    (["response_length", "max"], "Response Max", "#FF9800"),
    (["response_length", "min"], "Response Min", "#4CAF50"),
]:
    vals = extract(metric)
    ax.plot(steps, vals, marker='o', markersize=3, linewidth=1.5, label=label, color=color)
ax.set_title("Response Length", fontsize=13, fontweight="bold")
ax.set_xlabel("Step")
ax.set_ylabel("Tokens")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Round 2 invalid ratio
ax = axes[2]
for metric, label, color in [
    (["round2", "invalid_ratio"], "Invalid Ratio", "#F44336"),
    (["round2", "dedup_removed_question_ratio"], "Dedup Removed Ratio", "#FF9800"),
]:
    vals = extract(metric)
    ax.plot(steps, vals, marker='o', markersize=3, linewidth=1.5, label=label, color=color)
ax.set_title("Round 2 Quality Metrics", fontsize=13, fontweight="bold")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

fig.suptitle("Self-Evolve Training Curves — Critic & Quality", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "training_critic_quality.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- Print summary ----
print("\n=== Key Metrics Summary ===")
print(f"{'Step':>5s}  {'R2_Overall':>10s}  {'R2_Acc':>8s}  {'R1_Quality':>10s}  {'KL_R2':>8s}  {'GradR2':>8s}  {'RespLen':>8s}")
for i, r in enumerate(records):
    s = r["step"]
    r2o = r.get("reward_round2", {}).get("overall", float('nan'))
    r2a = r.get("reward_round2", {}).get("accuracy", float('nan'))
    r1q = r.get("reward_round1", {}).get("quality", float('nan'))
    kl = r.get("actor", {}).get("kl_loss_round2", float('nan'))
    gn = r.get("actor", {}).get("grad_norm_round2", float('nan'))
    rl = r.get("response_length", {}).get("mean", float('nan'))
    print(f"{s:>5d}  {r2o:>10.4f}  {r2a:>8.4f}  {r1q:>10.4f}  {kl:>8.4f}  {gn:>8.4f}  {rl:>8.1f}")

print(f"\nSaved to: {OUTDIR}")
print(f"  - training_reward_loss.png")
print(f"  - training_critic_quality.png")
