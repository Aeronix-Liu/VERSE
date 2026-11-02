#!/usr/bin/env python3
"""Plot imitate-0808 training curves for the first 100 steps.

Emits four files under `training_curve/2026-08-09_imitate0808_step100/`:
  * solver_overall_reward.png   — Round-2 overall reward (per-step + smoothed)
  * solver_invalid_ratio.png    — Round-2 invalid_ratio (per-step + smoothed)
  * overall_loss.png            — Actor pg_loss for Round-1 and Round-2 combined
  * dashboard.png               — 3x3 dashboard with reward, loss, KL, gradnorm,
                                  entropy, response_length, per-family invalid.

Only steps 1..100 are plotted (the first checkpoint we're evaluating).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = "/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr-self-evolve-imitate-0808/experiment_log.jsonl"
SCHED = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/easy_r1/dataset/full_selfevolve_schedule_v2.json"
OUT_DIR = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/training_curve/2026-08-09_imitate0808_step100"
MAX_STEP = 100

os.makedirs(OUT_DIR, exist_ok=True)


def load_rows():
    with open(LOG) as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r.get("step", 0) <= MAX_STEP]
    rows.sort(key=lambda r: r["step"])
    return rows


def load_schedule():
    with open(SCHED) as f:
        return json.load(f)


def family(task: str) -> str:
    if task == "maze_navigation":
        return "maze"
    if task.startswith("spar_"):
        return "spar"
    return "vsi"


def smooth(vals, w=5):
    """Simple moving average, edge-preserving."""
    a = np.asarray(vals, dtype=float)
    n = len(a)
    if n == 0:
        return a
    w = max(1, min(w, n))
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, i + w // 2 + 1)
        out[i] = a[lo:hi].mean()
    return out


def save_fig(fig, out_path, dpi=150, **kwargs):
    fig.savefig(out_path, dpi=dpi, **kwargs)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    return out_path, pdf_path


def extract(rows, schedule):
    L = len(schedule)
    data = {
        "step": [], "task": [], "fam": [],
        "step_time": [],
        "invalid": [],
        "r1_quality": [], "r1_format": [], "r1_valid": [],
        "r2_overall": [], "r2_accuracy": [], "r2_format": [], "r2_diversity": [],
        "pg1": [], "pg2": [],
        "kl1": [], "kl2": [],
        "ent1": [], "ent2": [],
        "gn1": [], "gn2": [],
        "resp_len_mean": [], "resp_len_max": [],
    }
    for r in rows:
        s = r["step"]
        tt = schedule[s % L]
        a = r.get("actor", {})
        data["step"].append(s)
        data["task"].append(tt)
        data["fam"].append(family(tt))
        data["step_time"].append(r["timing_s"]["step"])
        data["invalid"].append(r["round2"]["invalid_ratio"])
        r1 = r["reward_round1"]
        r2 = r["reward_round2"]
        data["r1_quality"].append(r1.get("quality"))
        data["r1_format"].append(r1.get("quality_format"))
        data["r1_valid"].append(r1.get("quality_valid_factor"))
        data["r2_overall"].append(r2.get("overall"))
        data["r2_accuracy"].append(r2.get("accuracy"))
        data["r2_format"].append(r2.get("format"))
        data["r2_diversity"].append(r2.get("diversity"))
        data["pg1"].append(a.get("pg_loss_round1"))
        data["pg2"].append(a.get("pg_loss_round2"))
        data["kl1"].append(a.get("kl_loss_round1"))
        data["kl2"].append(a.get("kl_loss_round2"))
        data["ent1"].append(a.get("entropy_loss_round1"))
        data["ent2"].append(a.get("entropy_loss_round2"))
        data["gn1"].append(a.get("grad_norm_round1"))
        data["gn2"].append(a.get("grad_norm_round2"))
        rl = r.get("response_length") or {}
        data["resp_len_mean"].append(rl.get("mean"))
        data["resp_len_max"].append(rl.get("max"))
    return data


FAM_COLOR = {"vsi": "#1f77b4", "spar": "#ff7f0e", "maze": "#2ca02c"}


def scatter_by_family(ax, d, ykey, label=None, s=18, alpha=0.75):
    for fam in ("vsi", "spar", "maze"):
        idx = [i for i, f in enumerate(d["fam"]) if f == fam]
        if not idx:
            continue
        xs = [d["step"][i] for i in idx]
        ys = [d[ykey][i] for i in idx]
        ax.scatter(xs, ys, s=s, alpha=alpha, c=FAM_COLOR[fam],
                   label=(f"{fam.upper()}" if label is None else None))


# ---------------------- separate panels ------------------------------

def plot_solver_reward(d):
    fig, ax = plt.subplots(figsize=(11, 5))
    scatter_by_family(ax, d, "r2_overall")
    ax.plot(d["step"], smooth(d["r2_overall"], 9), color="black", lw=2,
            label="smoothed (window=9)")
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.set_title("Solver Overall Reward — imitate-0808 (steps 1–100)")
    ax.set_xlabel("step")
    ax.set_ylabel("round2 / overall")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "solver_overall_reward.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_solver_reward_by_family(d):
    """Task-specific solver overall reward — VSI / SPAR / MAZE stacked panels.

    X-axis is the per-family training step index (1st time that family was
    trained, 2nd time, …), NOT the global step. E.g. VSI at global step 6
    could be the 2nd VSI step if step 5 was maze.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharey=True)
    ymin = min([v for v in d["r2_overall"] if v is not None], default=0.0) - 0.02
    ymax = max([v for v in d["r2_overall"] if v is not None], default=1.0) + 0.02
    for ax, fam in zip(axes, ("vsi", "spar", "maze")):
        idx = [i for i, f in enumerate(d["fam"]) if f == fam]
        ys = [d["r2_overall"][i] for i in idx]
        gsteps = [d["step"][i] for i in idx]  # for annotation only
        n = len(ys)
        xs = list(range(1, n + 1))  # per-family step index (1..n)

        ax.scatter(xs, ys, s=32, color=FAM_COLOR[fam], alpha=0.85,
                   label=f"{fam.upper()} per-family step (n={n})")
        if n >= 2:
            w = min(5, n)
            ma = np.convolve(np.asarray(ys, float), np.ones(w)/w, mode="valid")
            ma_x = xs[w-1:]
            ax.plot(ma_x, ma, color="black", lw=2,
                    label=f"rolling mean (w={w})")
        if ys:
            m = float(np.mean(ys))
            ax.axhline(m, color=FAM_COLOR[fam], lw=1.2, ls="--", alpha=0.7,
                       label=f"{fam.upper()} mean = {m:.3f}")
        ax.axhline(0, color="grey", lw=0.6, ls=":", alpha=0.5)
        ax.set_ylabel(f"{fam.upper()}\noverall reward")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0.5, max(3, n) + 0.5)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        # annotate first/last with global step for reference
        if n:
            ax.annotate(f"(global step {gsteps[0]})", (xs[0], ys[0]),
                        xytext=(5, 6), textcoords="offset points", fontsize=8, color="grey")
            ax.annotate(f"(global step {gsteps[-1]})", (xs[-1], ys[-1]),
                        xytext=(-70, 6), textcoords="offset points", fontsize=8, color="grey")
    axes[-1].set_xlabel("per-family training step (1st, 2nd, ... time this family was trained)")
    fig.suptitle("Task-Specific Solver Overall Reward — imitate-0808 (100 global steps → per-family step index)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT_DIR, "solver_overall_reward_by_family.png")
    save_fig(fig, out)
    plt.close(fig)
    return out


def plot_solver_invalid(d):
    fig, ax = plt.subplots(figsize=(11, 5))
    scatter_by_family(ax, d, "invalid")
    ax.plot(d["step"], smooth(d["invalid"], 9), color="black", lw=2,
            label="smoothed (window=9)")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.set_title("Solver Invalid Ratio — imitate-0808 (steps 1–100)")
    ax.set_xlabel("step")
    ax.set_ylabel("round2 / invalid_ratio")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "solver_invalid_ratio.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_overall_loss(d):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.plot(d["step"], d["pg1"], marker=".", ms=4, lw=0.6, alpha=0.5,
            color="#1f77b4", label="R1 pg_loss (proposer)")
    ax.plot(d["step"], smooth(d["pg1"], 9), color="#1f77b4", lw=2, label="R1 pg_loss (smoothed)")
    ax.plot(d["step"], d["pg2"], marker=".", ms=4, lw=0.6, alpha=0.5,
            color="#d62728", label="R2 pg_loss (solver)")
    ax.plot(d["step"], smooth(d["pg2"], 9), color="#d62728", lw=2, label="R2 pg_loss (smoothed)")
    ax.set_title("Actor PG Loss — Round-1 (proposer) vs Round-2 (solver) — imitate-0808")
    ax.set_xlabel("step")
    ax.set_ylabel("pg_loss")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "overall_loss.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------- dashboard -----------------------------------

def plot_dashboard(d):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)

    # 1. Solver overall reward
    ax = axes[0][0]
    scatter_by_family(ax, d, "r2_overall")
    ax.plot(d["step"], smooth(d["r2_overall"], 9), color="black", lw=2)
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.set_title("Solver Overall Reward (round2/overall)")
    ax.set_xlabel("step"); ax.set_ylabel("reward")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # 2. Solver accuracy
    ax = axes[0][1]
    scatter_by_family(ax, d, "r2_accuracy")
    ax.plot(d["step"], smooth(d["r2_accuracy"], 9), color="black", lw=2)
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.set_title("Solver Accuracy (round2/accuracy)")
    ax.set_xlabel("step"); ax.set_ylabel("accuracy")
    ax.grid(alpha=0.25)

    # 3. Invalid ratio
    ax = axes[0][2]
    scatter_by_family(ax, d, "invalid")
    ax.plot(d["step"], smooth(d["invalid"], 9), color="black", lw=2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Solver Invalid Ratio (round2/invalid_ratio)")
    ax.set_xlabel("step"); ax.set_ylabel("invalid_ratio")
    ax.grid(alpha=0.25)

    # 4. Proposer quality decomposition
    ax = axes[1][0]
    ax.plot(d["step"], d["r1_quality"], color="#1f77b4", lw=1, alpha=0.5, label="r1 quality")
    ax.plot(d["step"], smooth(d["r1_quality"], 9), color="#1f77b4", lw=2, label="quality (smoothed)")
    if any(v is not None for v in d["r1_format"]):
        ax.plot(d["step"], d["r1_format"], color="#2ca02c", lw=1, alpha=0.4, label="r1 format")
    if any(v is not None for v in d["r1_valid"]):
        ax.plot(d["step"], d["r1_valid"], color="#d62728", lw=1, alpha=0.4, label="r1 valid factor")
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.set_title("Proposer Reward (round1/quality + components)")
    ax.set_xlabel("step"); ax.set_ylabel("value")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # 5. Actor PG loss (both rounds)
    ax = axes[1][1]
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.plot(d["step"], d["pg1"], lw=0.8, alpha=0.4, color="#1f77b4")
    ax.plot(d["step"], smooth(d["pg1"], 9), lw=2, color="#1f77b4", label="R1 pg_loss")
    ax.plot(d["step"], d["pg2"], lw=0.8, alpha=0.4, color="#d62728")
    ax.plot(d["step"], smooth(d["pg2"], 9), lw=2, color="#d62728", label="R2 pg_loss")
    ax.set_title("Actor PG Loss")
    ax.set_xlabel("step"); ax.set_ylabel("pg_loss")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # 6. KL loss
    ax = axes[1][2]
    ax.plot(d["step"], d["kl1"], lw=0.8, alpha=0.5, color="#1f77b4", label="R1 kl_loss")
    ax.plot(d["step"], d["kl2"], lw=0.8, alpha=0.5, color="#d62728", label="R2 kl_loss")
    ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.6)
    ax.set_title("KL Loss (both rounds)")
    ax.set_xlabel("step"); ax.set_ylabel("kl_loss")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # 7. Grad norm
    ax = axes[2][0]
    ax.plot(d["step"], d["gn1"], lw=0.8, alpha=0.5, color="#1f77b4", label="R1 grad_norm")
    ax.plot(d["step"], smooth(d["gn1"], 9), lw=2, color="#1f77b4")
    ax.plot(d["step"], d["gn2"], lw=0.8, alpha=0.5, color="#d62728", label="R2 grad_norm")
    ax.plot(d["step"], smooth(d["gn2"], 9), lw=2, color="#d62728")
    ax.set_title("Gradient Norm")
    ax.set_xlabel("step"); ax.set_ylabel("grad_norm")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # 8. Entropy
    ax = axes[2][1]
    ax.plot(d["step"], d["ent1"], lw=0.8, alpha=0.5, color="#1f77b4", label="R1 entropy")
    ax.plot(d["step"], smooth(d["ent1"], 9), lw=2, color="#1f77b4")
    ax.plot(d["step"], d["ent2"], lw=0.8, alpha=0.5, color="#d62728", label="R2 entropy")
    ax.plot(d["step"], smooth(d["ent2"], 9), lw=2, color="#d62728")
    ax.set_title("Policy Entropy")
    ax.set_xlabel("step"); ax.set_ylabel("entropy_loss")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # 9. Response length + step time overlay
    ax = axes[2][2]
    ax2 = ax.twinx()
    valid_rl = [v if v is not None else np.nan for v in d["resp_len_mean"]]
    ax.plot(d["step"], valid_rl, lw=1, color="#9467bd", alpha=0.7,
            label="response length (mean)")
    ax.plot(d["step"], smooth(valid_rl, 9), lw=2, color="#9467bd")
    ax.set_ylabel("response length (tokens)", color="#9467bd")
    ax.tick_params(axis='y', labelcolor="#9467bd")
    st = [t/60.0 for t in d["step_time"]]
    ax2.plot(d["step"], st, lw=1, color="#8c564b", alpha=0.7, label="step time (min)")
    ax2.plot(d["step"], smooth(st, 9), lw=2, color="#8c564b")
    ax2.set_ylabel("step time (min)", color="#8c564b")
    ax2.tick_params(axis='y', labelcolor="#8c564b")
    ax.set_title("Response Length & Step Time")
    ax.set_xlabel("step")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"imitate-0808 Training Dashboard  (steps 1–{d['step'][-1]}, "
        f"base=rrs-744, batch=16 rollout.n=8, KL=0.01)",
        fontsize=14
    )
    out = os.path.join(OUT_DIR, "dashboard.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def print_summary(d):
    def mean(xs):
        xs = [x for x in xs if x is not None and x == x]
        return float(np.mean(xs)) if xs else float("nan")
    n = len(d["step"])
    print(f"steps: {n} (1..{d['step'][-1]})")
    print(f"solver overall reward:  mean={mean(d['r2_overall']):.3f}")
    print(f"solver invalid ratio:   mean={mean(d['invalid']):.3f}")
    print(f"solver accuracy:        mean={mean(d['r2_accuracy']):.3f}")
    print(f"proposer quality:       mean={mean(d['r1_quality']):.3f}")
    print(f"R1 pg_loss / R2 pg_loss: {mean(d['pg1']):.4f} / {mean(d['pg2']):.4f}")
    print(f"R1 kl_loss / R2 kl_loss: {mean(d['kl1']):.5f} / {mean(d['kl2']):.5f}")
    print(f"step time: mean {mean(d['step_time'])/60:.1f} min  total {sum(d['step_time'])/3600:.1f} h")


def main():
    rows = load_rows()
    sched = load_schedule()
    d = extract(rows, sched)
    out1 = plot_solver_reward(d)
    out2 = plot_solver_invalid(d)
    out3 = plot_overall_loss(d)
    out4 = plot_dashboard(d)
    out5 = plot_solver_reward_by_family(d)
    print(f"wrote:\n  {out1}\n  {out2}\n  {out3}\n  {out4}\n  {out5}\n")
    print_summary(d)

    # per-family summary
    for fam in ("vsi", "spar", "maze"):
        ys = [d["r2_overall"][i] for i, f in enumerate(d["fam"]) if f == fam]
        accs = [d["r2_accuracy"][i] for i, f in enumerate(d["fam"]) if f == fam]
        invs = [d["invalid"][i] for i, f in enumerate(d["fam"]) if f == fam]
        if not ys: continue
        print(f"  {fam.upper():5s} n={len(ys):3d}  reward_mean={np.mean(ys):.3f}  "
              f"acc_mean={np.mean(accs):.3f}  invalid_mean={np.mean(invs):.3f}")


if __name__ == "__main__":
    main()
