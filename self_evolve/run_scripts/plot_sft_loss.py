#!/usr/bin/env python3
"""Plot cold-start SFT and RRS SFT loss curves.

Sources (LLaMA-Factory `trainer_log.jsonl`, one record per 10 optim steps):
  * cold-start: /data/…/checkpoints/vilasr_cold_start_4gpu/trainer_log.jsonl
  * RRS:        /data/…/checkpoints/vilasr_reflective_4gpu/trainer_log.jsonl

Outputs go to `training_curve/SFT/` (PNG + vector PDF):
  * cold_start_loss.{png,pdf}
  * rrs_loss.{png,pdf}
  * sft_loss_combined.{png,pdf} (both on same axes, epoch-normalized)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XINRAN = "/data/pengtianshuo/self_evolve/Xinran"
OUT_DIR = f"{XINRAN}/ViLaSR-self-evolve/training_curve/SFT"
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = {
    "cold-start (SFT, 3258 steps, 3 epochs)":
        f"{XINRAN}/checkpoints/vilasr_cold_start_4gpu/trainer_log.jsonl",
    "RRS (SFT, 744 steps, 3 epochs)":
        f"{XINRAN}/checkpoints/vilasr_reflective_4gpu/trainer_log.jsonl",
}
COLORS = {
    "cold-start (SFT, 3258 steps, 3 epochs)": "#1f77b4",
    "RRS (SFT, 744 steps, 3 epochs)":         "#d62728",
}


def load_run(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "loss" in r and r.get("loss") is not None:
            rows.append(r)
    return rows


def save_fig(fig, out_path, dpi=150):
    fig.savefig(out_path, dpi=dpi)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    return out_path, pdf_path


def smooth(vals, w=5):
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


def plot_single(name, rows, color, out_path, xkey="current_steps",
                xlabel="optimizer step"):
    xs = [r[xkey] for r in rows]
    losses = [r["loss"] for r in rows]
    lrs = [r.get("lr") for r in rows]
    epochs = [r.get("epoch") for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(xs, losses, lw=0.9, alpha=0.5, color=color)
    ax.plot(xs, smooth(losses, 9), lw=2.2, color=color, label="loss (smoothed)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("training loss", color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax.grid(alpha=0.25)
    # epoch boundaries (integer epochs)
    if epochs and any(e is not None for e in epochs):
        for e in range(1, int(max(epochs)) + 1):
            # find first step at or after epoch e
            idx = next((i for i, ep in enumerate(epochs) if ep is not None and ep >= e), None)
            if idx is not None:
                ax.axvline(xs[idx], color="grey", lw=0.7, ls=":", alpha=0.6)
                ax.text(xs[idx], ax.get_ylim()[1] * 0.98, f"epoch {e}",
                        rotation=90, va="top", ha="right", fontsize=8, color="grey")
    if any(v is not None for v in lrs):
        ax2 = ax.twinx()
        ax2.plot(xs, lrs, lw=0.9, alpha=0.5, color="#2ca02c", label="lr")
        ax2.set_ylabel("learning rate", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax.set_title(f"{name}  —  loss (min {min(losses):.4f} → last {losses[-1]:.4f})")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    return out_path


def plot_combined(runs_data, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, rows in runs_data.items():
        epochs = [r.get("epoch") for r in rows]
        losses = [r["loss"] for r in rows]
        color = COLORS[name]
        ax.plot(epochs, losses, lw=0.9, alpha=0.4, color=color)
        ax.plot(epochs, smooth(losses, 9), lw=2, color=color,
                label=f"{name}: min={min(losses):.4f} last={losses[-1]:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("SFT training loss — cold-start vs RRS (epoch-aligned)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)
    return out_path


def main():
    runs_data = {}
    outs = []
    tag_map = {
        "cold-start (SFT, 3258 steps, 3 epochs)": "cold_start_loss.png",
        "RRS (SFT, 744 steps, 3 epochs)":         "rrs_loss.png",
    }
    for name, path in RUNS.items():
        rows = load_run(path)
        runs_data[name] = rows
        out = plot_single(name, rows, COLORS[name],
                          os.path.join(OUT_DIR, tag_map[name]))
        outs.append(out)
        print(f"  {name}: n={len(rows)}  min_loss={min(r['loss'] for r in rows):.4f}  "
              f"last_loss={rows[-1]['loss']:.4f}  final_epoch={rows[-1].get('epoch')}")
    combined = plot_combined(runs_data, os.path.join(OUT_DIR, "sft_loss_combined.png"))
    outs.append(combined)
    print(f"wrote:")
    for o in outs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
