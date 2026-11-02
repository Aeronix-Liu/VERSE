#!/usr/bin/env python3
"""Extract per-step training metrics from an experiment_log.jsonl into a dated
training_curve/<YYYY-MM-DD>/ folder, write CSV + PNG plots + a markdown analysis.

Reusable across experiments. Run from the repo root:

    python run_scripts/export_training_curve.py \
        --log  /data/pengtianshuo/self_evolve/Xinran/checkpoints/<EXP>/experiment_log.jsonl \
        --date 2026-07-24 \
        --tag  vilasr_selfevolve_4gpu

If --date is omitted, the date is read from the first row's wall-clock (if
present) or the file's mtime; falling back to "unknown".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Metric selection — the keys we actually care about for self-evolve analysis.
# ---------------------------------------------------------------------------
ROUND1_KEYS = [
    "quality", "quality_format", "quality_valid_factor", "quality_f_observation",
    "quality_question_quality", "quality_accuracy",
]
ROUND2_KEYS = [
    "overall", "format", "accuracy", "answer_correctness",
    "valid_case", "invalid_case", "invalid_reason_score", "reflection", "diversity",
]
ROUND2_META_KEYS = [
    "invalid_ratio", "invalid_count",
    "dedup_removed_question_ratio", "unique_question_count_before_answer",
]
ACTOR_KEYS = [
    "pg_loss_round1", "pg_loss_round2", "kl_loss_round1", "kl_loss_round2",
    "ppo_kl_round1", "ppo_kl_round2", "grad_norm_round1", "grad_norm_round2",
    "entropy_loss_round1", "entropy_loss_round2",
]
LENGTH_KEYS = ["prompt_mean", "response_mean", "prompt_max", "response_max"]
PERF_KEYS = ["time_per_step", "throughput"]
TIMING_KEYS = ["gen_round1", "gen_round2", "reward_round1", "reward_round2",
               "update_actor_round1", "update_actor_round2"]


def flatten_row(row: dict) -> dict:
    out = {"step": int(row["step"])}
    r1 = row.get("reward_round1", {}) or {}
    r2 = row.get("reward_round2", {}) or {}
    r2m = row.get("round2", {}) or {}
    act = row.get("actor", {}) or {}
    perf = row.get("perf", {}) or {}
    tim = row.get("timing_s", {}) or {}
    pl = row.get("prompt_length", {}) or {}
    rl = row.get("response_length", {}) or {}
    for k in ROUND1_KEYS:
        out[f"r1_{k}"] = r1.get(k)
    for k in ROUND2_KEYS:
        out[f"r2_{k}"] = r2.get(k)
    for k in ROUND2_META_KEYS:
        out[f"r2m_{k}"] = r2m.get(k)
    for k in ACTOR_KEYS:
        out[f"act_{k}"] = act.get(k)
    for k in PERF_KEYS:
        out[f"perf_{k}"] = perf.get(k)
    for k in TIMING_KEYS:
        out[f"tim_{k}"] = tim.get(k)
    out["len_prompt_mean"] = pl.get("mean")
    out["len_response_mean"] = rl.get("mean")
    out["len_prompt_max"] = pl.get("max")
    out["len_response_max"] = rl.get("max")
    return out


def load_rows(log_path: Path) -> list[dict]:
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(flatten_row(json.loads(line)))
    return rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else f"{r[k]}" for k in keys) + "\n")


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        print(f"[export] matplotlib unavailable ({exc}); skipping plots")
        return None


def plot_series(plt, steps, series_dict, ylabel, title, out_path, smooth=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, vals in series_dict.items():
        v = np.array([x if x is not None else np.nan for x in vals], dtype=float)
        if smooth and len(v) > smooth:
            k = np.ones(smooth) / smooth
            v_smooth = np.convolve(np.nan_to_num(v), k, mode="same")
            ax.plot(steps, v, alpha=0.25, linewidth=1)
            ax.plot(steps, v_smooth, label=name, linewidth=1.6)
        else:
            ax.plot(steps, v, label=name, linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_plots(rows, out_dir: Path):
    plt = try_import_matplotlib()
    if plt is None or not rows:
        return
    steps = [r["step"] for r in rows]

    def col(key):
        return [r.get(key) for r in rows]

    plot_series(plt, steps,
                {"quality": col("r1_quality"),
                 "format": col("r1_quality_format"),
                 "valid_factor": col("r1_quality_valid_factor"),
                 "obs": col("r1_quality_f_observation")},
                "reward", "Round-1 Questioner reward",
                out_dir / "round1_reward.png", smooth=10)

    plot_series(plt, steps,
                {"overall": col("r2_overall"),
                 "format": col("r2_format"),
                 "accuracy": col("r2_accuracy"),
                 "answer_correctness": col("r2_answer_correctness")},
                "reward", "Round-2 Solver reward",
                out_dir / "round2_reward.png", smooth=10)

    plot_series(plt, steps,
                {"invalid_ratio": col("r2m_invalid_ratio"),
                 "valid_case": col("r2_valid_case"),
                 "invalid_case": col("r2_invalid_case"),
                 "invalid_reason_score": col("r2_invalid_reason_score")},
                "ratio / score", "Round-2 validity & invalid-handling",
                out_dir / "round2_validity.png", smooth=10)

    plot_series(plt, steps,
                {"pg_loss_r1": col("act_pg_loss_round1"),
                 "pg_loss_r2": col("act_pg_loss_round2"),
                 "kl_r1": col("act_kl_loss_round1"),
                 "kl_r2": col("act_kl_loss_round2")},
                "loss", "Actor losses",
                out_dir / "actor_loss.png", smooth=10)

    plot_series(plt, steps,
                {"grad_r1": col("act_grad_norm_round1"),
                 "grad_r2": col("act_grad_norm_round2"),
                 "ppo_kl_r1": col("act_ppo_kl_round1"),
                 "ppo_kl_r2": col("act_ppo_kl_round2")},
                "norm", "Grad norm & PPO KL",
                out_dir / "actor_norms.png", smooth=10)

    plot_series(plt, steps,
                {"prompt": col("len_prompt_mean"),
                 "response": col("len_response_mean")},
                "tokens", "Prompt / response length",
                out_dir / "lengths.png", smooth=10)

    plot_series(plt, steps,
                {"time_per_step": col("perf_time_per_step"),
                 "gen_round2": col("tim_gen_round2"),
                 "update_r2": col("tim_update_actor_round2")},
                "seconds", "Per-step timing",
                out_dir / "timing.png", smooth=10)


def scheduler_table(rows) -> list[str]:
    """Per-task scheduler history_acc at the last step."""
    last = rows[-1] if rows else None
    if last is None:
        return []
    # We didn't flatten scheduler stats; re-read from the raw log.
    return []  # handled separately in main


def write_analysis(rows, raw_rows, out_dir: Path, tag: str, date: str):
    if not rows:
        return
    n = len(rows)
    last = rows[-1]
    first = rows[0]

    def stats(key, prefix=""):
        k = f"{prefix}{key}"
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        if not vals:
            return None
        v = np.array(vals, dtype=float)
        return dict(first=float(v[0]), last=float(v[-1]),
                    mean=float(np.mean(v)), min=float(np.min(v)),
                    max=float(np.max(v)))

    def fmt(x, p=3):
        return "-" if x is None else f"{x:.{p}f}"

    lines = []
    lines.append(f"# Training curve — {tag} ({date})\n")
    lines.append(f"Source: `experiment_log.jsonl` ({n} steps). "
                 f"Plots & CSV in this folder.\n")

    lines.append("## Headline numbers\n")
    lines.append(f"- steps: {n}")
    lines.append(f"- r1.quality: first={fmt(stats('quality','r1_')['first'])} "
                 f"→ last={fmt(stats('quality','r1_')['last'])} "
                 f"(mean {fmt(stats('quality','r1_')['mean'])})")
    lines.append(f"- r2.overall: first={fmt(stats('overall','r2_')['first'])} "
                 f"→ last={fmt(stats('overall','r2_')['last'])} "
                 f"(mean {fmt(stats('overall','r2_')['mean'])})")
    lines.append(f"- r2.accuracy: first={fmt(stats('accuracy','r2_')['first'])} "
                 f"→ last={fmt(stats('accuracy','r2_')['last'])} "
                 f"(mean {fmt(stats('accuracy','r2_')['mean'])})")
    lines.append(f"- r2.format: first={fmt(stats('format','r2_')['first'])} "
                 f"→ last={fmt(stats('format','r2_')['last'])} "
                 f"(mean {fmt(stats('format','r2_')['mean'])})")
    lines.append(f"- r2m.invalid_ratio: first={fmt(stats('invalid_ratio','r2m_')['first'])} "
                 f"→ last={fmt(stats('invalid_ratio','r2m_')['last'])} "
                 f"(mean {fmt(stats('invalid_ratio','r2m_')['mean'])})")
    lines.append(f"- r2.invalid_reason_score: mean={fmt(stats('invalid_reason_score','r2_')['mean'])}")
    lines.append(f"- response_length: first={fmt(stats('response_mean','len_')['first'])} "
                 f"→ last={fmt(stats('response_mean','len_')['last'])}")
    lines.append(f"- time_per_step: mean={fmt(stats('time_per_step','perf_')['mean'], 1)}s")

    # trend: compare first 20% vs last 20%
    def trend(key, prefix=""):
        k = f"{prefix}{key}"
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        if len(vals) < 10:
            return None
        v = np.array(vals, dtype=float)
        m = max(1, len(v) // 5)
        a = float(np.mean(v[:m]))
        b = float(np.mean(v[-m:]))
        return a, b

    lines.append("\n## Trend (first 20% vs last 20%)\n")
    lines.append("| metric | first-20% | last-20% | Δ |")
    lines.append("|---|---|---|---|")
    for name, key, prefix in [
        ("r1.quality", "quality", "r1_"),
        ("r2.overall", "overall", "r2_"),
        ("r2.accuracy", "accuracy", "r2_"),
        ("r2.format", "format", "r2_"),
        ("r2.invalid_ratio", "invalid_ratio", "r2m_"),
        ("r2.invalid_reason_score", "invalid_reason_score", "r2_"),
        ("len.response_mean", "response_mean", "len_"),
    ]:
        t = trend(key, prefix)
        if t is None:
            lines.append(f"| {name} | - | - | - |")
        else:
            a, b = t
            lines.append(f"| {name} | {a:.3f} | {b:.3f} | {b-a:+.3f} |")

    # scheduler acc per task at last step
    lines.append("\n## Task scheduler (last step history_acc)\n")
    if raw_rows:
        last_raw = raw_rows[-1]
        sched = (last_raw.get("round1_scheduler") or {}).get("history_acc") or {}
        hist = (last_raw.get("round1_scheduler") or {}).get("history_count") or {}
        if sched:
            lines.append("| task | count | history_acc |")
            lines.append("|---|---|---|")
            for task in sorted(sched.keys()):
                lines.append(f"| {task} | {int(hist.get(task, 0))} | {sched.get(task, 0):.3f} |")

    # selected_count aggregated across all steps
    if raw_rows:
        agg = {}
        for r in raw_rows:
            sc = (r.get("round1_scheduler") or {}).get("selected_count") or {}
            for t, c in sc.items():
                agg[t] = agg.get(t, 0.0) + float(c)
        if agg:
            lines.append("\n## Task selection frequency (all steps)\n")
            lines.append("| task | selected_count |")
            lines.append("|---|---|")
            for t, c in sorted(agg.items(), key=lambda x: -x[1]):
                lines.append(f"| {t} | {int(c)} |")

    lines.append("\n## Plots\n")
    lines.append("- `round1_reward.png` — Questioner reward components")
    lines.append("- `round2_reward.png` — Solver reward components")
    lines.append("- `round2_validity.png` — invalid_ratio / valid_case / invalid_case")
    lines.append("- `actor_loss.png` — pg_loss / kl_loss (both rounds)")
    lines.append("- `actor_norms.png` — grad_norm / ppo_kl")
    lines.append("- `lengths.png` — prompt / response length")
    lines.append("- `timing.png` — per-step timing breakdown")
    lines.append("- `metrics.csv` — all flattened per-step metrics")

    (out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to experiment_log.jsonl")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD folder name")
    ap.add_argument("--tag", default="run", help="experiment tag (used in analysis header)")
    ap.add_argument("--out-root", default=None,
                    help="training_curve root (default: <repo>/training_curve)")
    args = ap.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        print(f"ERROR: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    raw_rows = [json.loads(l) for l in open(log_path) if l.strip()]
    rows = [flatten_row(r) for r in raw_rows]
    if not rows:
        print("ERROR: no rows", file=sys.stderr)
        sys.exit(1)

    if args.date:
        date = args.date
    else:
        date = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y-%m-%d")

    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "training_curve"
    out_dir = out_root / date
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_dir / "metrics.csv")
    make_plots(rows, out_dir)
    write_analysis(rows, raw_rows, out_dir, tag=args.tag, date=date)
    print(f"\n[export] wrote {out_dir}")


if __name__ == "__main__":
    main()
