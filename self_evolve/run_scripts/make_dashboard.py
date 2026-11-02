#!/usr/bin/env python3
"""Compose the 7 mix_training curve PNGs into a single dashboard.png (2x4 grid)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CURVE_DIR = Path("/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/training_curve/mix_training")
OUT = CURVE_DIR / "dashboard.png"

PLOTS = [
    ("round1_reward.png", "Round-1 Questioner reward"),
    ("round2_reward.png", "Round-2 Solver reward"),
    ("round2_validity.png", "Round-2 validity"),
    ("actor_loss.png", "Actor loss"),
    ("actor_norms.png", "Actor norms"),
    ("lengths.png", "Sequence lengths"),
    ("timing.png", "Per-step timing"),
]

fig, axes = plt.subplots(2, 4, figsize=(28, 13))
fig.suptitle("vilasr_full_selfevolve — mix training (maze+VSI+SPAR, 200 steps)", fontsize=20, fontweight="bold", y=0.98)

for idx, (fname, title) in enumerate(PLOTS):
    r, c = divmod(idx, 4)
    ax = axes[r][c]
    img = plt.imread(str(CURVE_DIR / fname))
    ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

# hide the last (empty) cell
axes[1][3].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(str(OUT), dpi=120, bbox_inches="tight", facecolor="white")
print(f"[dashboard] wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
