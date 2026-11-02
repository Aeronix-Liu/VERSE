#!/usr/bin/env python3
"""Plot imitate-0808 first-71-step curves vs old imitate_porposer run (same schedule)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NEW = '/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr-self-evolve-imitate-0808/experiment_log.jsonl'
OLD = '/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr_self_evolve_imitate_porposer/experiment_log.jsonl'
OUT = '/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/training_curve/2026-08-09_imitate0808_step71/dashboard.png'

def load(p):
    return [json.loads(l) for l in open(p)]

def task_of(r):
    sc = r.get('round1_scheduler', {}).get('selected_count', {})
    return max(sc, key=sc.get) if sc else '?'

def modality(t):
    if t == 'maze_navigation':
        return 'maze'
    if t.endswith('_mv'):
        return 'image_pair'
    if t.startswith('spar_') or t in ('single_image_relative_direction', 'distance_cam_obj',
                                      'depth_order_obj_obj', 'visibility_compare',
                                      'position_cam_obj', 'position_cam_reg', 'attribute_measurement'):
        return 'single_image'
    return 'scene'

def rolling(x, w=9):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = np.nanmean(x[lo:i + 1])
    return out

new, old = load(NEW), load(OLD)
n = len(new)
old = old[:n]
steps = [r['step'] for r in new]
mods = [modality(task_of(r)) for r in new]
MOD_COLOR = {'maze': 'tab:purple', 'scene': 'tab:green', 'single_image': 'tab:orange', 'image_pair': 'tab:red'}

new_ovr = [r['reward_round2']['overall'] for r in new]
new_acc = [r['reward_round2']['accuracy'] for r in new]
old_ovr = [r['reward_round2']['overall'] for r in old]
old_acc = [r['reward_round2']['accuracy'] for r in old]
new_inv = [r['round2']['invalid_ratio'] for r in new]
old_inv = [r['round2']['invalid_ratio'] for r in old]
r1q = [r['reward_round1']['quality'] for r in new]
r1f = [r['reward_round1']['quality_format'] for r in new]
kl2 = [r['actor']['kl_loss_round2'] for r in new]
gn1 = [r['actor']['grad_norm_round1'] for r in new]
gn2 = [r['actor']['grad_norm_round2'] for r in new]
ent2 = [r['actor']['entropy_loss_round2'] for r in new]

fig, axes = plt.subplots(2, 3, figsize=(19, 9))
fig.suptitle('imitate-0808 @ step %d / 300  —  vs old imitate_porposer (same schedule v2, same steps)' % n, fontsize=14)

ax = axes[0, 0]
for m in MOD_COLOR:
    xs = [s for s, mm in zip(steps, mods) if mm == m]
    ys = [v for v, mm in zip(new_ovr, mods) if mm == m]
    ax.scatter(xs, ys, s=18, color=MOD_COLOR[m], label=m)
ax.plot(steps, rolling(new_ovr), color='black', lw=2, label='new (roll-9)')
ax.plot(steps, rolling(old_ovr), color='gray', lw=2, ls='--', label='old (roll-9)')
ax.set_title('Round-2 overall reward')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
for m in MOD_COLOR:
    xs = [s for s, mm in zip(steps, mods) if mm == m]
    ys = [v for v, mm in zip(new_acc, mods) if mm == m]
    ax.scatter(xs, ys, s=18, color=MOD_COLOR[m])
ax.plot(steps, rolling(new_acc), color='black', lw=2, label='new (roll-9)')
ax.plot(steps, rolling(old_acc), color='gray', lw=2, ls='--', label='old (roll-9)')
ax.set_title('Round-2 accuracy')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.plot(steps, rolling(new_inv), color='tab:blue', lw=2, label='new (roll-9)')
ax.plot(steps, rolling(old_inv), color='gray', lw=2, ls='--', label='old (roll-9)')
ax.scatter(steps, new_inv, s=10, color='tab:blue', alpha=0.4)
ax.set_title('Round-2 invalid ratio')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
for m in MOD_COLOR:
    xs = [s for s, mm in zip(steps, mods) if mm == m]
    ys = [v for v, mm in zip(r1q, mods) if mm == m]
    ax.scatter(xs, ys, s=18, color=MOD_COLOR[m], label=m)
ax.plot(steps, rolling(r1q), color='black', lw=1.5)
ax.plot(steps, r1f, color='tab:cyan', lw=1, alpha=0.7, label='r1 format')
ax.set_title('Round-1 proposer quality / format')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(steps, gn1, color='tab:red', lw=1, alpha=0.7, label='grad_norm R1')
ax.plot(steps, gn2, color='tab:blue', lw=1, alpha=0.9, label='grad_norm R2')
ax2 = ax.twinx()
ax2.plot(steps, kl2, color='tab:green', lw=1, label='KL R2 (right)')
ax2.tick_params(axis='y', labelcolor='tab:green')
ax.set_title('Grad norms + KL(R2)')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.3)

ax = axes[1, 2]
ax.plot(steps, ent2, color='tab:brown', lw=1.5, label='entropy R2')
ax.set_title('Round-2 entropy')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

for a in axes.flat:
    a.set_xlabel('step')

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=110)
print('saved', OUT)
