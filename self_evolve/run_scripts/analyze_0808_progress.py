#!/usr/bin/env python3
"""Analyze imitate-0808 training progress (first ~70 steps) vs old imitate_porposer run."""
import json
from collections import defaultdict

NEW = '/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr-self-evolve-imitate-0808/experiment_log.jsonl'
OLD = '/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr_self_evolve_imitate_porposer/experiment_log.jsonl'

def load(p):
    return [json.loads(l) for l in open(p)]

def task_of(r):
    sc = r.get('round1_scheduler', {}).get('selected_count', {})
    return max(sc, key=sc.get) if sc else '?'

def modality(t):
    if t == 'maze_navigation':
        return 'maze'
    if t.endswith('_mv') or t in ('motion_camera', 'position_cam_cam', 'elevation_cam_cam'):
        return 'image_pair'
    if t.startswith('spar_') or t in ('single_image_relative_direction', 'distance_cam_obj',
                                      'depth_order_obj_obj', 'visibility_compare',
                                      'position_cam_obj', 'position_cam_reg', 'attribute_measurement'):
        return 'single_image'
    return 'scene'

new = load(NEW)
old = load(OLD)

print('=' * 100)
print('PER-STEP TABLE (new run imitate-0808, %d steps)' % len(new))
print('=' * 100)
hdr = f"{'step':>4} {'task':<34} {'r1_qual':>7} {'r1_fmt':>6} {'r1_valid':>8} {'r2_ovr':>6} {'r2_acc':>6} {'r2_fmt':>6} {'inv':>5} {'kl2':>7} {'pg2':>8} {'gn2':>6} {'ent2':>5} {'rlen':>5} {'t(s)':>5}"
print(hdr)
for r in new:
    t = task_of(r)
    r1 = r['reward_round1']
    r2 = r['reward_round2']
    a = r['actor']
    print(f"{r['step']:>4} {t:<34} {r1['quality']:>7.3f} {r1['quality_format']:>6.3f} {r1.get('quality_valid_factor', float('nan')):>8.3f} "
          f"{r2['overall']:>6.3f} {r2['accuracy']:>6.3f} {r2['format']:>6.3f} {r['round2']['invalid_ratio']:>5.2f} "
          f"{a['kl_loss_round2']:>7.4f} {a['pg_loss_round2']:>8.4f} {a['grad_norm_round2']:>6.2f} {a['entropy_loss_round2']:>5.2f} "
          f"{r['response_length']['mean']:>5.0f} {r['timing_s']['step']:>5.0f}")

# ---- aggregate halves ----
def agg(rows, keys):
    out = {}
    for name, fn in keys.items():
        vals = [fn(r) for r in rows]
        vals = [v for v in vals if v == v]
        out[name] = sum(vals) / max(len(vals), 1)
    return out

keys = {
    'r1_quality': lambda r: r['reward_round1']['quality'],
    'r1_format': lambda r: r['reward_round1']['quality_format'],
    'r2_overall': lambda r: r['reward_round2']['overall'],
    'r2_accuracy': lambda r: r['reward_round2']['accuracy'],
    'r2_format': lambda r: r['reward_round2']['format'],
    'invalid_ratio': lambda r: r['round2']['invalid_ratio'],
    'valid_case': lambda r: r['reward_round2']['valid_case'],
    'kl_r2': lambda r: r['actor']['kl_loss_round2'],
    'ppo_kl_r2': lambda r: r['actor']['ppo_kl_round2'],
    'grad_norm_r2': lambda r: r['actor']['grad_norm_round2'],
    'grad_norm_r1': lambda r: r['actor']['grad_norm_round1'],
    'entropy_r2': lambda r: r['actor']['entropy_loss_round2'],
    'resp_len': lambda r: r['response_length']['mean'],
    'step_time': lambda r: r['timing_s']['step'],
    'dedup_ratio': lambda r: r['round2'].get('dedup_removed_question_ratio', float('nan')),
}

n = len(new)
h1, h2 = new[:n // 2], new[n // 2:]
print()
print('=' * 100)
print('NEW RUN: first half (1-%d) vs second half (%d-%d)' % (n // 2, n // 2 + 1, n))
print('=' * 100)
a1, a2 = agg(h1, keys), agg(h2, keys)
for k in keys:
    print(f"  {k:<16} {a1[k]:>9.4f}  ->  {a2[k]:>9.4f}   ({a2[k] - a1[k]:+.4f})")

# ---- same-step comparison with old run ----
m = min(len(new), len(old))
print()
print('=' * 100)
print('SAME-STEP COMPARISON: new(1-%d) vs old imitate_porposer(1-%d)  [same schedule v2]' % (m, m))
print('=' * 100)
# verify schedule alignment
mismatch = sum(1 for i in range(m) if task_of(new[i]) != task_of(old[i]))
print('task-type mismatches over %d steps: %d' % (m, mismatch))
anew, aold = agg(new[:m], keys), agg(old[:m], keys)
for k in keys:
    print(f"  {k:<16} old {aold[k]:>9.4f}   new {anew[k]:>9.4f}   ({anew[k] - aold[k]:+.4f})")

# ---- per-task-type ----
print()
print('=' * 100)
print('PER-TASK-TYPE (new run): r2_overall / r2_acc / invalid, with old-run same-step values')
print('=' * 100)
groups = defaultdict(list)
for i, r in enumerate(new):
    groups[task_of(r)].append(i)
print(f"{'task':<36} {'mod':<12} {'n':>2} | {'new_ovr':>7} {'new_acc':>7} {'new_inv':>7} | {'old_ovr':>7} {'old_acc':>7} {'old_inv':>7}")
rows_sorted = sorted(groups.items(), key=lambda kv: -len(kv[1]))
for t, idxs in rows_sorted:
    def mean(rs, fn):
        vs = [fn(rs[i]) for i in idxs if i < len(rs)]
        return sum(vs) / len(vs) if vs else float('nan')
    no = mean(new, lambda r: r['reward_round2']['overall'])
    na = mean(new, lambda r: r['reward_round2']['accuracy'])
    ni = mean(new, lambda r: r['round2']['invalid_ratio'])
    oo = mean(old, lambda r: r['reward_round2']['overall'])
    oa = mean(old, lambda r: r['reward_round2']['accuracy'])
    oi = mean(old, lambda r: r['round2']['invalid_ratio'])
    print(f"{t:<36} {modality(t):<12} {len(idxs):>2} | {no:>7.3f} {na:>7.3f} {ni:>7.3f} | {oo:>7.3f} {oa:>7.3f} {oi:>7.3f}")

# ---- per-modality trend within new run ----
print()
print('=' * 100)
print('PER-MODALITY TREND (new run): first-half vs second-half mean of r2_overall / r2_acc / invalid')
print('=' * 100)
mod_groups = defaultdict(list)
for i, r in enumerate(new):
    mod_groups[modality(task_of(r))].append(i)
for mod, idxs in sorted(mod_groups.items()):
    fh = [i for i in idxs if i < n // 2]
    sh = [i for i in idxs if i >= n // 2]
    def mm(ii, fn):
        vs = [fn(new[i]) for i in ii]
        return sum(vs) / len(vs) if vs else float('nan')
    print(f"{mod:<14} n={len(fh)}/{len(sh)}  "
          f"ovr {mm(fh, lambda r: r['reward_round2']['overall']):.3f}->{mm(sh, lambda r: r['reward_round2']['overall']):.3f}  "
          f"acc {mm(fh, lambda r: r['reward_round2']['accuracy']):.3f}->{mm(sh, lambda r: r['reward_round2']['accuracy']):.3f}  "
          f"inv {mm(fh, lambda r: r['round2']['invalid_ratio']):.3f}->{mm(sh, lambda r: r['round2']['invalid_ratio']):.3f}  "
          f"r1q {mm(fh, lambda r: r['reward_round1']['quality']):.3f}->{mm(sh, lambda r: r['reward_round1']['quality']):.3f}")
