#!/bin/bash
# Lightweight persistent monitor for the grounding self-evolve run. Appends a
# health snapshot (grounding trend + last steps + ckpts) to logs/grnd_monitor.log
# every INTERVAL seconds until training finishes. Survives SSH disconnect (tmux).
set -u
XINRAN="/data/pengtianshuo/self_evolve/Xinran"
EXP="vilasr-se-grounding-100step"
ELOG="${XINRAN}/checkpoints/${EXP}/experiment_log.jsonl"
TRAIN_LOG="$(cat ${XINRAN}/ViLaSR-self-evolve/logs/latest_${EXP}.logpath 2>/dev/null)"
OUT="${XINRAN}/logs/grnd_monitor.log"
INTERVAL="${INTERVAL:-1200}"

snap() {
    echo "===== $(date '+%F %T') ($(date -u '+%H:%M')UTC) ====="
    if tmux ls 2>/dev/null | grep -q grnd_run; then echo "grnd_run: ALIVE"; else echo "grnd_run: GONE"; fi
    ls -d ${XINRAN}/checkpoints/${EXP}/global_step_*/ 2>/dev/null | grep -v _merged | sed 's#.*/global#  ckpt: global#'
    python3 - "$ELOG" <<'PYEOF'
import json, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print("  (no experiment_log yet)"); raise SystemExit
R = [json.loads(l) for l in open(p)]
print(f"  steps logged: {len(R)}  last step: {R[-1]['step']}")
g = [(r['step'], round(r.get('reward_round2', {}).get('vsi_grounding'), 3))
     for r in R if r.get('reward_round2', {}).get('vsi_grounding') is not None]
if g:
    print(f"  scene grounding by step: {g}")
    early = [v for _, v in g[:max(1, len(g)//3)]]
    late = [v for _, v in g[-max(1, len(g)//3):]]
    print(f"  grounding early-avg={sum(early)/len(early):.3f}  late-avg={sum(late)/len(late):.3f}"
          f"  (trend {'UP' if sum(late)/len(late) > sum(early)/len(early) else 'flat/down'})")
mz = [(r['step'], round(r.get('reward_round2', {}).get('maze_traj_fidelity'), 3))
      for r in R if r.get('reward_round2', {}).get('maze_traj_fidelity') is not None]
if mz:
    print(f"  maze fidelity by step: {mz}")
for r in R[-3:]:
    rr = r.get('reward_round2', {})
    print(f"  step {r['step']} r2={round(rr.get('overall',-1),3)} "
          f"grnd={rr.get('vsi_grounding')} cov={rr.get('vsi_grounding_coverage')} "
          f"mazefid={rr.get('maze_traj_fidelity')} step_s={round(r.get('timing_s',{}).get('step',0))}")
PYEOF
    echo
}

echo "[grnd_monitor started $(date '+%F %T'); interval=${INTERVAL}s]" >> "${OUT}"
while true; do
    snap >> "${OUT}" 2>&1
    # stop once training has clearly finished
    if [ -n "${TRAIN_LOG}" ] && grep -q "Training process exited rc=" "${TRAIN_LOG}" 2>/dev/null; then
        echo "[grnd_monitor: training exited; final snapshot above; stopping]" >> "${OUT}"; break
    fi
    if ! pgrep -f "train_main.py.*${EXP}" >/dev/null 2>&1 && [ -d "${XINRAN}/checkpoints/${EXP}/global_step_100" ]; then
        echo "[grnd_monitor: train_main gone + step_100 exists; stopping]" >> "${OUT}"; break
    fi
    sleep "${INTERVAL}"
done
