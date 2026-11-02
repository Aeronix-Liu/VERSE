#!/bin/bash
# Monitor the full self-evolve run; emit a sentinel on crash or completion.
LOG=$(cat /data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/logs/latest_vilasr_full_selfevolve.logpath 2>/dev/null)
EXPLOG=/data/pengtianshuo/self_evolve/Xinran/checkpoints/vilasr_full_selfevolve/experiment_log.jsonl
while true; do
  sleep 1200
  if [ -f "$LOG" ]; then
    if grep -qE "Traceback|RuntimeError|Too many consecutively|Caught FileNotFoundError" "$LOG" 2>/dev/null; then
      echo "TRAINING_CRASH detected in $LOG"; exit 0
    fi
    if grep -qE "exited rc=|Training completed|Done \(log:" "$LOG" 2>/dev/null; then
      echo "TRAINING_DONE: $LOG"; exit 0
    fi
  fi
  STEPS=$(python3 -c "import json;print(len([1 for _ in open('$EXPLOG')]))" 2>/dev/null || echo 0)
  echo "monitor_tick: steps_done=$STEPS at $(date +%H:%M)"
done
