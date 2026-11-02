#!/bin/bash
# VERSE agentic evaluation on the three target benchmarks: VSI-Bench,
# SPAR-Bench, MAZE. Runs the ViLaSR-style think->draw->observe rollout
# (eval/infer.py), sharded across ${VERSE_GPUS}, then scores with
# eval/evaluate.py + the official category-mean metrics (eval/utils/cal_metric_*).
#
# Usage:
#   bash scripts/eval_benchmarks.sh MODEL_DIR [vsi_bench|spar_bench|maze ...]
# MODEL_DIR is an HF-format model dir. For an FSDP self-evolve checkpoint, first
# merge it:  python self_evolve/easy_r1/scripts/model_merger.py --local_dir <ckpt>/actor
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/configs/paths.env"

MODEL_DIR="${1:?usage: eval_benchmarks.sh MODEL_DIR [benches...]}"; shift || true
BENCHES=("${@:-vsi_bench spar_bench maze}"); BENCHES=(${BENCHES[@]})
IFS=', ' read -r -a GPUS <<< "${VERSE_GPUS}"
NSHARD="${#GPUS[@]}"
OUTROOT="${VERSE_OUTPUT}/eval/$(basename "${MODEL_DIR}")"
EVAL="${HERE}/eval"
RUN() { conda run -n verse-eval --no-capture-output "$@"; }
mkdir -p "${OUTROOT}"

declare -A DJSON=( [vsi_bench]=vsi_bench.json [spar_bench]=spar_bench.json [maze]=maze.json )
declare -A BFOLDER=( [vsi_bench]=VSI-Bench [spar_bench]=SPAR-Bench [maze]=MAZE )

for bkey in "${BENCHES[@]}"; do
  MDIR="${OUTROOT}/${BFOLDER[$bkey]}"; mkdir -p "${MDIR}"
  echo "[eval] ${bkey} | model=${MODEL_DIR} | shards=${NSHARD}"
  PIDS=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" RUN python "${EVAL}/infer.py" \
      --model-path "${MODEL_DIR}" --model-name Qwen2.5_VL_7B --dataset "${bkey}" \
      --input-file "${VERSE_BENCH}/${DJSON[$bkey]}" --image-folder "${VERSE_BENCH}" \
      --output-dir "${MDIR}" --temperature 0.75 --max-frames 16 --max-pixels 351232 \
      --split "${i}" --all "${NSHARD}" > "${MDIR}/shard${i}.log" 2>&1 &
    PIDS+=($!); sleep 2
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
  # merge shard results
  MERGED="${MDIR}/results.jsonl"; : > "${MERGED}"
  for ((i=0;i<NSHARD;i++)); do [ -f "${MDIR}/${i}_${NSHARD}/results.jsonl" ] && cat "${MDIR}/${i}_${NSHARD}/results.jsonl" >> "${MERGED}"; done
  echo "[eval] ${bkey}: $(wc -l < "${MERGED}") results"
  RUN python "${EVAL}/evaluate.py" --dataset "${bkey}" \
      --question-file "${MERGED}" --result-file "${MERGED}" \
      --output-result "${MDIR}/scores.jsonl" 2>&1 | tail -5
done
echo "[eval] done. Raw + scores under ${OUTROOT}"
echo "[eval] VSI/SPAR official category-mean overall: see eval/utils/cal_metric_{vsibench,sparbench}.py"
