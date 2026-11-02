#!/bin/bash
# VERSE environment setup. Creates the three conda envs the pipeline needs:
#   1) verse-sft   — LLaMA-Factory (cold-start + reflective SFT stages)
#   2) verse       — verl/EasyR1 + vLLM>=0.8 (self-evolve GRPO stage)
#   3) verse-eval  — vLLM 0.7.3 pinned (agentic 3-benchmark eval)
#
# Usage: bash scripts/setup_env.sh [sft|selfevolve|eval|all]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/configs/paths.env"
WHICH="${1:-all}"
PYVER="${PYVER:-3.10}"

make_env () {  # name
  local name="$1"
  if ! conda env list | grep -qE "^${name}\s|/${name}$"; then
    echo "[setup] creating conda env ${name} (python ${PYVER})"
    conda create -y -n "${name}" "python=${PYVER}"
  fi
}

if [[ "${WHICH}" == "sft" || "${WHICH}" == "all" ]]; then
  # SFT uses LLaMA-Factory. Clone + install it as an upstream dependency.
  LF_DIR="${VERSE_ROOT}/third_party/LLaMA-Factory"
  if [[ ! -d "${LF_DIR}" ]]; then
    mkdir -p "${VERSE_ROOT}/third_party"
    git clone https://gh-proxy.com/https://github.com/hiyouga/LLaMA-Factory.git "${LF_DIR}"
  fi
  make_env verse-sft
  conda run -n verse-sft pip install -e "${LF_DIR}[torch,metrics,deepspeed,qwen]"
  echo "[setup] register VERSE SFT datasets: copy sft/dataset_info.json into your data dir (download_data.sh does this)."
fi

if [[ "${WHICH}" == "selfevolve" || "${WHICH}" == "all" ]]; then
  make_env verse
  conda run -n verse pip install -r "${HERE}/requirements/requirements-selfevolve.txt"
  # editable install of the bundled EasyR1 trainer
  conda run -n verse pip install -e "${HERE}/self_evolve/easy_r1" || true
fi

if [[ "${WHICH}" == "eval" || "${WHICH}" == "all" ]]; then
  make_env verse-eval
  conda run -n verse-eval pip install -r "${HERE}/requirements/requirements-eval.txt"
fi

echo "[setup] done. Envs: verse-sft / verse / verse-eval"
