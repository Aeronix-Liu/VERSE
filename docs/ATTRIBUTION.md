# Attribution & Licenses

VERSE builds on several open-source projects. We keep their names and credit
them explicitly; only our own internal project branding was renamed to VERSE.

## Upstream dependencies

- **ViLaSR** — *Reinforcing Spatial Reasoning in Vision-Language Models with
  Interwoven Thinking and Visual Drawing* (NeurIPS 2025). We build on ViLaSR's
  agentic drawing tools (Object Mapper / Path Tracer), its cold-start and
  reflective-rejection-sampling SFT data, and its evaluation harness.
  The bundled `self_evolve/` trainer and `eval/` code derive from ViLaSR; SFT
  images are ViLaSR's public datasets and are **not** re-hosted here.
- **EasyR1 / verl** — the RL training framework underlying the self-evolve GRPO
  stage (`self_evolve/easy_r1`).
- **LLaMA-Factory** — used for the cold-start and reflective SFT stages
  (`sft/`). Installed as a third-party dependency by `scripts/setup_env.sh`.
- **Qwen2.5-VL-7B-Instruct** (Alibaba) — the base vision-language model.
- **Benchmarks**: VSI-Bench, SPAR-Bench, and the MAZE navigation benchmark.

## What is ours (renamed to VERSE)

- The self-evolve two-round loop (proposer/solver), the deterministic
  ground-truth engine (DGE) under `self_evolve/src/`, the structured-imitate
  proposer, the engine-verified process rewards (maze path fidelity, VSI object
  grounding), the data/schedule construction, and all top-level `scripts/`.

## Licenses

Each upstream component retains its original license (see the respective
upstream repositories). Please consult ViLaSR, verl/EasyR1, LLaMA-Factory, and
Qwen2.5-VL for terms before redistribution. This repository's own additions are
released under the MIT License at the repository root. Bundled upstream code
retains its original terms; see also NOTICE and self_evolve/easy_r1/LICENSE.
