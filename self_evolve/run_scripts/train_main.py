#!/usr/bin/env python3
"""Launch VERSE verl training with flash-attn disabled for broken vLLM ViT kernels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _disable_broken_flash_attn() -> None:
    try:
        import vllm.attention.layer as layer

        layer.check_upstream_fa_availability = lambda dtype: False
    except Exception:
        pass


def _patch_ray_init() -> None:
    import ray

    original_init = ray.init

    def patched_init(*args, **kwargs):
        runtime_env = dict(kwargs.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars.setdefault(
            "VLLM_ATTENTION_BACKEND",
            os.environ.get("VLLM_ATTENTION_BACKEND", "TORCH_SDPA"),
        )
        env_vars.setdefault(
            "SPATIALEVO_ATTN_IMPLEMENTATION",
            os.environ.get("SPATIALEVO_ATTN_IMPLEMENTATION", "sdpa"),
        )
        # Keep verl default (expandable_segments:False) — True breaks vLLM memory pool.
        runtime_env["env_vars"] = env_vars
        runtime_env["worker_process_setup_hook"] = "vllm_worker_setup_hook.setup_worker"
        kwargs["runtime_env"] = runtime_env
        return original_init(*args, **kwargs)

    ray.init = patched_init


_disable_broken_flash_attn()
_patch_ray_init()

from verl.trainer.main import main

if __name__ == "__main__":
    main()
