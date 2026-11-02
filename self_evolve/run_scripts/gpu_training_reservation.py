#!/usr/bin/env python3
"""Hold >1GB VRAM per visible GPU so gpu_guardian treats them as user jobs (30min grace).

Start after `gpu_guardian.py --stop`, stop once real training allocates GPU memory.
Does NOT kill guardian — only prevents empty-GPU re-occupy during slow CPU bootstrap.
"""
from __future__ import annotations

import os
import signal
import sys
import time

import torch

_STOP = False


def _handle(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True


def main() -> int:
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    if not torch.cuda.is_available():
        print("[gpu_reservation] CUDA unavailable, exiting", flush=True)
        return 1

    n = torch.cuda.device_count()
    tensors = []
    # > USER_MEM_THRESHOLD_MB (1024) in gpu_guardian.py
    mb_per_gpu = int(os.environ.get("GPU_RESERVE_MB", "1100"))
    n_elems = (mb_per_gpu * 1024 * 1024) // 4  # float32
    for i in range(n):
        torch.cuda.set_device(i)
        t = torch.empty(n_elems, dtype=torch.float32, device=i)
        t.fill_(1.0)
        tensors.append(t)
        print(f"[gpu_reservation] reserved ~{mb_per_gpu}MB on cuda:{i}", flush=True)

    print(f"[gpu_reservation] holding {n} GPU(s), pid={os.getpid()}", flush=True)
    while not _STOP:
        time.sleep(5)
    print("[gpu_reservation] released", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
