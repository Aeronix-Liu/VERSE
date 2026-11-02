#!/usr/bin/env python3
"""Upload VERSE release artifacts to the HuggingFace Hub (maintainers only).

Uploads target the REAL hub (https://huggingface.co) — NOT the read-only
hf-mirror. Authenticate first with your own token:
    export HF_TOKEN=hf_xxx            # a WRITE token for the target namespace
    # or: hf auth login

What it can upload (choose with --what):
  sft         cleaned SFT annotations (*_clean.json) + dataset_info.json  -> <ns>/VERSE-data
  context     self-evolve context (ScanNet frames + metadata, maze pool,
              manifest jsonl, schedule)                                   -> <ns>/VERSE-selfevolve-context
  cold_start  cold-start SFT checkpoint (model-only)                       -> <ns>/VERSE-cold-start-7B
  rrs         RRS SFT checkpoint (model-only)                             -> <ns>/VERSE-rrs-7B

Examples:
    python upload/upload_to_hf.py --what sft --src /path/to/ViLaSR-data --namespace Aeronix-zzz
    python upload/upload_to_hf.py --what cold_start --src /path/to/vilasr_cold_start_4gpu
    python upload/upload_to_hf.py --what sft --dry-run     # list what would upload
"""
import argparse
import os
import sys

# Force real hub for uploads even if HF_ENDPOINT points at a mirror.
os.environ["HF_ENDPOINT"] = os.environ.get("HF_UPLOAD_ENDPOINT", "https://huggingface.co")

from huggingface_hub import HfApi  # noqa: E402

# Model-only allowlist: weights + config/tokenizer, no optimizer/trainer state.
MODEL_ALLOW = [
    "*.safetensors", "*.json", "*.jinja", "*.txt", "*.model",
    "tokenizer*", "*preprocessor*", "generation_config*", "chat_template*",
]
# SFT annotations: only the cleaned sharegpt json + registry (NOT the images).
SFT_ALLOW = ["**/*_clean.json", "dataset_info.json", "maze_calibration/*.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["sft", "context", "cold_start", "rrs"])
    ap.add_argument("--src", required=True, help="local source dir to upload from")
    ap.add_argument("--namespace", default=os.environ.get("VERSE_HF_NAMESPACE", "Aeronix-zzz"))
    ap.add_argument("--repo", default=None, help="override full repo id (namespace/name)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = {
        "sft":        ("VERSE-data",                 "dataset", SFT_ALLOW),
        "context":    ("VERSE-selfevolve-context",   "dataset", None),
        "cold_start": ("VERSE-cold-start-7B",        "model",   MODEL_ALLOW),
        "rrs":        ("VERSE-rrs-7B",               "model",   MODEL_ALLOW),
    }
    name, repo_type, allow = cfg[args.what]
    repo_id = args.repo or f"{args.namespace}/{name}"
    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        sys.exit(f"source dir not found: {src}")

    print(f"[upload] what={args.what} repo={repo_id} ({repo_type}) src={src}")
    print(f"[upload] endpoint={os.environ['HF_ENDPOINT']}  allow_patterns={allow}")
    if args.dry_run:
        print("[upload] DRY RUN — no network calls made.")
        return

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type=repo_type, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=src,
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow,
        ignore_patterns=["**/optimizer*", "**/*.pt", "**/global_step*", "**/.tokenized*", "**/__pycache__/**"],
    )
    print(f"[upload] done -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
