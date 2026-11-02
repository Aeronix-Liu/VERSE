#!/usr/bin/env python3
"""Prepare the small ScanNet subset for ViLaSR-self-evolve (CPU-only).

Steps:
 1. Extract frames/pose/depth/intrinsics for the chosen scenes from
    scannet_frames_25k.zip into ScanNetData_local/train/<sid>/.
    Color frames go to frame_processed/ (the layout the DGE expects).
 2. Apply layout fixes (aggregation symlinks for the metadata extractor).
 3. Build the training contexts jsonl: filter spatialevo_train_local_cap4.jsonl
    to scenes with complete data, rewrite paths to this repo, merge with the
    synthetic maze contexts, and shuffle.

Run AFTER the aria2 downloads finish. Metadata extraction is launched
separately via run_scripts/run_metadata_extraction.sh (needs open3d env).
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "ScanNetData_local"
TRAIN_ROOT = DATA_ROOT / "train"
ZIP_PATH = DATA_ROOT / "downloads" / "scannet_frames_25k.zip"
CHOSEN_SCENES = DATA_ROOT / "downloads" / "chosen_scenes.txt"
CAP4_JSONL = REPO / "easy_r1" / "dataset" / "spatialevo_train_local_cap4.jsonl"
MAZE_JSONL = REPO / "dataset" / "maze_contexts.jsonl"
OUT_JSONL = REPO / "easy_r1" / "dataset" / "selfevolve_train.jsonl"
OLD_TRAIN_PREFIX = "/data/pengtianshuo/self_evolve/Xinran/VERSE/ScanNetData/train"


def extract_frames(chosen: set[str]) -> None:
    extracted = 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            scene_id = next((p for p in parts if p.startswith("scene")), None)
            if scene_id not in chosen:
                continue
            rel = Path(*[p for p in parts if p != "scannet_frames_25k"])
            if name.endswith(".jpg") and "/color/" in name:
                dest = TRAIN_ROOT / scene_id / "frame_processed" / rel.name
            elif "/pose/" in name and name.endswith(".txt"):
                dest = TRAIN_ROOT / scene_id / "pose" / rel.name
            elif "/depth/" in name and name.endswith(".png"):
                dest = TRAIN_ROOT / scene_id / "depth" / rel.name
            elif "intrinsic" in name and name.endswith(".txt"):
                new_name = rel.name.replace("intrinsics_color", "intrinsic_color").replace(
                    "intrinsics_depth", "intrinsic_depth"
                )
                dest = TRAIN_ROOT / scene_id / new_name
            else:
                continue
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, dest.open("wb") as out:
                out.write(src.read())
            extracted += 1
    print(f"Extracted {extracted} files from {ZIP_PATH.name}")


def fix_layout(chosen: set[str]) -> None:
    for sid in sorted(chosen):
        scene_dir = TRAIN_ROOT / sid
        if not scene_dir.is_dir():
            continue
        agg = scene_dir / f"{sid}.aggregation.json"
        agg_link = scene_dir / f"{sid}_vh_clean.aggregation.json"
        if agg.exists() and not agg_link.exists():
            agg_link.symlink_to(agg.name)


def scene_ready(sid: str) -> bool:
    scene_dir = TRAIN_ROOT / sid
    return (
        (scene_dir / f"{sid}_vh_clean_2.ply").exists()
        and (scene_dir / f"{sid}_vh_clean_2.0.010000.segs.json").exists()
        and (scene_dir / f"{sid}.aggregation.json").exists()
        and (scene_dir / "frame_processed").is_dir()
        and (scene_dir / "pose").is_dir()
    )


def metadata_ready(sid: str) -> bool:
    meta_dir = DATA_ROOT / "metadata" / sid
    return meta_dir.is_dir() and (meta_dir / f"{sid}.json").exists()


def build_jsonl(chosen: set[str], require_metadata: bool, seed: int = 20260724) -> None:
    usable = {sid for sid in chosen if scene_ready(sid) and (not require_metadata or metadata_ready(sid))}
    print(f"Usable scenes: {len(usable)}/{len(chosen)}: {sorted(usable)}")

    records = []
    kept = dropped = 0
    with CAP4_JSONL.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            new_images = []
            ok = True
            for raw in row.get("images", []):
                if not raw.startswith(OLD_TRAIN_PREFIX):
                    ok = False
                    break
                rel = Path(raw).relative_to(OLD_TRAIN_PREFIX)
                sid = rel.parts[0]
                if sid not in usable:
                    ok = False
                    break
                new_path = TRAIN_ROOT / rel
                if not new_path.exists():
                    ok = False
                    break
                new_images.append(str(new_path))
            if ok and new_images:
                records.append({"problem": None, "answer": None, "images": new_images})
                kept += 1
            else:
                dropped += 1
    print(f"ScanNet contexts: kept {kept}, dropped {dropped}")

    maze_records = []
    if MAZE_JSONL.exists():
        with MAZE_JSONL.open(encoding="utf-8") as f:
            maze_records = [json.loads(line) for line in f]
    print(f"Maze contexts: {len(maze_records)}")

    all_records = records + maze_records
    random.Random(seed).shuffle(all_records)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_records)} contexts -> {OUT_JSONL}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--require-metadata", action="store_true",
                        help="Only include scenes whose DGE metadata has been extracted.")
    args = parser.parse_args()

    chosen = {line.strip() for line in CHOSEN_SCENES.read_text().splitlines() if line.strip()}
    print(f"Chosen scenes: {len(chosen)}")

    if not args.skip_extract:
        extract_frames(chosen)
    fix_layout(chosen)
    build_jsonl(chosen, require_metadata=args.require_metadata)


if __name__ == "__main__":
    main()
