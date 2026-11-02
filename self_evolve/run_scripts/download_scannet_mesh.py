#!/usr/bin/env python3
"""Download mesh files for new ScanNet scenes from kaldir (publicly accessible).

Each scene needs 3 files:
  - {sid}_vh_clean_2.ply            (~5-12 MB, the mesh)
  - {sid}_vh_clean_2.0.010000.segs.json  (~1-2 MB, segmentation)
  - {sid}_vh_clean.aggregation.json     (~5-50 KB, object instances)

Total per scene: ~8 MB. For 150 scenes: ~1.2 GB.

Usage:
    python download_scannet_mesh.py --n-scenes 150
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCAN_BASE_URL = "https://kaldir.vc.in.tum.de/scannet/v2/scans"
UA = {"User-Agent": "ViLaSR-SelfEvolve/1.0"}

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "ScanNetData_local"
TRAIN_ROOT = DATA_ROOT / "train"
CHOSEN_SCENES_FILE = DATA_ROOT / "downloads" / "chosen_scenes.txt"
READY_SCENES_FILE = DATA_ROOT / "downloads" / "scenes_ready_for_metadata.txt"

# All 1513 ScanNet v2 train scene IDs (we'll fetch from the zip or hardcode a subset)
# For now, use the SR-91k scenes as candidates (they have frames already)
SR91K_ROOT = Path("/data/pengtianshuo/self_evolve/Xinran/ViLaSR/ViLaSR-data/rl/SR_91k")


def get_existing_scenes() -> set[str]:
    """Scenes that already have mesh + frames."""
    scenes = set()
    if TRAIN_ROOT.exists():
        for d in TRAIN_ROOT.iterdir():
            if d.is_dir() and d.name.startswith("scene"):
                ply = d / f"{d.name}_vh_clean_2.ply"
                if ply.exists() and ply.stat().st_size > 1000:
                    scenes.add(d.name)
    return scenes


def get_candidate_scenes() -> list[str]:
    """Get scene IDs from SR-91k (which has 1201 scenes with frames)."""
    if not SR91K_ROOT.exists():
        return []
    scenes = sorted(d.name for d in SR91K_ROOT.iterdir() if d.is_dir() and d.name.startswith("scene"))
    return scenes


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) < 100:
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True
        except Exception as exc:
            if attempt == retries - 1:
                return False
    return False


def download_scene_mesh(scene_id: str) -> tuple[bool, str]:
    """Download mesh files for a single scene."""
    scene_dir = TRAIN_ROOT / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    files = [
        (f"{scene_id}_vh_clean_2.ply", f"{scene_id}_vh_clean_2.ply"),
        (f"{scene_id}_vh_clean_2.0.010000.segs.json", f"{scene_id}_vh_clean_2.0.010000.segs.json"),
        (f"{scene_id}_vh_clean.aggregation.json", f"{scene_id}_vh_clean.aggregation.json"),
    ]

    missing = []
    for remote_name, local_name in files:
        dest = scene_dir / local_name
        if dest.exists() and dest.stat().st_size > 100:
            continue
        url = f"{SCAN_BASE_URL}/{scene_id}/{remote_name}"
        if not download_file(url, dest):
            missing.append(remote_name)

    # Check if mesh is valid
    ply = scene_dir / f"{scene_id}_vh_clean_2.ply"
    if not ply.exists() or ply.stat().st_size < 1000:
        return False, f"mesh missing or too small"

    # Create aggregation symlink if needed
    agg_link = scene_dir / f"{scene_id}_vh_clean.aggregation.json"
    agg_real = scene_dir / f"{scene_id}.aggregation.json"
    if agg_link.exists() and agg_link.is_symlink():
        pass  # already symlinked
    elif agg_real.exists() and not agg_link.exists():
        os.symlink(agg_real.name, agg_link)

    return True, "" if not missing else f"missing: {missing}"


def main():
    parser = argparse.ArgumentParser(description="Download ScanNet mesh files")
    parser.add_argument("--n-scenes", type=int, default=150, help="Number of NEW scenes to download")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    args = parser.parse_args()

    existing = get_existing_scenes()
    print(f"[existing] {len(existing)} scenes already have mesh")

    candidates = get_candidate_scenes()
    print(f"[candidates] {len(candidates)} scenes available in SR-91k")

    new_scenes = [s for s in candidates if s not in existing][:args.n_scenes]
    print(f"[target] Downloading mesh for {len(new_scenes)} new scenes")

    # Download in parallel
    ok_count = 0
    fail_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_scene_mesh, sid): sid for sid in new_scenes}
        for i, future in enumerate(as_completed(futures)):
            sid = futures[future]
            try:
                ok, msg = future.result()
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    print(f"  FAIL: {sid} - {msg}")
            except Exception as exc:
                fail_count += 1
                print(f"  ERROR: {sid} - {exc}")
            if (i + 1) % 20 == 0:
                print(f"  progress: {i + 1}/{len(new_scenes)} ({ok_count} ok, {fail_count} fail)")

    print(f"\n=== Mesh Download Summary ===")
    print(f"  ok: {ok_count}")
    print(f"  fail: {fail_count}")
    print(f"  total with mesh: {len(existing) + ok_count}")

    # Update chosen_scenes.txt with all scenes that have mesh
    all_with_mesh = sorted(get_existing_scenes() | {sid for sid in new_scenes if (TRAIN_ROOT / sid / f"{sid}_vh_clean_2.ply").exists()})
    CHOSEN_SCENES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHOSEN_SCENES_FILE.write_text("\n".join(all_with_mesh) + "\n")
    print(f"  Updated {CHOSEN_SCENES_FILE} with {len(all_with_mesh)} scenes")

    print(f"\nNext: Run fetch_25k_partial.py to download frames for these scenes")


if __name__ == "__main__":
    main()
