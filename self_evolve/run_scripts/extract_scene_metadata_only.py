#!/usr/bin/env python3
"""Extract SCENE-LEVEL metadata only (objects + 3d_bbox + room_size).

This is a subset of metadata_extractor_small_mesh.py that does NOT require
per-frame color/depth/pose files. It works on mesh-only scenes (vh_clean_2.ply
+ segs + aggregation), producing the scene-level JSON that DGE needs for
object_counting, room_size, and world-frame absolute_distance tasks.

Usage:
    python extract_scene_metadata_only.py [--dataset-root DIR] [--workers N]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNET_PROCESS_ROOT = REPO_ROOT / "data" / "scannet_process"
if str(SCANNET_PROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCANNET_PROCESS_ROOT))

METADATA_EXTRACTOR = SCANNET_PROCESS_ROOT / "data_scripts" / "metadata_extractor.py"
import importlib.util

_spec = importlib.util.spec_from_file_location("metadata_extractor", METADATA_EXTRACTOR)
_me = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_me)

from utils.bbox_extractor import (  # type: ignore
    align_mesh_vertices,
    export_3d_bbox,
    get_object_id_to_label_id,
    read_label_mapping,
    read_mesh_vertices_rgb,
    read_scene_axis_alignment,
)
from utils.metadata_builder import export_scene_metadata  # type: ignore
from utils.room_size_extractor import calculate_room_area  # type: ignore

DEFAULT_DATASET_ROOT = REPO_ROOT / "ScanNetData_local"
DEFAULT_LABEL_MAP_FILE = REPO_ROOT / "data/scannet_process/scannet_metadata/scannetv2-labels-v2.tsv"


def scene_mesh_paths(scene_dir: Path, scene_id: str):
    return (
        scene_dir / f"{scene_id}_vh_clean_2.ply",
        scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json",
        scene_dir / f"{scene_id}_vh_clean.aggregation.json",
    )


def process_scene_scene_only(scene_id: str, scene_root: Path, metadata_root: Path,
                             raw_id_map, id_label_map) -> dict:
    scene_dir = scene_root / scene_id
    ply_file, seg_file, agg_file = scene_mesh_paths(scene_dir, scene_id)

    if not ply_file.is_file() or ply_file.stat().st_size < 1000:
        return {"scene_id": scene_id, "status": "skip", "reason": "no mesh"}
    if not seg_file.is_file() or seg_file.stat().st_size < 1:
        return {"scene_id": scene_id, "status": "skip", "reason": "no segs"}

    out_dir = metadata_root / scene_id
    out_file = out_dir / f"{scene_id}.json"
    if out_file.exists():
        return {"scene_id": scene_id, "status": "exists"}

    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_mesh_vertices, object_id_to_segs, instance_ids = align_mesh_vertices(
        str(ply_file), str(seg_file), str(agg_file), instance_id_to_label_id_map=None,
        raw_id_map=raw_id_map, id_label_map=id_label_map, str_scene_dir=str(scene_dir),
        visualize=False,
    )
    room_size = calculate_room_area(aligned_mesh_vertices, visualize=False)
    bboxes_3d = export_3d_bbox(
        aligned_mesh_vertices, object_id_to_segs, instance_ids, id_label_map,
        str(scene_dir), visualize=False,
    )
    scene_metadata = export_scene_metadata(scene_id, bboxes_3d, room_size)
    out_file.write_text(json.dumps(scene_metadata, ensure_ascii=False, indent=4), encoding="utf-8")
    return {"scene_id": scene_id, "status": "done", "objects": len(scene_metadata.get("objects", []))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    ap.add_argument("--label-map-file", default=str(DEFAULT_LABEL_MAP_FILE))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scene_root = Path(args.dataset_root) / "train"
    metadata_root = Path(args.dataset_root) / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    raw_id_map, id_label_map = read_label_mapping(args.label_map_file)

    # Find all scenes with valid mesh
    targets = []
    for d in sorted(scene_root.iterdir()):
        if d.is_dir() and d.name.startswith("scene"):
            ply, segs, _ = scene_mesh_paths(d, d.name)
            if ply.is_file() and ply.stat().st_size > 1000 and segs.is_file():
                targets.append(d.name)

    print(f"[scene-only] {len(targets)} scenes with mesh; extracting scene-level metadata")
    ok = skip = exist = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_scene_scene_only, s, scene_root, metadata_root,
                          raw_id_map, id_label_map): s for s in targets}
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            st = r["status"]
            if st == "done":
                ok += 1
                print(f"  done {r['scene_id']}: {r.get('objects', 0)} objects")
            elif st == "exists":
                exist += 1
            else:
                skip += 1
                print(f"  {st} {r['scene_id']}: {r.get('reason', '')}")

    print(f"\n[scene-only] complete: {ok} new, {exist} existing, {skip} skipped")
    print(f"metadata dir: {metadata_root} ({len(list(metadata_root.iterdir()))} scene dirs)")


if __name__ == "__main__":
    main()
