#!/usr/bin/env python3
"""Directly test the DGE (WorldSimulator) for a SPAR distance_prediction_oc task."""
import os, sys, json
sys.path.insert(0, "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve")
from src.simulator.world_simulator import WorldSimulator

SCENE_ROOT = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/ScanNetData_local/train"
META_DIR = "/data/pengtianshuo/self_evolve/Xinran/ViLaSR-self-evolve/ScanNetData_local/metadata"

# pick first scene with a frame
import glob
scenes = sorted(d for d in os.listdir(SCENE_ROOT) if d.startswith("scene"))
scene_id = scenes[0]
frame_dir = os.path.join(SCENE_ROOT, scene_id, "frame_processed")
frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
frame_file = frames[len(frames)//2]
frame_id = frame_file.replace(".jpg","")
img_path = os.path.join(frame_dir, frame_file)
print(f"scene_id={scene_id} frame_id={frame_id} img={img_path}")

sim = WorldSimulator(scannet_root=SCENE_ROOT, enable_vlm=False)
# list a few object labels from metadata to craft a question
meta_file = os.path.join(META_DIR, scene_id, scene_id+".json")
meta = json.load(open(meta_file))
objs = meta.get("objects", [])
print(f"num objects in metadata: {len(objs)}")
print("first 5 objects:", [o.get("label") or o.get("name") for o in objs[:5]])

# Build a query task
query = {
    "task_type": "spar_distance_prediction_oc",
    "scene_id": scene_id,
    "metadata_dir": META_DIR,
    "image_paths": [img_path],
    "image_path": img_path,
    "question": "What is the distance from the camera to the cabinet in meters?",
    "object_labels": ["cabinet"],
    "mode": "oc",
}
print("=== calling validate_and_answer ===")
try:
    res = sim.validate_and_answer(query)
    # trim large fields
    for k in list(res.keys()):
        v = res[k]
        s = json.dumps(v) if not isinstance(v,str) else v
        if len(s) > 300: s = s[:300]+"..."
        print(f"  {k}: {s}")
except Exception as e:
    import traceback; traceback.print_exc()
