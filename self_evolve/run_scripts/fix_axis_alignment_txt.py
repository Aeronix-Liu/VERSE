#!/usr/bin/env python3
"""Download the missing {scene_id}.txt axis-alignment file for all mesh-equipped scenes."""
from __future__ import annotations
import urllib.request, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SCAN_BASE_URL = "https://kaldir.vc.in.tum.de/scannet/v2/scans"
UA = {"User-Agent": "ViLaSR-SelfEvolve/1.0"}
REPO = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO / "ScanNetData_local" / "train"


def fetch_txt(scene_id: str) -> tuple[str, bool, str]:
    dest = TRAIN_ROOT / scene_id / f"{scene_id}.txt"
    if dest.exists() and dest.stat().st_size > 10:
        return scene_id, True, "exists"
    url = f"{SCAN_BASE_URL}/{scene_id}/{scene_id}.txt"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 10:
            return scene_id, False, "too small"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return scene_id, True, "downloaded"
    except Exception as exc:
        return scene_id, False, str(exc)[:80]


def main():
    scenes = sorted(d.name for d in TRAIN_ROOT.iterdir()
                    if d.is_dir() and d.name.startswith("scene"))
    print(f"[fix-txt] {len(scenes)} scenes; checking for missing .txt axis-alignment files")
    ok = fail = exist = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(fetch_txt, s): s for s in scenes}
        for f in as_completed(futs):
            sid, success, msg = f.result()
            if success and msg == "downloaded":
                ok += 1
            elif success and msg == "exists":
                exist += 1
            else:
                fail += 1
                print(f"  FAIL {sid}: {msg}")
    print(f"\n[fix-txt] done: {ok} downloaded, {exist} already present, {fail} failed")


if __name__ == "__main__":
    main()
