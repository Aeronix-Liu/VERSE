#!/usr/bin/env python3
"""Fetch ONLY the chosen scenes from scannet_frames_25k.zip via HTTP Range requests.

The full zip is 5.6GB; we need ~30 scenes (~150MB). This reads the zip central
directory remotely, then downloads + decompresses just the members we need:
    color/*.jpg  -> train/<sid>/frame_processed/
    pose/*.txt   -> train/<sid>/pose/
    depth/*.png  -> train/<sid>/depth/
    intrinsics_* -> train/<sid>/intrinsic_*.txt
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://kaldir.vc.in.tum.de/scannet/v2/tasks/scannet_frames_25k.zip"
REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "ScanNetData_local"
TRAIN_ROOT = DATA_ROOT / "train"
CHOSEN_SCENES = DATA_ROOT / "downloads" / "chosen_scenes.txt"
UA = {"User-Agent": "ViLaSR-SelfEvolve/1.0"}


def fetch_range(start: int, length: int, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, headers={**UA, "Range": f"bytes={start}-{start + length - 1}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"range fetch retry {attempt + 1}: {exc}", file=sys.stderr)
    raise RuntimeError("unreachable")


def get_total_size() -> int:
    req = urllib.request.Request(URL, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


class RangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range (used ONLY for the central directory)."""

    def __init__(self, size: int):
        self._size = size
        self._pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        n = min(n, self._size - self._pos)
        if n <= 0:
            return b""
        data = fetch_range(self._pos, n)
        self._pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def dest_for_member(name: str, chosen: set[str]) -> Path | None:
    parts = Path(name).parts
    scene_id = next((p for p in parts if p.startswith("scene")), None)
    if scene_id not in chosen:
        return None
    fname = Path(name).name
    if "/color/" in name and name.endswith(".jpg"):
        return TRAIN_ROOT / scene_id / "frame_processed" / fname
    if "/pose/" in name and name.endswith(".txt"):
        return TRAIN_ROOT / scene_id / "pose" / fname
    if "/depth/" in name and name.endswith(".png"):
        return TRAIN_ROOT / scene_id / "depth" / fname
    if "intrinsic" in fname and name.endswith(".txt"):
        fixed = fname.replace("intrinsics_color", "intrinsic_color").replace("intrinsics_depth", "intrinsic_depth")
        return TRAIN_ROOT / scene_id / fixed
    return None


def download_member(info: zipfile.ZipInfo, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    # Local file header: 30 bytes fixed + name + extra (lengths from the LOCAL header).
    header = fetch_range(info.header_offset, 30)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    data_start = info.header_offset + 30 + name_len + extra_len
    payload = fetch_range(data_start, info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        data = payload
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(payload)
    else:
        raise RuntimeError(f"unsupported compression {info.compress_type} for {info.filename}")
    if len(data) != info.file_size:
        raise RuntimeError(f"size mismatch for {info.filename}: {len(data)} != {info.file_size}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    chosen = {line.strip() for line in CHOSEN_SCENES.read_text().splitlines() if line.strip()}
    total_size = get_total_size()
    print(f"zip size: {total_size / 1e9:.2f} GB; chosen scenes: {len(chosen)}")

    print("reading central directory (remote)...")
    zf = zipfile.ZipFile(io.BufferedReader(RangeFile(total_size), buffer_size=4 * 1024 * 1024))
    jobs = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        dest = dest_for_member(info.filename, chosen)
        if dest is not None:
            jobs.append((info, dest))
    total_bytes = sum(info.compress_size for info, _ in jobs)
    print(f"members to fetch: {len(jobs)} ({total_bytes / 1e6:.1f} MB compressed)")

    done = 0
    fetched = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_member, info, dest): (info, dest) for info, dest in jobs}
        for future in as_completed(futures):
            info, dest = futures[future]
            try:
                fetched += future.result()
            except Exception as exc:
                print(f"FAILED {info.filename}: {exc}", file=sys.stderr)
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(jobs)} members, {fetched / 1e6:.1f} MB written")
    print(f"DONE: {done}/{len(jobs)} members, {fetched / 1e6:.1f} MB written")


if __name__ == "__main__":
    main()
