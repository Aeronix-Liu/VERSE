"""ScanNet data loader."""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional

from collections import Counter


# Synonym / alias map: GRPO question wording -> canonical ScanNet object label.
# Built from the DGE x GRPO compatibility audit (see
# cursor_memory/2026-07-29_dge_grpo_compat_report.md) so that questions asking
# about "couch" / "doorframe" / "refrigerator" / "nightstand" etc. resolve to the
# labels actually stored in the scene metadata.
LABEL_ALIASES: Dict[str, str] = {
    "couch": "sofa",
    "sofas": "sofa",
    "doorframe": "door",
    "doors": "door",
    "nightstand": "night stand",
    "nightstands": "night stand",
    "refrigerator": "refridgerator",  # ScanNet metadata uses this spelling
    "refrigerators": "refridgerator",
    "coffee table": "table",
    "coffee tables": "table",
    "desk": "table",
    "desks": "table",
    "cabinet": "closet",
    "cabinets": "closet",
    "shelf": "bookshelf",
    "shelves": "bookshelf",
    "bookshelves": "bookshelf",
    "rubbish bin": "trash can",
    "garbage can": "trash can",
    "trash bin": "trash can",
    "wastebasket": "trash can",
    "bin": "trash can",
    "tv": "tv",
    "tvs": "tv",
    "television": "tv",
    "televisions": "tv",
    "toilets": "toilet",
    "pillows": "pillow",
    "chairs": "chair",
    "tables": "table",
    "windows": "window",
    "monitors": "monitor",
    "lamps": "lamp",
    "beds": "bed",
    "plants": "plant",
    "stools": "stool",
    "mirrors": "mirror",
    "keyboards": "keyboard",
    "telephones": "telephone",
    "suitcases": "suitcase",
    "baskets": "basket",
    "buckets": "bucket",
    "clocks": "clock",
    "bowls": "bowl",
    "cups": "cup",
    "ovens": "oven",
    "fans": "fan",
    "guitars": "guitar",
    "laptops": "laptop",
    "blankets": "blanket",
    "kettles": "kettle",
    "heaters": "heater",
    "dishwashers": "dishwasher",
    "fireplaces": "fireplace",
    "crates": "crate",
    "backpacks": "backpack",
    "towels": "towel",
    "radiators": "radiator",
    "counters": "counter",
    "microwaves": "microwave",
    "printers": "printer",
    "pianos": "piano",
    "cushions": "cushion",
}


def _canonicalize_label(label: str) -> str:
    """Map a (possibly GRPO-style) label to its canonical ScanNet form."""
    if not isinstance(label, str):
        return label
    key = label.strip().lower()
    return LABEL_ALIASES.get(key, key)


class ScanNetLoader:
    """Load scene metadata, frame metadata, and camera data."""

    def __init__(
        self,
        scannet_root: str = "/mnt/jfs/lidingm/data/dataset/ScanNet/train",
        visibility_floor: float = 0.1
    ):
        """Initialize the loader."""
        self.scannet_root = scannet_root
        self.visibility_floor = float(visibility_floor)
        self.cache = {}

    def load_scene_metadata(self, scene_id: str, metadata_dir: str = None) -> Dict[str, Any]:
        """Load scene-level metadata."""
        cache_key = f"scene_{scene_id}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if metadata_dir is None:
            metadata_dir = os.path.join(self.scannet_root, "..", "metadata")

        metadata_file = os.path.join(metadata_dir, scene_id, f"{scene_id}.json")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Scene metadata not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        self.cache[cache_key] = metadata
        return metadata

    def load_frame_metadata(
        self,
        scene_id: str,
        frame_id: str,
        frame_type: str = "frame_processed",
        metadata_dir: str = None
    ) -> Dict[str, Any]:
        """Load frame-level metadata."""
        if isinstance(frame_id, int):
            frame_id = str(frame_id)

        # Normalize the frame id and strip the image suffix when needed.
        frame_id_num = frame_id[:-4] if frame_id.endswith('.jpg') else frame_id

        cache_key = f"frame_{scene_id}_{frame_id_num}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if metadata_dir is None:
            metadata_dir = os.path.join(self.scannet_root, "..", "metadata")

        frame_metadata_file = os.path.join(
            metadata_dir, scene_id, frame_type, f"{frame_id_num}.json"
        )

        if not os.path.exists(frame_metadata_file):
            raise FileNotFoundError(f"Frame metadata not found: {frame_metadata_file}")

        with open(frame_metadata_file, 'r') as f:
            metadata = json.load(f)

        self.cache[cache_key] = metadata
        return metadata

    def load_camera_intrinsics(self, scene_id: str, camera_type: str = "color") -> np.ndarray:
        """Load camera intrinsics."""
        intrinsic_file = os.path.join(
            self.scannet_root, scene_id, f"intrinsic_{camera_type}.txt"
        )

        if not os.path.exists(intrinsic_file):
            raise FileNotFoundError(f"Camera intrinsics not found: {intrinsic_file}")

        return np.loadtxt(intrinsic_file)

    def load_camera_pose(self, scene_id: str, frame_id: int) -> np.ndarray:
        """Load a camera pose."""
        pose_file = os.path.join(
            self.scannet_root, scene_id, "pose", f"{frame_id}.txt"
        )

        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"Camera pose not found: {pose_file}")

        return np.loadtxt(pose_file)

    @staticmethod
    def _match_label(obj_label: str, query_label: str) -> bool:
        """Match an object label against a query, tolerating synonyms/plurals.

        Tries (1) exact, (2) alias map, (3) safe plural strip — but only strips
        a trailing 's' when the singular form matches and the plural does not,
        so canonical plurals like "shoes"/"headphones" are preserved.
        """
        if not isinstance(query_label, str) or not isinstance(obj_label, str):
            return obj_label == query_label
        o = obj_label.lower()
        q = query_label.strip().lower()
        if o == q:
            return True
        if _canonicalize_label(obj_label) == _canonicalize_label(query_label):
            return True
        # safe plural strip: query plural -> singular matching the object
        if q.endswith("s") and not q.endswith("ss"):
            qs = q[:-1]
            if o == qs and o != q:
                return True
        return False

    def get_object_by_label(
        self,
        scene_metadata: Dict[str, Any],
        label: str,
        check_uniqueness: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Return the object with the requested label."""
        objects = scene_metadata.get("objects", [])
        matching_objects = [obj for obj in objects if self._match_label(obj["label"], label)]

        if len(matching_objects) == 0:
            return None

        if check_uniqueness and len(matching_objects) > 1:
            return None

        return matching_objects[0]

    def count_objects_by_label(
        self,
        scene_metadata: Dict[str, Any],
        label: str
    ) -> int:
        """Count objects with the requested label."""
        objects = scene_metadata.get("objects", [])
        return sum(1 for obj in objects if self._match_label(obj["label"], label))

    def get_scene_labels(
        self,
        scene_id: str,
        unique: bool = True,
        sort: bool = True,
        lowercase: bool = True,
        metadata_dir: Optional[str] = None,
        annotate_uniqueness: bool = False,
    ) -> List[str]:
        """Return object labels for the scene."""
        scene_meta = self.load_scene_metadata(scene_id, metadata_dir=metadata_dir)
        objects = scene_meta.get("objects", []) or []

        # Collect raw labels, including duplicates.
        all_labels = []
        for obj in objects:
            lbl = obj.get("label")
            if not isinstance(lbl, str):
                continue
            all_labels.append(lbl.lower() if lowercase else lbl)

        if annotate_uniqueness:
            # Count occurrences and return deduplicated labels with a suffix.
            label_counts = Counter(all_labels)
            annotated_labels = []
            unique_keys = label_counts.keys()

            if sort:
                unique_keys = sorted(unique_keys)

            for label in unique_keys:
                count = label_counts[label]
                suffix = "(Unique)" if count == 1 else "(Non-Unique)"
                annotated_labels.append(f"{label}{suffix}")

            return annotated_labels

        if unique:
            if sort:
                return sorted(set(all_labels))
            # Preserve first-seen order when sorting is disabled.
            seen = set()
            deduped = []
            for label in all_labels:
                if label not in seen:
                    seen.add(label)
                    deduped.append(label)
            return deduped

        return sorted(all_labels) if sort else all_labels

    def get_visible_objects(
        self,
        frame_metadata: Dict[str, Any],
        min_visibility: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Return visible objects from a frame."""
        min_visibility = max(self.visibility_floor, float(min_visibility))
        objects = frame_metadata.get("objects", [])
        return [obj for obj in objects if obj.get("visibility", 0) >= min_visibility]

    def check_label_ambiguity(
        self,
        scene_metadata: Dict[str, Any],
        labels: List[str],
        min_visibility: float = 0.5,
        frame_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, bool]:
        """Check whether each label is unique."""
        result = {}

        for label in labels:
            if frame_metadata:
                # Use frame-visible objects when frame metadata is available.
                visible_objects = self.get_visible_objects(frame_metadata, min_visibility)
                count = sum(1 for obj in visible_objects if self._match_label(obj["label"], label))
            else:
                # Otherwise use the full scene inventory.
                count = self.count_objects_by_label(scene_metadata, label)

            result[label] = (count == 1)

        return result

    def get_image_path(
        self,
        scene_id: str,
        frame_id: int,
        frame_type: str = "frame_processed"
    ) -> str:
        """Return an image path for the requested frame."""
        if frame_type in {"frame_processed", "color"}:
            ext = ".jpg"
        elif frame_type == "depth":
            ext = ".png"
        else:
            ext = ".jpg"

        return os.path.join(
            self.scannet_root, scene_id, frame_type, f"{frame_id}{ext}"
        )

    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache = {}
