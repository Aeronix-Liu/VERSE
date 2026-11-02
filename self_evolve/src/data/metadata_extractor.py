"""Metadata extractor."""

import os
import json
import numpy as np
from typing import Dict, List, Any

class MetadataExtractor:
    """Helper methods for scene metadata."""

    def __init__(self):
        pass

    def extract_scene_metadata(
        self,
        scene_dir: str,
        scene_id: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Load scene metadata from the extracted output."""
        # Placeholder wrapper around the original extraction pipeline.
        metadata_file = os.path.join(output_dir, scene_id, f"{scene_id}.json")

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)

        print(f"Metadata not found for {scene_id}. Please run metadata extraction first.")
        return None

    @staticmethod
    def _aabb_closest_distance(bbox1: List[float], bbox2: List[float]) -> float:
        """Closest-point distance between two axis-aligned bounding boxes.

        Each bbox is [xmin, ymin, zmin, xmax, ymax, zmax]. For every axis the
        per-axis gap is 0 when the intervals overlap, otherwise the separation
        between them. The Euclidean norm of the gap vector is the distance
        between the two closest points of the boxes (0 when they intersect).
        """
        gap = 0.0
        for axis in range(3):
            lo1, hi1 = bbox1[axis], bbox1[axis + 3]
            lo2, hi2 = bbox2[axis], bbox2[axis + 3]
            if hi1 < lo2:
                gap += (lo2 - hi1) ** 2
            elif hi2 < lo1:
                gap += (lo1 - hi2) ** 2
            # else: intervals overlap -> 0 contribution
        return float(np.sqrt(gap))

    @staticmethod
    def compute_object_distance(obj1: Dict, obj2: Dict) -> float:
        """Distance between two objects.

        Uses the closest-point distance between the objects' 3D bounding boxes
        when both expose a ``3d_bbox`` (matches the VSI-Bench / GRPO protocol of
        "measuring from the closest point of each object"). Falls back to the
        center-to-center distance (``3d_location``) when bbox data is missing.
        """
        bbox1 = obj1.get("3d_bbox")
        bbox2 = obj2.get("3d_bbox")
        if bbox1 and bbox2 and len(bbox1) == 6 and len(bbox2) == 6:
            return MetadataExtractor._aabb_closest_distance(bbox1, bbox2)
        loc1 = np.array(obj1["3d_location"])
        loc2 = np.array(obj2["3d_location"])
        return float(np.linalg.norm(loc1 - loc2))

    @staticmethod
    def compute_relative_direction(
        positioning_obj: Dict,
        orienting_obj: Dict,
        querying_obj: Dict
    ) -> str:
        """Compute the querying object's direction in the local reference frame."""
        pos_loc = np.array(positioning_obj["3d_location"])
        orient_loc = np.array(orienting_obj["3d_location"])
        query_loc = np.array(querying_obj["3d_location"])

        # Build the forward vector from positioning to orienting.
        forward = orient_loc - pos_loc
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        # Build the query vector from positioning to querying.
        to_query = query_loc - pos_loc
        to_query = to_query / (np.linalg.norm(to_query) + 1e-8)

        # Use the XY plane only and ignore height.
        forward_2d = forward[:2]
        to_query_2d = to_query[:2]

        # Compute the right-hand vector and the two projections.
        right = np.array([forward_2d[1], -forward_2d[0]])
        forward_proj = np.dot(to_query_2d, forward_2d)
        right_proj = np.dot(to_query_2d, right)

        threshold = 0.3

        if abs(forward_proj) < threshold:
            return "left" if right_proj < 0 else "right"
        if abs(right_proj) < threshold:
            return "back" if forward_proj < 0 else "front"
        if forward_proj > 0:
            return "front-left" if right_proj < 0 else "front-right"
        return "back-left" if right_proj < 0 else "back-right"

    @staticmethod
    def get_object_size(obj: Dict) -> float:
        """Return the longest object dimension in centimeters."""
        # New format: `size` is already the longest dimension in centimeters.
        if isinstance(obj["size"], (int, float)):
            return float(obj["size"])

        # Legacy format: `size` stores width/length/height.
        if isinstance(obj["size"], dict):
            dimensions = [obj["size"]["width"], obj["size"]["length"], obj["size"]["height"]]
            return max(dimensions) * 100

        # Fallback: compute the size from the 3D box.
        bbox = obj["3d_bbox"]
        width = abs(bbox[3] - bbox[0]) * 100
        length = abs(bbox[4] - bbox[1]) * 100
        height = abs(bbox[5] - bbox[2]) * 100
        return max(width, length, height)

    @staticmethod
    def get_room_size(scene_metadata: Dict) -> float:
        """Return room area in square meters."""
        room_size = scene_metadata.get("room_size", 0.0)

        # New format: direct numeric area.
        if isinstance(room_size, (int, float)):
            return float(room_size)

        # Legacy format: dictionary with an `area` field.
        if isinstance(room_size, dict):
            return room_size.get("area", 0.0)

        return 0.0

    @staticmethod
    def compute_closest_object(
        target_obj: Dict,
        candidate_objs: List[Dict]
    ) -> Dict:
        """Return the closest candidate object."""
        if not candidate_objs:
            return None

        target_loc = np.array(target_obj["3d_location"])
        min_dist = float('inf')
        closest_obj = None

        for obj in candidate_objs:
            obj_loc = np.array(obj["3d_location"])
            dist = np.linalg.norm(obj_loc - target_loc)

            if dist < min_dist:
                min_dist = dist
                closest_obj = obj

        return closest_obj
