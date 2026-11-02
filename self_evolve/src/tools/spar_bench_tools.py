"""SPAR-Bench deterministic geometry tools for DGE ground-truth computation.

Each tool takes scene/frame metadata + object labels and returns the
ground-truth answer for a SPAR-Bench question type. All computation is
pure geometry from camera extrinsics (camera-to-world pose) + object
3d_location (world coords).

Conventions:
  - Camera extrinsics E (4x4) maps camera->world. To get world->camera:
        cam_point = inv(E) @ [world_x, world_y, world_z, 1]
  - depth(obj)   = cam_point[2]   (z in camera frame)
  - distance(cam,obj) = ||cam_pos - obj_center||  (Euclidean, world frame)
  - distance(obj_a, obj_b) = ||a_center - b_center|| (Euclidean, world frame)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .base_tool import BaseTool
from .camera_tools import CameraParameterTool
from ..data.scannet_loader import ScanNetLoader


def _find_visible_object(frame_metadata: Dict[str, Any], label: str,
                         min_visibility: float = 0.1) -> Optional[Dict[str, Any]]:
    """Find a unique visible object by label in frame metadata."""
    objs = frame_metadata.get("objects", [])
    matches = [o for o in objs if o.get("label", "").lower() == label.lower()
               and o.get("visibility", 1.0) >= min_visibility]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # pick the most visible
        return max(matches, key=lambda o: o.get("visibility", 0.0))
    # fallback: case-insensitive contains
    matches = [o for o in objs if label.lower() in o.get("label", "").lower()
               and o.get("visibility", 1.0) >= min_visibility]
    if matches:
        return max(matches, key=lambda o: o.get("visibility", 0.0))
    return None


def _world_to_camera(extrinsics: np.ndarray, world_point: np.ndarray) -> np.ndarray:
    """Transform a world point to camera coordinates."""
    E = np.eye(4, dtype=float)
    E[:4, :4] = extrinsics
    cam_E = np.linalg.inv(E)
    homo = np.array([world_point[0], world_point[1], world_point[2], 1.0])
    cam = cam_E @ homo
    return cam[:3]


class SPARDepthTool(BaseTool):
    """Compute depth (camera-frame z) of objects for depth_prediction_* tasks."""

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="spar_depth_tool",
            description="Compute object depth in camera frame for SPAR depth_prediction tasks",
            aliases=("spar_depth_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.camera_tool = CameraParameterTool()

    def execute(self, scene_id: str, frame_id: Any,
                frame_metadata: Dict[str, Any],
                object_labels: List[str],
                mode: str = "oo",
                min_visibility: float = 0.1, **kwargs) -> Dict[str, Any]:
        """mode: 'oc' = answer is the depth of the SECOND label (first is the
        anchor whose depth is given in the question); 'oo' = answer is the
        absolute depth difference between the two labels.

        2026-08-08 fix: previously the answer was `str(list(depths.values()))`
        (a stringified list) for BOTH variants, which the numerical reward
        could not parse — depth task rewards were noise.
        """
        scene_data_path = os.path.join(self.loader.scannet_root, str(scene_id).strip())
        cam_params = self.camera_tool.execute(scene_data_path=scene_data_path, frame_id=frame_id)
        extrinsics = cam_params["extrinsics"]

        depths = {}
        ordered_depths = []
        for label in object_labels:
            obj = _find_visible_object(frame_metadata, label, min_visibility)
            if obj is None:
                return {"success": False, "error": f"Object '{label}' not found or not visible"}
            world_loc = np.array(obj.get("3d_location", [0.0, 0.0, 0.0]), dtype=float)
            cam_pt = _world_to_camera(extrinsics, world_loc)
            d = round(float(cam_pt[2]), 2)
            depths[label] = d
            ordered_depths.append(d)

        if len(ordered_depths) < 2:
            return {"success": False, "error": "Need 2 object labels for depth task"}
        if any(d <= 0 for d in ordered_depths):
            return {"success": False, "error": "Object behind the camera (non-positive depth)"}

        if str(mode).strip().lower() == "oc":
            # [anchor (depth given in question), target (depth asked)]
            answer_val = ordered_depths[1]
        else:
            answer_val = round(abs(ordered_depths[0] - ordered_depths[1]), 2)

        return {"success": True, "depths": depths,
                "ordered_depths": ordered_depths,
                "anchor_depth": ordered_depths[0],
                "answer": str(answer_val)}


class SPARDistanceTool(BaseTool):
    """Compute Euclidean distances for distance_prediction_* tasks."""

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="spar_distance_tool",
            description="Compute Euclidean distances for SPAR distance_prediction tasks",
            aliases=("spar_distance_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.camera_tool = CameraParameterTool()

    def execute(self, scene_id: str, frame_id: Any,
                frame_metadata: Dict[str, Any],
                mode: str = "oo",
                object_labels: List[str] = None,
                min_visibility: float = 0.1, **kwargs) -> Dict[str, Any]:
        """mode: 'oc' = camera-to-object, 'oo' = object-to-object."""
        scene_data_path = os.path.join(self.loader.scannet_root, str(scene_id).strip())
        cam_params = self.camera_tool.execute(scene_data_path=scene_data_path, frame_id=frame_id)
        cam_pos = np.array(cam_params["position"], dtype=float)

        if mode == "oc":
            label = object_labels[0]
            obj = _find_visible_object(frame_metadata, label, min_visibility)
            if obj is None:
                return {"success": False, "error": f"Object '{label}' not found"}
            obj_loc = np.array(obj.get("3d_location", [0.0, 0.0, 0.0]), dtype=float)
            dist = float(np.linalg.norm(cam_pos - obj_loc))
            return {"success": True, "distance_m": round(dist, 2),
                    "answer": str(round(dist, 2))}
        else:  # oo
            if len(object_labels) < 2:
                return {"success": False, "error": "Need 2 object labels for oo mode"}
            locs = []
            for label in object_labels[:2]:
                obj = _find_visible_object(frame_metadata, label, min_visibility)
                if obj is None:
                    return {"success": False, "error": f"Object '{label}' not found"}
                locs.append(np.array(obj.get("3d_location", [0.0, 0.0, 0.0]), dtype=float))
            dist = float(np.linalg.norm(locs[0] - locs[1]))
            return {"success": True, "distance_m": round(dist, 2),
                    "answer": str(round(dist, 2))}


class SPARDistanceInferTool(BaseTool):
    """For distance_infer_center_oo: which of two objects is farther from a third."""

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="spar_distance_infer_tool",
            description="Determine which of two objects is farther from an anchor",
            aliases=("spar_distance_infer_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()

    def execute(self, frame_metadata: Dict[str, Any],
                anchor_label: str, option_a_label: str, option_b_label: str,
                min_visibility: float = 0.1, **kwargs) -> Dict[str, Any]:
        anchor = _find_visible_object(frame_metadata, anchor_label, min_visibility)
        opt_a = _find_visible_object(frame_metadata, option_a_label, min_visibility)
        opt_b = _find_visible_object(frame_metadata, option_b_label, min_visibility)
        if anchor is None or opt_a is None or opt_b is None:
            return {"success": False, "error": "One or more objects not found"}
        a_loc = np.array(anchor.get("3d_location", [0.0, 0.0, 0.0]), dtype=float)
        loc_a = np.array(opt_a.get("3d_location", [0.0, 0.0, 0.0]), dtype=float)
        loc_b = np.array(opt_b.get("3d_location", [0.0, 0.0, 0.0]), dtype=float)
        dist_a = float(np.linalg.norm(a_loc - loc_a))
        dist_b = float(np.linalg.norm(a_loc - loc_b))
        if dist_a > dist_b:
            answer = "A"
        elif dist_b > dist_a:
            answer = "B"
        else:
            answer = "A"  # tie-break
        return {"success": True, "answer": answer,
                "dist_a": round(dist_a, 2), "dist_b": round(dist_b, 2)}


class SPARSpatialRelationTool(BaseTool):
    """For obj_spatial_relation_*: classify spatial relation between two objects."""

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="spar_spatial_relation_tool",
            description="Classify spatial relation (left/right/front/behind/above/below)",
            aliases=("spar_spatial_relation_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.camera_tool = CameraParameterTool()

    # "empty if indistinguishable" margin (meters), mirroring SPAR-Bench's
    # triple format where a near-tie axis is left blank.
    AXIS_MARGIN_M = 0.15

    def execute(self, scene_id: str, frame_id: Any,
                frame_metadata: Dict[str, Any],
                ref_label: str, target_label: str,
                mode: str = "oo",
                min_visibility: float = 0.1, **kwargs) -> Dict[str, Any]:
        """Compute the SPAR-Bench spatial-relation triple.

        mode 'oo' / 'oo_mv' (two objects, observer's view):
            triple = (left|right|'', above|below|'', farther|closer|'')
            of the TARGET relative to the REFERENCE.
        mode 'oc_mv' (object vs observer):
            triple = (left|right|'', above|below|'', front|behind|'')
            of the TARGET relative to the observer's main viewpoint
            (ref_label is validated but not used for the answer).

        Answer = bench-style string "x, y, z" with blank slots kept, e.g.
        "right, , farther".

        2026-08-08 fix: previously this tool collapsed the relation to a fixed
        secret letter map (A=left, B=right, ...) that the Solver never saw —
        the Round-2 question showed no options, so the reward was noise.
        """
        scene_data_path = os.path.join(self.loader.scannet_root, str(scene_id).strip())
        cam_params = self.camera_tool.execute(scene_data_path=scene_data_path, frame_id=frame_id)
        extrinsics = cam_params["extrinsics"]

        ref_obj = _find_visible_object(frame_metadata, ref_label, min_visibility)
        tgt_obj = _find_visible_object(frame_metadata, target_label, min_visibility)
        if ref_obj is None or tgt_obj is None:
            return {"success": False, "error": "Object not found"}

        ref_world = np.array(ref_obj.get("3d_location", [0.0, 0.0, 0.0]), dtype=float)
        tgt_world = np.array(tgt_obj.get("3d_location", [0.0, 0.0, 0.0]), dtype=float)

        # Camera frame: x=right, y=down, z=forward (ScanNet convention)
        ref_cam = _world_to_camera(extrinsics, ref_world)
        tgt_cam = _world_to_camera(extrinsics, tgt_world)

        margin = self.AXIS_MARGIN_M
        mode_norm = str(mode).strip().lower()

        if mode_norm == "oc_mv":
            vec = tgt_cam  # object position relative to the observer
            depth_pos, depth_neg = "front", "behind"
        else:  # oo / oo_mv: target relative to reference
            vec = tgt_cam - ref_cam
            depth_pos, depth_neg = "farther", "closer"

        lr = "" if abs(vec[0]) <= margin else ("right" if vec[0] > 0 else "left")
        ab = "" if abs(vec[1]) <= margin else ("below" if vec[1] > 0 else "above")
        fd = "" if abs(vec[2]) <= margin else (depth_pos if vec[2] > 0 else depth_neg)

        triple = [lr, ab, fd]
        if not any(triple):
            return {"success": False,
                    "error": "Objects indistinguishable on all axes (relation is ambiguous)"}
        answer = ", ".join(triple)
        return {"success": True, "relation_triple": triple, "answer": answer,
                "relation_mode": mode_norm}


class SPARPositionMatchingTool(BaseTool):
    """position_matching: find object's bbox in frame 2 given its bbox in frame 1.

    Projects the object's 3D bbox into frame 2's image plane and compares
    against 4 candidate 2D bboxes (IoU), returning the best match (A/B/C/D).
    """

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="spar_position_matching_tool",
            description="Match object bbox across two frames",
            aliases=("spar_position_matching_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.camera_tool = CameraParameterTool()

    def execute(self, scene_id: str, frame_id_1: Any, frame_id_2: Any,
                frame_metadata_1: Dict, frame_metadata_2: Dict,
                target_label: str,
                candidate_bboxes: List[List[float]],
                min_visibility: float = 0.1, **kwargs) -> Dict[str, Any]:
        from ..utils.marker_render import load_intrinsics, project_3d_to_2d

        scene_dir = os.path.join(self.loader.scannet_root, str(scene_id).strip())
        obj = _find_visible_object(frame_metadata_1, target_label, min_visibility)
        if obj is None:
            return {"success": False, "error": f"Object '{target_label}' not found"}

        bbox3d = obj.get("3d_bbox", None)
        if not bbox3d or len(bbox3d) < 6:
            return {"success": False, "error": "No 3d bbox for object"}

        # Load frame 2 camera
        cam2 = self.camera_tool.execute(scene_data_path=scene_dir, frame_id=frame_id_2)
        intrinsics = load_intrinsics(scene_dir)

        # Project 3D bbox corners into frame 2
        mins = np.array(bbox3d[:3], dtype=float)
        maxs = np.array(bbox3d[3:6], dtype=float)
        corners = [np.array([x, y, z]) for x in (mins[0], maxs[0])
                    for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])]
        pts2d = [project_3d_to_2d(c, cam2["extrinsics"], intrinsics) for c in corners]
        xs = [p[0] for p in pts2d]
        ys = [p[1] for p in pts2d]
        projected_bbox = [min(xs), min(ys), max(xs), max(ys)]

        # Compare against candidates via IoU
        def iou(a, b):
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
            area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
            return inter / (area_a + area_b - inter + 1e-8)

        best_idx, best_iou = 0, -1.0
        for i, cand in enumerate(candidate_bboxes):
            v = iou(projected_bbox, cand)
            if v > best_iou:
                best_iou, best_idx = v, i
        answer = "ABCD"[best_idx] if best_idx < 4 else "A"
        return {"success": True, "answer": answer, "projected_bbox": projected_bbox,
                "best_iou": round(best_iou, 3)}


class SPARViewChangeInferTool(BaseTool):
    """view_change_infer: compute camera movement + rotation between two frames.

    Returns a string like:
        move_right:1.6,move_down:0,move_backward:0,rotate_up:15,rotate_left:75
    """

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="spar_view_change_infer_tool",
            description="Compute camera movement and rotation between two frames",
            aliases=("spar_view_change_infer_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.camera_tool = CameraParameterTool()

    def execute(self, scene_id: str, frame_id_1: Any, frame_id_2: Any,
                **kwargs) -> Dict[str, Any]:
        scene_dir = os.path.join(self.loader.scannet_root, str(scene_id).strip())
        cam1 = self.camera_tool.execute(scene_data_path=scene_dir, frame_id=frame_id_1)
        cam2 = self.camera_tool.execute(scene_data_path=scene_dir, frame_id=frame_id_2)
        p1 = np.array(cam1["position"], dtype=float)
        p2 = np.array(cam2["position"], dtype=float)
        delta = p2 - p1

        # Movement: camera frame x=right, y=down, z=forward
        move_right = round(float(delta[0]), 1)
        move_down = round(float(delta[1]), 1)
        move_forward = round(float(delta[2]), 1)

        def _dir(val, pos_word, neg_word):
            if val > 0.05:
                return f"{pos_word}:{abs(val)}"
            elif val < -0.05:
                return f"{neg_word}:{abs(val)}"
            return f"{pos_word}:0"

        move_str = ",".join([
            _dir(move_right, "move_right", "move_left"),
            _dir(move_down, "move_down", "move_up"),
            _dir(move_forward, "move_forward", "move_backward"),
        ])

        # Rotation: compute relative rotation R_rel = R1^T @ R2
        R1 = np.array(cam1["rotation"], dtype=float)
        R2 = np.array(cam2["rotation"], dtype=float)
        R_rel = R1.T @ R2
        # Extract Euler angles (ZYX / yaw-pitch-roll approximation)
        # yaw (around y), pitch (around x), roll (around z)
        pitch = round(float(np.degrees(np.arcsin(-R_rel[2, 0]))), 0)
        yaw = round(float(np.degrees(np.arctan2(R_rel[1, 0], R_rel[0, 0]))), 0)
        roll = round(float(np.degrees(np.arctan2(R_rel[2, 1], R_rel[2, 2]))), 0)

        rot_str = ",".join([
            _dir(pitch, "rotate_up", "rotate_down"),
            _dir(yaw, "rotate_right", "rotate_left"),
        ])

        answer = f"{move_str},{rot_str}"
        return {"success": True, "answer": answer,
                "move": move_str, "rotation": rot_str}


class SPARCameraMotionInferTool(BaseTool):
    """camera_motion_infer: project camera 2 position into camera 1 image plane + depth.

    Returns the matching option (A or B) from two candidate (x, y, depth) tuples.
    """

    def __init__(self, scannet_loader: ScanNetLoader = None):
        super().__init__(
            name="spar_camera_motion_infer_tool",
            description="Project camera 2 position into camera 1 image plane",
            aliases=("spar_camera_motion_infer_tool",),
        )
        self.loader = scannet_loader or ScanNetLoader()
        self.camera_tool = CameraParameterTool()

    def execute(self, scene_id: str, frame_id_1: Any, frame_id_2: Any,
                options: List[dict], **kwargs) -> Dict[str, Any]:
        """options: [{"letter":"A","x":569,"y":18,"depth":1.1}, {"letter":"B",...}]"""
        from ..utils.marker_render import load_intrinsics, project_3d_to_2d

        scene_dir = os.path.join(self.loader.scannet_root, str(scene_id).strip())
        cam1 = self.camera_tool.execute(scene_data_path=scene_dir, frame_id=frame_id_1)
        cam2 = self.camera_tool.execute(scene_data_path=scene_dir, frame_id=frame_id_2)
        intrinsics = load_intrinsics(scene_dir)

        # Project cam2 position into cam1 image plane
        px, py = project_3d_to_2d(np.array(cam2["position"]), cam1["extrinsics"], intrinsics)
        # Depth = distance from cam1 to cam2
        depth = round(float(np.linalg.norm(
            np.array(cam1["position"]) - np.array(cam2["position"]))), 1)

        # Match against options
        best_idx, best_dist = 0, 1e9
        for i, opt in enumerate(options):
            ox, oy, od = opt.get("x", 0), opt.get("y", 0), opt.get("depth", 0)
            d = (px - ox) ** 2 + (py - oy) ** 2 + (depth - od) ** 2 * 100
            if d < best_dist:
                best_dist, best_idx = d, i
        answer = options[best_idx].get("letter", "ABCD"[best_idx])
        return {"success": True, "answer": answer,
                "projected": (round(px, 0), round(py, 0), depth)}

