"""Render colored markers (points, bboxes) onto scene images for SPAR-Bench questions.

SPAR-Bench questions reference objects with colored markers:
  - "red point", "blue point", "green point"  (small filled circles)
  - "red bbox", "blue bbox"                    (rectangle outlines)

This module draws these markers onto the frame image so the Solver sees
the same visual context as SPAR-Bench evaluates.
"""

from __future__ import annotations

import io
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Color name -> RGB
COLOR_MAP = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 180, 40),
    "yellow": (220, 220, 40),
    "white": (255, 255, 255),
    "black": (30, 30, 30),
}

POINT_RADIUS = 6
BBOX_WIDTH = 3


def draw_point(image: Image.Image, x: float, y: float, color: str = "red",
               radius: int = POINT_RADIUS) -> Image.Image:
    """Draw a filled circle marker at (x, y) on a copy of the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    rgb = COLOR_MAP.get(color.lower(), COLOR_MAP["red"])
    r = radius
    draw.ellipse([x - r, y - r, x + r, y + r], fill=rgb, outline=(255, 255, 255))
    return img


def draw_bbox(image: Image.Image, bbox: List[float], color: str = "red",
              width: int = BBOX_WIDTH) -> Image.Image:
    """Draw a rectangle outline for bbox=[x1, y1, x2, y2] on a copy."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    rgb = COLOR_MAP.get(color.lower(), COLOR_MAP["red"])
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=rgb, width=width)
    return img


def render_markers(image_path: str, markers: List[dict],
                   output_path: Optional[str] = None) -> str:
    """Render a list of markers onto an image.

    Each marker dict:
        {"type": "point", "x": float, "y": float, "color": "red"}
        {"type": "bbox", "bbox": [x1,y1,x2,y2], "color": "red"}

    Returns the output path (or in-memory path if output_path is None).
    """
    img = Image.open(image_path).convert("RGB")
    for m in markers:
        if m.get("type") == "point":
            img = draw_point(img, m["x"], m["y"], m.get("color", "red"))
        elif m.get("type") == "bbox":
            img = draw_bbox(img, m["bbox"], m.get("color", "red"))
    if output_path is None:
        # Save to a temp path next to the original
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_marked{ext or '.jpg'}"
    img.save(output_path)
    return output_path


def project_3d_to_2d(world_point: np.ndarray, extrinsics: np.ndarray,
                     intrinsics: np.ndarray) -> Tuple[float, float]:
    """Project a 3D world point to 2D pixel coordinates.

    extrinsics: 4x4 camera-to-world. intrinsics: 3x3.
    Returns (x, y) pixel coords.
    """
    E = np.eye(4, dtype=float)
    E[:4, :4] = extrinsics
    cam_E = np.linalg.inv(E)
    homo = np.array([world_point[0], world_point[1], world_point[2], 1.0])
    cam_pt = cam_E @ homo  # [x, y, z, 1] in camera frame
    if abs(cam_pt[2]) < 1e-6:
        return (0.0, 0.0)
    px = intrinsics[0, 0] * cam_pt[0] / cam_pt[2] + intrinsics[0, 2]
    py = intrinsics[1, 1] * cam_pt[1] / cam_pt[2] + intrinsics[1, 2]
    return (float(px), float(py))


def load_intrinsics(scene_dir: str) -> np.ndarray:
    """Load intrinsic_color.txt (3x3 or 4x4 -> take 3x3)."""
    path = os.path.join(scene_dir, "intrinsic_color.txt")
    if not os.path.exists(path):
        return np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1.0]])
    mat = np.loadtxt(path)
    if mat.shape == (4, 4):
        return mat[:3, :3]
    return mat
