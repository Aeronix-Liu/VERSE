"""Shared DGE data paths under VERSE/ScanNetData."""

from __future__ import annotations

import os
from pathlib import Path

SPATIALEVO_ROOT = Path(__file__).resolve().parent
SCANNET_DATA_ROOT = SPATIALEVO_ROOT / "ScanNetData"

SCANNET_ROOT = os.getenv(
    "SPATIALEVO_SCANNET_TRAIN",
    str(SCANNET_DATA_ROOT / "train"),
)
METADATA_ROOT = os.getenv(
    "SPATIALEVO_SCANNET_METADATA",
    str(SCANNET_DATA_ROOT / "metadata"),
)

# Set after placing processed scenes; used by examples when present.
DEFAULT_SCENE_ID = os.getenv("SPATIALEVO_DEFAULT_SCENE_ID", "scene0000_01")
