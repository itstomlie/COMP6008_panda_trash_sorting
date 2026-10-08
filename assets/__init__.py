"""Scene construction. The Panda XML comes from mujoco-menagerie; everything else is built here."""

from assets.panda import panda_path
from assets.scene import (
    BIN_HALF,
    BIN_RIM,
    BIN_XY,
    CLASS_NAMES,
    DROP_HEIGHT,
    OBJECT_HEIGHT,
    OBJECT_RADIUS,
    make_scene,
)

__all__ = [
    "BIN_HALF",
    "BIN_RIM",
    "BIN_XY",
    "CLASS_NAMES",
    "DROP_HEIGHT",
    "OBJECT_HEIGHT",
    "OBJECT_RADIUS",
    "make_scene",
    "panda_path",
]
