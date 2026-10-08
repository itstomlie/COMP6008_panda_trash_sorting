"""Prefetch Franka Panda from mujoco-menagerie into the per-user cache.

Run ``python -m assets.panda`` once after installing.
"""

import mujoco_menagerie as mm

ROBOT = "franka_emika_panda"


def panda_path():
    """Cached model directory on disk."""
    return mm.get(ROBOT).path()


if __name__ == "__main__":
    mm.prefetch([ROBOT])
    robot = mm.get(ROBOT)
    model = robot.model(robot.default_model)
    print(f"Panda ready: {robot.path()}")
    print(f"  entries: {robot.entry_names}")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu}")
