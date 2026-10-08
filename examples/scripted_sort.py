"""Hand-coded pick-and-place: a physics check and a baseline for RL.

Run `python examples/scripted_sort.py` (macOS: `mjpython`) to watch it sort one of each class.
"""

import gymnasium as gym
import numpy as np

import envs  # noqa: F401  Registers the envs.
from assets import BIN_XY, CLASS_NAMES, DROP_HEIGHT, OBJECT_HEIGHT

# The palm sits ~4 cm above the fingertips, so tall objects are gripped near the top.
GRASP_BELOW_TOP = 0.03


def scripted_sort(env):
    """Drive the grasp point through waypoints. Returns True if the episode ended during the script."""
    grasp = env.data.site(env.site_id)
    seen = env.perceive()
    i = seen["class_id"]
    x, y, z = seen["position"]
    grip_z = max(z, OBJECT_HEIGHT[i] - GRASP_BELOW_TOP)
    # (grasp point xyz, gripper, hold steps)
    tour = [
        ((x, y, OBJECT_HEIGHT[i] + 0.06), 1, 0),  # Above the object, open.
        ((x, y, grip_z), 1, 0),  # Down around it.
        ((x, y, grip_z), -1, 15),  # Close and wait.
        ((x, y, 0.30), -1, 0),  # Lift.
        ((*BIN_XY[i], DROP_HEIGHT), -1, 0),  # Over the bin.
        ((*BIN_XY[i], DROP_HEIGHT), 1, 40),  # Open, wait for it to fall and settle.
    ]
    done = False
    for goal, gripper, hold in tour:
        goal = np.array(goal)
        moves = 0
        # Give up on a waypoint after 150 steps: contact can block the gripper short of it.
        while not done and np.linalg.norm(goal - grasp.xpos) > 0.01 and moves < 150:
            move = np.clip((goal - grasp.xpos) * 20, -1, 1)
            *_, terminated, truncated, _ = env.step(np.r_[move, gripper])
            done = terminated or truncated
            moves += 1
        for _ in range(hold):
            if not done:
                *_, terminated, truncated, _ = env.step(np.r_[0, 0, 0, gripper])
                done = terminated or truncated
    return done


if __name__ == "__main__":
    env = gym.make("PandaTrashSort-v0", render_mode="human")
    for object_class in range(len(CLASS_NAMES)):
        env.reset(seed=object_class, options={"object_class": object_class})
        scripted_sort(env.unwrapped)
    env.close()
