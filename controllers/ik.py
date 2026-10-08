"""Differential inverse kinematics for the Panda's seven arm joints."""

import mujoco
import numpy as np


def ee_joint_targets(model, data, site_id, goal, down, damping=1e-4):
    """Joint targets that move `site_id` to `goal` while rotating its frame back onto `down`.

    `down` is the 3x3 rotation the site should hold. Returns the current seven arm
    joint angles plus one damped-least-squares step towards the goal.
    """
    position = data.site_xpos[site_id]
    # Small rotation that turns the site's current axes back onto the downward axes.
    rotation = data.site_xmat[site_id].reshape(3, 3)
    twist = 0.5 * np.cross(rotation.T, down.T).sum(axis=0)
    # Jacobian: how the site moves (3 rows) and turns (3 rows) per unit of each joint.
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jacobian = np.vstack([jacp[:, :7], jacr[:, :7]])
    error = np.r_[goal - position, twist]
    # Solve jacobian @ dq = error. The damping term keeps dq small near poses where
    # the arm cannot move in some direction.
    dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
    return data.qpos[:7] + dq
