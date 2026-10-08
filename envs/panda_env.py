"""Base class: Menagerie's Panda in MuJoCo with end-effector control and Gymnasium rendering.

Task envs subclass this and add objects, observations and rewards. This file only knows
about the robot: joint targets, the grasp site, the viewer and the offscreen renderer.
"""

from contextlib import nullcontext
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from controllers import ee_joint_targets

# Joint angles that point the gripper straight down above the table.
PANDA_HOME = np.array([0, -0.45, 0, -2.2, 0, 1.75, 0.7854])
# The grasp point is kept inside this box: above the table, over every bin.
WORKSPACE_BOUNDS = np.array([[0.25, -0.50, 0.01], [0.80, 0.50, 0.50]])
MAX_EE_SPEED = 0.25  # m/s
MAX_GRIPPER_COMMAND = 255

DEFAULT_CAMERA_CONFIG = {
    "lookat": np.array([0.4, 0.0, 0.15]),
    "distance": 1.7,
    "azimuth": 135,
    "elevation": -30,
}


class PandaEnv(gym.Env):
    """Action: [dx, dy, dz, gripper], each in [-1, 1]. Gripper -1 closed, +1 open."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        xml,
        render_mode=None,
        frame_skip=20,
        home=PANDA_HOME,
        workspace_bounds=WORKSPACE_BOUNDS,
        max_ee_speed=MAX_EE_SPEED,
        camera="overview",
        render_width=640,
        render_height=480,
    ):
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip  # 500 Hz physics, 25 Hz policy.
        self.dt = self.model.opt.timestep * self.frame_skip
        self.home = np.asarray(home, dtype=np.float64)
        self.workspace_low, self.workspace_high = np.asarray(workspace_bounds)
        self.max_ee_speed = max_ee_speed
        self.camera = camera
        self.render_width = render_width
        self.render_height = render_height
        self.joint_limits = self.model.jnt_range[:7].copy()
        # The servos are springs (kp) with dampers (kd). Damping holds a moving joint back, so
        # the arm only reaches ~40% of the commanded speed. Placing each target `lead` times
        # further ahead makes the spring pull cancel the damper drag. Controller only, not physics.
        kp = self.model.actuator_gainprm[:7, 0]
        kd = -self.model.actuator_biasprm[:7, 2] + self.model.dof_damping[:7]
        self.lead = 1 + kd / (kp * self.dt)
        self.site_id = self.model.site("grasp").id
        # Remember the gripper's downward orientation at home; ee control holds it.
        self.data.qpos[:7] = self.home
        mujoco.mj_forward(self.model, self.data)
        self.down = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self._viewer = None
        self._renderer = None
        self._last_frame = 0.0

    # Helpers for subclasses.

    def _lock(self):
        """The human viewer runs in another thread; hold its lock while changing the sim."""
        return self._viewer.lock() if self._viewer is not None else nullcontext()

    def _reset_robot(self):
        """Robot to home with the fingers open. Call inside ``_lock()`` after ``mj_resetData``."""
        self.data.qpos[:7] = self.home
        self.data.qpos[7:9] = 0.04  # Fingers open.
        self.data.ctrl[:7] = self.home
        self.data.ctrl[7] = MAX_GRIPPER_COMMAND

    def _apply_action(self, action):
        """One policy step: set joint and gripper targets, then run ``frame_skip`` physics steps."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        position = self.data.site_xpos[self.site_id]
        goal = np.clip(
            position + action[:3] * self.max_ee_speed * self.dt,
            self.workspace_low,
            self.workspace_high,
        )
        # The PD servos chase these joint targets; the gripper maps -1..1 to ctrl 0..255.
        q = self.data.qpos[:7]
        targets = ee_joint_targets(self.model, self.data, self.site_id, goal, self.down)
        targets = q + self.lead * (targets - q)
        self.data.ctrl[:7] = np.clip(
            targets, self.joint_limits[:, 0] + 0.02, self.joint_limits[:, 1] - 0.02
        )
        self.data.ctrl[7] = (action[3] + 1) * MAX_GRIPPER_COMMAND / 2
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_forward(self.model, self.data)

    def get_robot_state(self):
        """Grasp point position, grasp point linear velocity, finger opening (0 to 0.08 m)."""
        grasp = self.data.site_xpos[self.site_id]
        velocity = np.zeros(6)  # [angular, linear]
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.site_id, velocity, 0
        )
        return np.concatenate([grasp, velocity[3:], [self.data.qpos[7] + self.data.qpos[8]]])

    # Gymnasium API.

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=self.render_height, width=self.render_width
                )
            self._renderer.update_scene(self.data, camera=self.camera)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
                self._viewer.cam.lookat[:] = DEFAULT_CAMERA_CONFIG["lookat"]
                self._viewer.cam.distance = DEFAULT_CAMERA_CONFIG["distance"]
                self._viewer.cam.azimuth = DEFAULT_CAMERA_CONFIG["azimuth"]
                self._viewer.cam.elevation = DEFAULT_CAMERA_CONFIG["elevation"]
            if self._viewer.is_running():
                self._viewer.sync()
            # Real time: one policy step (dt of simulated time) per dt of wall-clock time.
            time.sleep(max(0.0, self._last_frame + self.dt - time.perf_counter()))
            self._last_frame = time.perf_counter()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
