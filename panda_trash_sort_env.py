"""Gymnasium env: a Panda sorts one piece of trash into the bin matching its class."""

from contextlib import nullcontext
import time

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from scene import BIN_XY, DROP_HEIGHT, make_scene

ENV_ID = "PandaTrashSort-v0"
# Joint angles that point the gripper straight down above the table.
HOME = np.array([0, -0.45, 0, -2.2, 0, 1.75, 0.7854])
# The grasp point is kept inside this box: above the table, over every bin.
WORKSPACE_LOW = np.array([0.25, -0.50, 0.01])
WORKSPACE_HIGH = np.array([0.80, 0.50, 0.50])
MAX_EE_SPEED = 0.25  # m/s
# Objects spawn in front of the robot, clear of every bin, with a random yaw.
SPAWN_LOW = np.array([0.36, -0.17])
SPAWN_HIGH = np.array([0.49, 0.17])


class PandaTrashEnv(gym.Env):
    """Action: [dx, dy, dz, gripper], each in [-1, 1]. Gripper -1 closed, +1 open.

    Observation (25 numbers):
      0-2    grasp point position
      3-5    grasp point velocity
      6      finger opening (0 to 0.08 m)
      7-9    object position (centre of mass)
      10-12  object minus grasp point
      13-15  object's up axis (tells upright from tipped)
      16-18  object velocity
      19-21  drop target minus object
      22-24  one-hot class (can, bottle, paper)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode=None, max_episode_steps=400, object_class=None):
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.object_class = object_class  # None picks a random class each reset.
        self.model = mujoco.MjModel.from_xml_string(make_scene())
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 20  # 500 Hz physics, 25 Hz policy.
        self.dt = self.model.opt.timestep * self.frame_skip
        self.joint_limits = self.model.jnt_range[:7].copy()
        # The servos are springs (kp) with dampers (kd). Damping holds a moving joint back, so
        # the arm only reaches ~40% of the commanded speed. Placing each target `lead` times
        # further ahead makes the spring pull cancel the damper drag. Controller only, not physics.
        kp = self.model.actuator_gainprm[:7, 0]
        kd = -self.model.actuator_biasprm[:7, 2] + self.model.dof_damping[:7]
        self.lead = 1 + kd / (kp * self.dt)
        self.site_id = self.model.site("grasp").id
        self.object_ids = [self.model.body(f"object{i}").id for i in range(3)]
        joints = [self.model.joint(f"object{i}_joint").id for i in range(3)]
        self.object_qpos = [self.model.jnt_qposadr[j] for j in joints]
        self.object_dofs = [self.model.jnt_dofadr[j] for j in joints]
        # Remember the gripper's downward orientation at HOME; ee control holds it.
        self.data.qpos[:7] = HOME
        mujoco.mj_forward(self.model, self.data)
        self.down = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (25,), dtype=np.float32)
        self._viewer = None
        self._renderer = None
        self._last_frame = 0.0

    def _lock(self):
        """The human viewer runs in another thread; hold its lock while changing the sim."""
        return self._viewer.lock() if self._viewer is not None else nullcontext()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        with self._lock():
            # Every object goes back to its parking spot under the table.
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:7] = HOME
            self.data.qpos[7:9] = 0.04  # Fingers open.
            self.data.ctrl[:7] = HOME
            self.data.ctrl[7] = 255
            chosen = (options or {}).get("object_class", self.object_class)
            self.target = int(self.np_random.integers(3) if chosen is None else chosen)
            # Move the chosen object onto the table. Free joint qpos = [x, y, z, quaternion].
            xy = self.np_random.uniform(SPAWN_LOW, SPAWN_HIGH)
            yaw = self.np_random.uniform(-np.pi, np.pi)
            q = self.object_qpos[self.target]
            self.data.qpos[q : q + 7] = [
                *xy,
                0.003,
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            ]
            mujoco.mj_step(self.model, self.data, nstep=100)  # Settle onto the table.
            mujoco.mj_forward(self.model, self.data)
            self.steps = 0
            observation = self._observation()
        if self.render_mode == "human":
            self.render()
        return observation, {}

    def perceive(self):
        """Everything the policy knows about the object. A camera pipeline could replace this."""
        i = self.target
        body_id = self.object_ids[i]
        v = self.object_dofs[i]
        return {
            "position": self.data.xipos[body_id].copy(),
            "up": self.data.xmat[body_id].reshape(3, 3)[:, 2].copy(),
            "velocity": self.data.qvel[v : v + 3].copy(),
            "class_id": i,
        }

    def _observation(self):
        seen = self.perceive()
        grasp = self.data.site_xpos[self.site_id]
        grasp_velocity = np.zeros(6)  # [angular, linear]
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.site_id,
            grasp_velocity,
            0,
        )
        drop = np.r_[BIN_XY[seen["class_id"]], DROP_HEIGHT]
        return np.concatenate(
            [
                grasp,
                grasp_velocity[3:],
                [self.data.qpos[7] + self.data.qpos[8]],
                seen["position"],
                seen["position"] - grasp,
                seen["up"],
                seen["velocity"],
                drop - seen["position"],
                np.eye(3)[seen["class_id"]],
            ]
        ).astype(np.float32)

    def _ee_targets(self, move):
        """Inverse kinematics: joint targets that move the grasp point by `move`
        while the gripper keeps pointing down."""
        position = self.data.site_xpos[self.site_id]
        goal = np.clip(
            position + move * MAX_EE_SPEED * self.dt, WORKSPACE_LOW, WORKSPACE_HIGH
        )
        # Small rotation that turns the gripper's current axes back onto the downward axes.
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        twist = 0.5 * np.cross(rotation.T, self.down.T).sum(axis=0)
        # Jacobian: how the grasp point moves (3 rows) and turns (3 rows) per unit of each joint.
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        jacobian = np.vstack([jacp[:, :7], jacr[:, :7]])
        error = np.r_[goal - position, twist]
        # Solve jacobian @ dq = error. The 1e-4 term (damped least squares) keeps dq small
        # near poses where the arm cannot move in some direction.
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1e-4 * np.eye(6), error
        )
        return self.data.qpos[:7] + dq

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        with self._lock():
            # The PD servos chase these joint targets; the gripper maps -1..1 to ctrl 0..255.
            q = self.data.qpos[:7]
            targets = q + self.lead * (self._ee_targets(action[:3]) - q)
            self.data.ctrl[:7] = np.clip(
                targets, self.joint_limits[:, 0] + 0.02, self.joint_limits[:, 1] - 0.02
            )
            self.data.ctrl[7] = (action[3] + 1) * 127.5
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
            mujoco.mj_forward(self.model, self.data)
            self.steps += 1
            truncated = self.steps >= self.max_episode_steps
            observation = self._observation()
        if self.render_mode == "human":
            self.render()
        return observation, 0.0, False, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer

                self._viewer = viewer.launch_passive(self.model, self.data)
                self._viewer.cam.lookat[:] = [0.4, 0, 0.15]
                self._viewer.cam.distance = 1.7
                self._viewer.cam.azimuth = 135
                self._viewer.cam.elevation = -30
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


if ENV_ID not in gym.envs.registry:
    gym.register(id=ENV_ID, entry_point="panda_trash_sort_env:PandaTrashEnv")


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    check_env(gym.make(ENV_ID).unwrapped, skip_render_check=True)
    print("check_env passed")
