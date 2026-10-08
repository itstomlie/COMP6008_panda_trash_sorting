"""Task env: the Panda sorts one piece of trash into the bin matching its class."""

from gymnasium import spaces
import mujoco
import numpy as np

from assets import BIN_XY, CLASS_NAMES, DROP_HEIGHT, make_scene
from envs.panda_env import PandaEnv

# Objects spawn in front of the robot, clear of every bin, with a random yaw.
SPAWN_BOUNDS = np.array([[0.36, -0.17], [0.49, 0.17]])


class PandaTrashSortEnv(PandaEnv):
    """Observation (25 numbers):
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

    def __init__(self, render_mode=None, max_episode_steps=400, object_class=None, **kwargs):
        super().__init__(make_scene(), render_mode=render_mode, **kwargs)
        self.max_episode_steps = max_episode_steps
        self.object_class = object_class  # None picks a random class each reset.
        self.object_ids = [self.model.body(f"object{i}").id for i in range(len(CLASS_NAMES))]
        joints = [self.model.joint(f"object{i}_joint").id for i in range(len(CLASS_NAMES))]
        self.object_qpos = [self.model.jnt_qposadr[j] for j in joints]
        self.object_dofs = [self.model.jnt_dofadr[j] for j in joints]
        self.observation_space = spaces.Box(-np.inf, np.inf, (25,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        with self._lock():
            # Every object goes back to its parking spot under the table.
            mujoco.mj_resetData(self.model, self.data)
            self._reset_robot()
            chosen = (options or {}).get("object_class", self.object_class)
            self.target = int(
                self.np_random.integers(len(CLASS_NAMES)) if chosen is None else chosen
            )
            # Move the chosen object onto the table. Free joint qpos = [x, y, z, quaternion].
            xy = self.np_random.uniform(*SPAWN_BOUNDS)
            yaw = self.np_random.uniform(-np.pi, np.pi)
            q = self.object_qpos[self.target]
            self.data.qpos[q : q + 7] = [*xy, 0.003, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            mujoco.mj_step(self.model, self.data, nstep=100)  # Settle onto the table.
            mujoco.mj_forward(self.model, self.data)
            self.steps = 0
            observation = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return observation, {}

    def step(self, action):
        with self._lock():
            self._apply_action(action)
            self.steps += 1
            truncated = self.steps >= self.max_episode_steps
            observation = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return observation, self._compute_reward(), False, truncated, {}

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

    def _get_obs(self):
        seen = self.perceive()
        grasp = self.data.site_xpos[self.site_id]
        drop = np.r_[BIN_XY[seen["class_id"]], DROP_HEIGHT]
        return np.concatenate(
            [
                self.get_robot_state(),
                seen["position"],
                seen["position"] - grasp,
                seen["up"],
                seen["velocity"],
                drop - seen["position"],
                np.eye(len(CLASS_NAMES))[seen["class_id"]],
            ]
        ).astype(np.float32)

    def _compute_reward(self):
        """Reward is always 0 for now."""
        return 0.0
