# COMP6008_panda_trash_sorting

Panda trash-sorting RL env (MuJoCo + Gymnasium). A Franka Panda picks up one piece of trash (can, bottle or paper) and drops it into the matching bin.

The package layout follows [panda_mujoco_gym](https://github.com/zichunxx/panda_mujoco_gym) and [gym-hil](https://github.com/huggingface/gym-hil): a base robot env, task envs that subclass it, a `controllers` module, and scene assets kept apart from the env code.

## Layout

```
envs/
  __init__.py            Registers the env IDs with Gymnasium.
  panda_env.py           PandaEnv: robot only. End-effector control, viewer, renderer.
  trash_sort.py          PandaTrashSortEnv: objects, bins, observation, reward.
controllers/
  ik.py                  Damped least squares inverse kinematics.
assets/
  panda.py               Downloads the Panda from mujoco-menagerie into the user cache.
  meshes.py              Visual-only meshes for the can and crumpled paper.
  scene.py               Builds the scene XML: robot tuning, table, bins, trash. Holds bin and object constants.
examples/
  scripted_sort.py       Hand-coded pick-and-place. Physics check and RL baseline.
tests/                   pytest: env checker, random rollouts, rendering.
```

## Environments

| Env ID                     | Object class             |
| -------------------------- | ------------------------ |
| `PandaTrashSort-v0`        | random each reset        |
| `PandaTrashSortCan-v0`     | can                      |
| `PandaTrashSortBottle-v0`  | bottle                   |
| `PandaTrashSortPaper-v0`   | paper                    |

Action is `[dx, dy, dz, gripper]`, each in [-1, 1]. Gripper -1 closes, +1 opens. The observation layout is in the `PandaTrashSortEnv` docstring. Reward is always 0 for now.

## Requirements

- Python 3.13 (tested). Other 3.10+ versions should work.
- `mujoco` 3.x, `mujoco-menagerie`, `gymnasium` 1.x, `numpy`
- For training: `stable-baselines3`, `tensorboard`, `matplotlib`

Exact versions are in `pyproject.toml` and `requirements.txt`.

## Install (conda)

```bash
conda create -n comp6008 python=3.13
conda activate comp6008
pip install -e ".[train,test]"
python -m assets.panda   # Download the Panda into ~/.cache (macOS: ~/Library/Caches)
```

## Test

```bash
pytest tests
```

Watch the scripted robot sort one of each class. The MuJoCo viewer on macOS needs `mjpython`, which ships with `mujoco`:

```bash
mjpython examples/scripted_sort.py   # macOS
python examples/scripted_sort.py     # Linux / Windows
```

## Use in code

```python
import gymnasium as gym
import envs  # Registers the envs

env = gym.make("PandaTrashSort-v0", render_mode="rgb_array")
obs, info = env.reset(seed=0, options={"object_class": 2})  # 0 can, 1 bottle, 2 paper. Omit for random.
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
frame = env.render()  # 480x640 RGB image
```
