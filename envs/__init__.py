"""Gymnasium environments for a Franka Panda sorting trash in MuJoCo.

Importing this package registers every env in ``ENV_IDS`` with Gymnasium.
"""

from gymnasium.envs.registration import register

from envs.panda_env import PandaEnv
from envs.trash_sort import PandaTrashSortEnv

__all__ = ["PandaEnv", "PandaTrashSortEnv", "ENV_IDS"]

ENV_IDS = []

for object_class, suffix in [(None, ""), (0, "Can"), (1, "Bottle"), (2, "Paper")]:
    env_id = f"PandaTrashSort{suffix}-v0"
    register(
        id=env_id,
        entry_point="envs.trash_sort:PandaTrashSortEnv",
        kwargs={"object_class": object_class},
    )
    ENV_IDS.append(env_id)
