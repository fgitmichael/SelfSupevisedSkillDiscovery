import gym
from typing import Type
from functools import wraps
import copy


def wrap_env_action_repeat(
        env_class_in: Type[gym.Env],
        num_action_repeat: int = 1,
):
    env_class = copy.deepcopy(env_class_in)
    orig_step = env_class.step

    @wraps(orig_step)
    def new_step(self, *args, **kwargs):
        step_return = None
        for _ in range(num_action_repeat):
            step_return = orig_step(self, *args, **kwargs)
        assert step_return is not None

        return step_return

    env_class.step = new_step

    return env_class

