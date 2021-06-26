import gym
from typing import Type
from functools import wraps
import copy


def wrap_env_action_repeat(
        env_class_in: Type[gym.Env],
        num_action_repeat: int = 1,
) -> Type[gym.Env]:

    class ActionRepeatClass(env_class_in):
        def step(self, *args, **kwargs):
            step_return = None
            for _ in range(num_action_repeat):
                step_return = super().step(*args, **kwargs)
            assert step_return is not None

            return step_return

    return ActionRepeatClass
