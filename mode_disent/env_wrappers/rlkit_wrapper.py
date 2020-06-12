import numpy as np
import gym
from rlkit.envs.wrappers import NormalizedBoxEnv

from code_slac.env.dm_control import DmControlEnvForPytorch
from code_slac.env.ordinary_env import OrdinaryEnvForPytorch


class NormalizedBoxEnvForPytorch(OrdinaryEnvForPytorch):

    def __init__(self,
                 gym_id,
                 action_repeat=1,
                 obs_type='state',
                 render_kwargs=None):
        super(DmControlEnvForPytorch, self).__init__()

        assert obs_type in self.keys

        # Only change to OrdinaryEnvForPytorch: Use NormalizedBoxEnv
        self.env = NormalizedBoxEnv(gym.make(gym_id))
        self.action_repeat = action_repeat
        self.obs_type = obs_type

        self.render_kwargs = dict(
            width=64,
            height=64,
            camera_id=0
        )
        if render_kwargs is not None:
            self.render_kwargs.update(render_kwargs)

        if obs_type == 'state':
            self.observation_space = self.env.observation_space
        elif obs_type == 'pixels':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.action_space = self.env.action_space
