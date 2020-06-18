import numpy as np
import gym
from rlkit.envs.wrappers import ProxyEnv
from gym.spaces import Box

from code_slac.env.dm_control import DmControlEnvForPytorch
from code_slac.env.ordinary_env import OrdinaryEnvForPytorch


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env


class NormalizedBoxEnvForPytorch(OrdinaryEnvForPytorch):

    def __init__(self,
                 gym_id,
                 action_repeat=1,
                 obs_type='state',
                 normalize_states=False,
                 render_kwargs=None):
        super(DmControlEnvForPytorch, self).__init__()

        assert obs_type in self.keys

        # Only change to OrdinaryEnvForPytorch: Use NormalizedBoxEnv
        self.normalize_states = normalize_states
        env_to_wrap = gym.make(gym_id)
        low = env_to_wrap.observation_space.low
        high = env_to_wrap.observation_space.high
        if self.normalize_states:
            self.obs_mean = low + (high-low)/2
            self.obs_std = high - low
            self.env = NormalizedBoxEnv(env_to_wrap,
                                        obs_mean=self.obs_mean,
                                        obs_std=self.obs_std)

        else:
            self.env = NormalizedBoxEnv(env_to_wrap)
        # Only change to OrdinaryEnvForPytorch: Use NormalizedBoxEnv
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

    def denormalize(self, state):
        if self.normalize_states:
            denormalized = state * (self.obs_std + 1e-8) + self.obs_mean
        else:
            denormalized = state

        return  denormalized
