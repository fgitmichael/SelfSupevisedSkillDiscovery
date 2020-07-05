import numpy as np
import gym
from rlkit.envs.wrappers import ProxyEnv
from gym.spaces import Box

from code_slac.env.ordinary_env import OrdinaryEnvForPytorch


# Copied from module rlkit
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
                 gym_id: str,
                 action_repeat: int = 1,
                 obs_type: str ='state',
                 normalize_states: bool = False,
                 render_kwargs: bool = None):
        super(NormalizedBoxEnvForPytorch, self).__init__()

        assert obs_type in self.keys

        # Only change to OrdinaryEnvForPytorch: Use NormalizedBoxEnv
        self.normalize_states = normalize_states
        env_to_wrap = gym.make(gym_id)
        obs_space = env_to_wrap.observation_space

        low = obs_space.low
        high = obs_space.high
        bound_above = bool(np.prod(obs_space.bounded_above.astype(np.int)))
        bound_below = bool(np.prod(obs_space.bounded_below.astype(np.int)))
        bounded = bound_above and bound_below

        self.env = NormalizedBoxEnv(env_to_wrap)
        self.estimate_obs_stats(num_data=100000)

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

    def estimate_obs_stats(self,
                           num_data: int):
        # Sample states for estimation of obs stats
        obs_list = []
        done = False
        self.env.reset()
        for _ in range(num_data):
            obs = self.env.observation_space.sample()
            obs_list.append(obs)
        obs_array = np.stack(obs_list, axis=0)

        self.env.estimate_obs_stats(obs_array)
        self.obs_mean = self.env._obs_mean
        self.obs_std =  self.env._obs_std
        self.env._should_normalize = True

    @property
    def state_normalization(self):
        return self.env._should_normalize

