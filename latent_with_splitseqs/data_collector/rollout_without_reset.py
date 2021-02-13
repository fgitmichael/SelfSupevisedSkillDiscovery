import numpy as np
import gym
from typing import Union

from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised, MakeDeterministicRevised

from diayn_seq_code_revised.base.rollouter_base import RolloutWrapperBase

import self_supervised.utils.typed_dicts as td


class CollectSeqOverHorizonWrapper(RolloutWrapperBase):

    def __init__(self):
        self.rollout_fun = collect_seq_without_reset
        self._obs_now = None

    def rollout(
            self,
            env: gym.Env,
            policy,
            seq_len=None,
    ):
        path = self.rollout_fun(
            env=env,
            agent=policy,
            max_path_length=seq_len,
            current_observation=self._obs_now,
        )
        self._obs_now = path['next_observations'][-1]

        assert len(path['observations'].shape) == 2
        assert path['observations'].shape[-1] == env.observation_space.shape[0]

        return path

    def reset(self, reset_obs, **kwargs):
        self._obs_now = reset_obs


def collect_seq_without_reset(
        env,
        agent: Union[
            SkillTanhGaussianPolicyRevised,
            MakeDeterministicRevised,
        ],
        current_observation: np.ndarray = None,
        max_path_length: int = None,
        render=False,
        render_kwargs=None,
):
    if max_path_length is None:
        max_path_length = np.inf

    if render_kwargs is None:
        render_kwargs = {}

    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []

    if current_observation is None:
        o = env.reset()
    else:
        o = current_observation

    next_o = None
    path_length = 0

    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1

        if max_path_length == np.inf and d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)

    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)

    next_observations = np.concatenate(
        [observations[1:, :], np.expand_dims(next_o, 0)]
    )

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
