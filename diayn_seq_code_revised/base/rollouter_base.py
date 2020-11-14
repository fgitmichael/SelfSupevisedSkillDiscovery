import abc
import gym
from typing import Union

from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised, MakeDeterministicRevised

import self_supervised.utils.typed_dicts as td


class RollouterBase(object, metaclass=abc.ABCMeta):

    def __init__(self,
                 env: gym.Env,
                 ):
        self.env = env

    @abc.abstractmethod
    def do_rollout(
            self,
            seq_len) -> td.TransitionMapping:
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class RolloutWrapperBase(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def rollout(
            self,
            env: gym.Env,
            policy,
            **kwargs,
    ):
        raise NotImplementedError
