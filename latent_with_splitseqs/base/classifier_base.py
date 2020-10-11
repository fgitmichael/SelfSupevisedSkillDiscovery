import abc
import torch

from code_slac.network.base import BaseNetwork


class SplitSeqClassifierBase(BaseNetwork, metaclass=abc.ABCMeta):

    def __init__(self,
                 obs_dim,
                 used_dims=None
                 ):
        super(SplitSeqClassifierBase, self).__init__()

        if used_dims is not None:
            assert isinstance(used_dims, list) \
                   or isinstance(used_dims, tuple)
            self.used_dims = used_dims
        else:
            self.used_dims = [i for i in range(obs_dim)]

    def _check_inputs(self, obs_seq, skill):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        if skill is not None:
            assert skill.size(batch_dim) == obs_seq.size(batch_dim)
            assert skill.size(data_dim) == self.skill_dim
            assert len(skill.shape) == 2
        assert len(obs_seq.shape) == 3

    def forward(self,
                obs_seq,
                skill=None
                ):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        self._check_inputs(
            obs_seq=obs_seq,
            skill=skill
        )
        obs_seq = obs_seq[..., self.used_dims]

        if self.training:
            return self.train_forwardpass(
                obs_seq=obs_seq,
                skill=skill,
            )

        else:
            with torch.no_grad():
                return self.eval_forwardpass(
                    obs_seq=obs_seq,
                    skill=skill,
                )

    @abc.abstractmethod
    def train_forwardpass(
            self,
            obs_seq,
            skill,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def eval_forwardpass(
            self,
            obs_seq,
            skill,
    ):
        raise NotImplementedError
