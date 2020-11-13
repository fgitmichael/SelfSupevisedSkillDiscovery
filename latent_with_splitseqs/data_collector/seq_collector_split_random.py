from warnings import warn
import torch

from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq
from latent_with_splitseqs.data_collector.random_rollouter import RollouterRandom

from diayn_seq_code_revised.base.rollouter_base import RollouterBase


class SeqCollectorSplitSeqRandom(SeqCollectorSplitSeq):

    def __init__(self,
                 *args,
                 policy,
                 **kwargs
                 ):
        if policy is not None:
            warn('For SeqCollecotrSplitSeqRandom the policy is not used. '
                 'Input policy=None to suppress the warning')

        super(SeqCollectorSplitSeqRandom, self).__init__(
            *args,
            policy=policy,
            **kwargs,
        )

    def create_rollouter(
            self,
            env,
            policy,
            reset_env_after_collection=False,
    ) -> RollouterBase:
        return RollouterRandom(
            env=env,
        )

    @property
    def skill(self):
        raise NotImplementedError('Policy is not used for random action sampling')

    @skill.setter
    def skill(self, skill: torch.Tensor):
        raise NotImplementedError('Policy is not used for random action sampling')
