from prodict import Prodict
import torch

import self_sup_combined.utils.typed_dicts as tdssc


class ModeTrainerDataMappingDiscreteSkills(
    tdssc.ModeTrainerDataMapping):
    skill_id: torch.Tensor

    def __init__(self,
                 skills_gt: torch.Tensor,
                 obs_seq: torch.Tensor,
                 skill_id: torch.Tensor):
        Prodict.__init__(
            self,
            skills_gt=skills_gt,
            obs_seq=obs_seq,
            skill_id=skill_id
        )
