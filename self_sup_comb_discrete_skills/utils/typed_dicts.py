import numpy as np
from prodict import Prodict
import torch

import self_sup_combined.utils.typed_dicts as tdssc
from self_supervised.utils import typed_dicts as td


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


class TransitonModeMappingDiscreteSkills(td.TransitionModeMapping):
    skill_id: np.ndarray

    def __init__(self,
                 obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminal: np.ndarray,
                 next_obs: np.ndarray,
                 mode: np.ndarray,
                 skill_id: np.ndarray,
                 agent_infos=None,
                 env_infos=None,
                 ):

        Prodict.__init__(
            self,
            obs=obs,
            action=action,
            reward=reward,
            terminal=terminal,
            next_obs=next_obs,
            mode=mode,
            skill_id=skill_id,
            agent_infos=agent_infos,
            env_infos=env_infos
        )