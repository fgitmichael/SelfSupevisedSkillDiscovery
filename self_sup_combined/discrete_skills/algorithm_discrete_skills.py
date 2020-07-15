import numpy as np
from tqdm import tqdm
from typing import Dict
import torch
import gtimer as gt

from self_supervised.base.data_collector.data_collector import \
    PathCollectorSelfSupervised
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.base.algo.algo_base import BaseRLAlgorithmSelfSup

from self_sup_combined.algo.trainer import SelfSupCombSACTrainer
from self_sup_combined.algo.trainer_mode import ModeTrainer
from self_sup_combined.algo.algorithm import SelfSupCombAlgo
import self_sup_combined.utils.typed_dicts as tdssc

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import _get_epoch_timings



class SelfSupCombAlgoDiscrete(SelfSupCombAlgo):

    def __init__(self,
                 num_skills,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.num_skills = num_skills

    def set_next_skill(self):
        new_skill = np.random.randint(self.num_skills - 1)
        skill_vec = ptu.zeros(self.num_skills)
        skill_vec[new_skill] += 1
        self.policy.set_skill(skill_vec)


