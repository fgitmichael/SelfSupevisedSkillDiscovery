import numpy as np
from tqdm import tqdm

from self_supervised.algo.trainer import SelfSupTrainer
from self_supervised.base.data_collector.data_collector import \
    PathCollectorSelfSupervised
from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.base.algo.algo_base import BaseRLAlgorithmSelfSup
from self_supervised.algo.trainer_mode_latent import ModeLatentTrainer

import rlkit.torch.pytorch_util as ptu


class SelfSupCombAlgo(BaseRLAlgorithmSelfSup):

    def __init__(self,
                 trainer:
                 ):