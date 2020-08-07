import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
from operator import itemgetter
from itertools import chain

from diayn_with_rnn_classifier.trainer. \
    seq_wise_trainer_with_diayn_classifier_vote \
    import DIAYNTrainerMajorityVoteSeqClassifier
from diayn_with_rnn_classifier.trainer.diayn_trainer_modularized import \
    DIAYNTrainerModularized
from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise import \
    BiRnnStepwiseClassifier
from diayn_rnn_seq_rnn_stepwise_classifier.trainer.diayn_step_wise_and_seq_wise_trainer \
    import DIAYNStepWiseSeqWiseRnnTrainer

import self_supervised.utils.my_pytorch_util as my_ptu

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict

class DIAYNAlgoStepwiseSeqwiseRevisedTrainer(DIAYNStepWiseSeqWiseRnnTrainer):

    @property
    def num_skills(self):
        return self.df.output_size
