import torch
from itertools import chain

from diayn_rnn_seq_rnn_stepwise_classifier.trainer.\
    diayn_step_wise_and_seq_wise_trainer import \
    DIAYNStepWiseSeqWiseRnnTrainer


class CnnStepwiseSeqwiseTrainer(DIAYNStepWiseSeqWiseRnnTrainer):

    def create_optimizer_seq(self, optimizer_class, df_lr):
        return optimizer_class(
            chain(
                self.df.seqwise_classifier.parameters(),
                self.df.feature_extractor.parameters()
            ),
            lr=df_lr
        )

    def create_optimizer_step(self, optimizer_class, df_lr):
        return optimizer_class(
            self.df.stepwise_classifier.parameters(),
            lr=df_lr
        )
