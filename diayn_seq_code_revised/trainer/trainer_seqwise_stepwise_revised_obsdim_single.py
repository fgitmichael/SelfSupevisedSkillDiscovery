from itertools import chain


from diayn_seq_code_revised.trainer.trainer_seqwise_stepwise_revised import \
    DIAYNAlgoStepwiseSeqwiseRevisedTrainer

class DIAYNAlgoStepwiseSeqwiseRevisedObsDimSingleTrainer(
    DIAYNAlgoStepwiseSeqwiseRevisedTrainer):

    def create_optimizer_step(self, optimizer_class, df_lr):
        return optimizer_class(
            chain(
                self.df.classifier.parameters(),
                self.df.hidden_features_dim_matcher.parameters(),
            )
        )
