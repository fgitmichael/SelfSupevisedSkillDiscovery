from itertools import chain

from seqwise_cont_skillspace.trainer.cont_skillspace_seqwise_trainer import \
    ContSkillTrainerSeqwiseStepwise

from seqwise_cont_skillspace.networks.bi_rnn_stepwise_seq_singledims_cont_output import \
    BiRnnStepwiseSeqWiseClassifierSingleDimsContOutput

class ContSkillTrainerSeqwiseStepwiseSingleDims(ContSkillTrainerSeqwiseStepwise):

    def create_optimizer_step(self, optimizer_class, df_lr):
        assert isinstance(self.df, BiRnnStepwiseSeqWiseClassifierSingleDimsContOutput)
        return optimizer_class(
            chain(
                self.df.classifier.parameters(),
                self.df.hidden_features_dim_matcher.parameters(),
                self.df.feature_decoder.parameters(),
            ),
            lr=df_lr,
        )
