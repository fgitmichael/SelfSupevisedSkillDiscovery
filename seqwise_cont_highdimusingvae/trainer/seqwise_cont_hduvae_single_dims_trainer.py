from itertools import chain

from seqwise_cont_highdimusingvae.trainer.seqwise_cont_highdimusingvae_trainer \
    import ContSkillTrainerSeqwiseStepwiseHighdimusingvae


class ContSkillTrainerSeqwiseStepwiseHighdimusingvaeSingleDims(
    ContSkillTrainerSeqwiseStepwiseHighdimusingvae
):

    def create_optimizer_step(self, optimizer_class, df_lr):
        return optimizer_class(
            chain(
                self.df.classifier.parameters(),
                self.df.hidden_features_dim_matcher.parameters(),
            ),
            lr=df_lr,
        )
