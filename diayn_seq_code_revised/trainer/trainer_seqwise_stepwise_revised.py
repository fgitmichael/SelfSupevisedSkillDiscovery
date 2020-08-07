from diayn_rnn_seq_rnn_stepwise_classifier.trainer.diayn_step_wise_and_seq_wise_trainer \
    import DIAYNStepWiseSeqWiseRnnTrainer


class DIAYNAlgoStepwiseSeqwiseRevisedTrainer(DIAYNStepWiseSeqWiseRnnTrainer):
    @property
    def num_skills(self):
        return self.df.output_size
