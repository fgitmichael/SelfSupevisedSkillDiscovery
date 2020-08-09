from code_slac.network.latent import Gaussian

from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise_seqwise import \
    BiRnnStepwiseSeqWiseClassifier

class BiRnnStepwiseSeqwiseNoidClassifier(BiRnnStepwiseSeqWiseClassifier):

    def create_classifier(self,
                          input_size,
                          output_size,
                          hidden_sizes):
        return Gaussian(
            input_dim=input_size,
            output_dim=output_size,
            hidden_units=hidden_sizes
        )



