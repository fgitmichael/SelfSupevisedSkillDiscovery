import torch

from diayn_seq_code_revised.networks.my_gaussian \
    import GaussianWrapper as Gaussian

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

    def _classify_stepwise(self, hidden_seq):
        hidden_seq = hidden_seq.detach()

        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        batch_size = hidden_seq.size(batch_dim)
        seq_len = hidden_seq.size(seq_dim)
        data_dim = hidden_seq.size(data_dim)

        hidden_seq_pos_encoded = self.pos_encoder(hidden_seq)
        assert hidden_seq.shape == torch.Size(
            (batch_size,
             seq_len,
             self.rnn_params['num_features'])
        )

        classified = self.classifier(hidden_seq_pos_encoded)
        assert classified.batch_shape == torch.Size(
            (batch_size,
             seq_len,
             self.classifier.output_size)
        )

        return classified
