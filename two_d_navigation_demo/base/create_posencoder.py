from diayn_rnn_seq_rnn_stepwise_classifier.networks.pos_encoder_oh import PositionalEncodingOh
from diayn_rnn_seq_rnn_stepwise_classifier.networks.positional_encoder import PositionalEncoding

from mode_disent_no_ssm.utils.empty_network import Empty

from seqwise_cont_skillspace.networks.nooh_encoder import NoohPosEncoder
from seqwise_cont_skillspace.networks.transformer_stack_pos_encoder import PositionalEncodingTransformerStacked


def create_pos_encoder(
        feature_dim: int,
        seq_len: int,
        pos_encoder_variant: str,
        encoding_dim=None
) -> dict:
    """
    Args:
        feature_dim
        seq_len
        pos_encoder_variant
        encoding_dim                : only used
                                    : if pos_encoder_varaint is 'cont_encoder'
    Return dict:
        pos_encoder                 : nn.Module
        pos_encoded_feature_dim     : int (= feature_dim + pos_encoding_dim)
    """
    if encoding_dim is not None:
        assert pos_encoder_variant=='cont_encoder'

    minimum_input_size_step_classifier = feature_dim
    if pos_encoder_variant=='transformer':
        pos_encoder = PositionalEncoding(
            d_model=minimum_input_size_step_classifier,
            max_len=seq_len,
            dropout=0.1
        )
        pos_encoded_feature_dim = minimum_input_size_step_classifier

    elif pos_encoder_variant=='transformer_stacked':
        pos_encoder = PositionalEncodingTransformerStacked(
            d_model=minimum_input_size_step_classifier,
            max_len=seq_len,
            dropout=0.1
        )
        pos_encoded_feature_dim = minimum_input_size_step_classifier * 2

    elif pos_encoder_variant=='oh_encoder':
        pos_encoder = PositionalEncodingOh()
        pos_encoded_feature_dim = minimum_input_size_step_classifier + seq_len

    elif pos_encoder_variant=='cont_encoder':
        pos_encoder = NoohPosEncoder(
            encode_dim=encoding_dim,
            max_seq_len=300,
        )
        pos_encoded_feature_dim = minimum_input_size_step_classifier + \
                                  pos_encoder.encode_dim

    elif pos_encoder_variant=='empty':
        pos_encoder = Empty()
        pos_encoded_feature_dim = minimum_input_size_step_classifier

    else:
        raise NotImplementedError(
            "{} encoding is not implement".format(pos_encoder_variant))

    return dict(
        pos_encoder=pos_encoder,
        pos_encoded_feature_dim=pos_encoded_feature_dim
    )