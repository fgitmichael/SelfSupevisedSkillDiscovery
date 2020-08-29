import abc

from cnn_classifier_stepwise.base.cnn_classifier_stepwise_base import CnnStepwiseClassifierBaseDf
from cnn_classifier_stepwise_seqwise.base.cnn_seqwise_classifier_base import \
    CnnForClassificationSeqwiseBase


class CnnStepwiseSeqwiseClassifierDfBase(CnnStepwiseClassifierBaseDf, metaclass=abc.ABCMeta):

    def __init__(self,
                 *args,
                 seqwise_classifier: CnnForClassificationSeqwiseBase,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.seqwise_classifier = seqwise_classifier

    def _process_seq(self, seq):
        """
        Args:
            seq             : (N, S, data_dim)
        Return:
            feature_seq     : (N, S, feature_dim)
            features_raw    : (N, C, S, data_dim)
        """
        return self.feature_extractor(
            seq,
            return_features_raw=True
        )






