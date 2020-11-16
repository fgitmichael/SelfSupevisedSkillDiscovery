from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

from latent_with_splitseqs.base.tb_logging_base import PostEpochDiagnoWritingBase


class PostEpochTbLogger(PostEpochDiagnoWritingBase):

    def __init__(self,
                 replay_buffer,
                 trainer,
                 **kwargs):
        super(PostEpochTbLogger, self).__init__(**kwargs)
        self.replay_buffer = replay_buffer
        self.trainer = trainer

    def __call__(self, *args, **kwargs):
        self.tb_logging(*args, **kwargs)

    def tb_logging(self, epoch):
        to_log = {
            'Replay-Buffer Eval Stats': self.replay_buffer.get_diagnostics(),
            'Trainer Eval Stats': self.trainer.get_diagnostics(),
        }

        for base_tag, dict_to_log in to_log.items():
            self.diagno_writer.writer.log_dict_scalars(
                dict_to_log=dict_to_log,
                step=epoch,
                base_tag=base_tag,
            )
