from rlkit.core import logger
from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import \
    DIAYNTorchOnlineRLAlgorithm
import gtimer as gt

from latent_with_splitseqs.base.my_object_base import MyObjectBase


class SelfSupAlgoBase(DIAYNTorchOnlineRLAlgorithm, MyObjectBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch_cnt = None

    @property
    def _objs_to_save(self):
        objs_to_save = super()._objs_to_save
        return dict(
            **objs_to_save,
            replay_buffer=self.replay_buffer,
            trainer=self.trainer,
            expl_data_collector=self.expl_data_collector,
            eval_data_collector=self.eval_data_collector,
            epoch_cnt=self.epoch_cnt,
        )

    @property
    def epoch_cnt(self):
        return self._epoch_cnt

    @epoch_cnt.setter
    def epoch_cnt(self, epoch):
        self._epoch_cnt = epoch

    def _end_epoch(self, epoch):
        """
        Change order compared to base method
        """
        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        self.epoch_cnt = epoch
        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch=epoch)

        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch)

