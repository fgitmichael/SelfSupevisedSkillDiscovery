import torch
import numpy as np
from typing import List


from diayn_with_rnn_classifier.algo.seq_wise_algo_classfier_perf_logging import \
    SeqWiseAlgoClassfierPerfLogging

from diayn_seq_code_revised.data_collector.seq_collector_revised_discrete_skills import \
    SeqCollectorRevisedDiscreteSkills
from diayn_seq_code_revised.data_collector.seq_collector_revised import \
    SeqCollectorRevised
from diayn_seq_code_revised.algo.seqwise_algo_revised import \
    SeqwiseAlgoRevisedDiscreteSkills

import self_supervised.utils.typed_dicts as td

import rlkit.torch.pytorch_util as ptu


class SeqwiseAlgoRevisedHighdim(SeqwiseAlgoRevisedDiscreteSkills):

    def __init__(self,
                 *args,
                 seqpixel_eval_collector,
                 seq_len_eval=200,
                 **kwargs):
        super(SeqwiseAlgoRevisedHighdim, self).__init__(
            *args,
            **kwargs
        )
        self.seq_len_eval = seq_len_eval
        self.seqpixel_eval_collector = seqpixel_eval_collector

    def set_next_skill(self, data_collector: SeqCollectorRevised):
        data_collector.skill_reset()

    def _get_paths_mode_influence_test_pixel(self, num_paths=1, seq_len=200) \
        -> List[dict]:

        for _ in range(num_paths):
            for skill in range(self.policy.skill_dim):
                self.seqpixel_eval_collector.set_skill(skill)
                self.seqpixel_eval_collector.collect_new_paths(
                    seq_len=seq_len,
                    num_seqs=1,
                )

        mode_influence_eval_paths_pixel = self.seqpixel_eval_collector.get_epoch_paths()
        return mode_influence_eval_paths_pixel

    def write_mode_influence_and_log(self, epoch):
        paths_pixel = self._get_paths_mode_influence_test_pixel(
            seq_len=self.seq_len_eval,
        )
        self._write_mode_influence_and_log_pixel(paths_pixel, epoch)

        paths_normal = self._get_paths_mode_influence_test(
            num_paths=1,
            seq_len=self.seq_len_eval,
        )
        self._write_mode_influence_and_log(paths=paths_normal, epoch=epoch)

    def _write_mode_influence(self,
                              path,
                              obs_dim,
                              action_dim,
                              epoch,
                              obs_lim=None,
                              ):
        path.obs = path.obs[:self.trainer.df.obs_dimensions_used]
        super()._write_mode_influence(
            path=path,
            obs_dim=self.trainer.df.obs_dimensions_used,
            action_dim=action_dim,
            epoch=epoch
        )

    def _write_mode_influence_and_log_pixel(self, paths, epoch):

        # Create videos for each path
        for path in paths:
            pixel = path['pixel_obs']
            skills = path['mode']
            T_dim = 0
            C_dim = -1
            H_dim = -3
            W_dim = -2

            assert len(pixel.shape) == 4
            assert pixel.shape[C_dim] == 3
            seq_len = pixel.shape[0]
            assert skills.shape[-1] == seq_len

            pixel_transposed = np.transpose(pixel, (T_dim, C_dim, H_dim, W_dim))
            pixel_tensor = torch.from_numpy(pixel_transposed).unsqueeze(dim=0)
            self.diagnostic_writer.writer.writer.add_video(
                tag='skill {}'.format(skills[:, 0]),
                vid_tensor=pixel_tensor,
                global_step=epoch,
            )


class SeqwiseAlgoRevisedDiscreteSkillsHighdim(SeqwiseAlgoRevisedHighdim):

    def _get_paths_mode_influence_test(self,
                                       num_paths=1,
                                       seq_len=200) \
            -> List[td.TransitonModeMappingDiscreteSkills]:
        assert isinstance(self.seq_eval_collector, SeqCollectorRevisedDiscreteSkills)

        for id, skill in enumerate(
                self.seq_eval_collector.skill_selector.get_skill_grid()):
            self.seq_eval_collector.skill = dict(
                skill=skill,
                id=id
            )
            self.seq_eval_collector.collect_new_paths(
                seq_len=seq_len,
                num_seqs=num_paths
            )

        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()
        return mode_influence_eval_paths

    def _get_paths_mode_influence_test_pixel(
            self,
            num_paths=1,
            seq_len=200) -> List[dict]:
        assert isinstance(self.seqpixel_eval_collector, SeqCollectorRevisedDiscreteSkills)

        for id, skill in enumerate(
                self.seqpixel_eval_collector.skill_selector.get_skill_grid()):
            self.seqpixel_eval_collector.skill = dict(
                skill=skill,
                id=id
            )
            self.seqpixel_eval_collector.collect_new_paths(
                seq_len=seq_len,
                num_seqs=num_paths
            )

        mode_influence_eval_paths = self.seqpixel_eval_collector.get_epoch_paths()
        return mode_influence_eval_paths
