import torch
import numpy as np
from torch.nn import functional as F
from typing import List

from diayn_seq_code_revised.algo.seqwise_algo_revised import SeqwiseAlgoRevised

from self_supervised.utils import typed_dicts as td
from self_supervised.base.replay_buffer.env_replay_buffer import NormalSequenceReplayBuffer

from cont_skillspace.data_collector.seq_collector_optional_skill_id import \
    SeqCollectorRevisedOptionalSkillId
from cont_skillspace.networks.rnn_vae_classifier import RnnVaeClassifierContSkills
from cont_skillspace.utils.get_colors import get_colors

import rlkit.torch.pytorch_util as ptu


class SeqwiseAlgoRevisedContSkills(SeqwiseAlgoRevised):

    def _log_perf_eval(self, epoch):
        classifier_accuracy_eval, post_fig = self._classfier_perf_eval()

        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Rnn Debug/Classfier accuracy eval",
            scalar_value=classifier_accuracy_eval,
            global_step=epoch
        )

        self.diagnostic_writer.writer.writer.add_figure(
            tag="Rnn Debug/Mode Post Plot of evaluation sequences from environment",
            figure=post_fig,
            global_step=epoch
        )

    def _get_paths_mode_influence_test(self, num_paths=1, seq_len=200)\
            -> List[td.TransitionModeMapping]:
        assert isinstance(self.seq_eval_collector, SeqCollectorRevisedOptionalSkillId)

        for skill_id, skill in enumerate(
                self.seq_eval_collector.skill_selector.get_skill_grid()):

            self.seq_eval_collector.skill = skill
            self.seq_eval_collector.collect_new_paths(
                seq_len=seq_len,
                num_seqs=num_paths,
                skill_id=skill_id # Use the Option to assign skill id
                # (new in SeqCollectorRevisedOptionalKSkillid)
            )

        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()
        assert type(mode_influence_eval_paths) is list
        assert type(mode_influence_eval_paths[0]) is td.TransitonModeMappingDiscreteSkills

        return mode_influence_eval_paths

    @torch.no_grad()
    def _classfier_perf_on_memory(self):
        len_memory = self.batch_size

        batch_size = len_memory

        assert isinstance(self.replay_buffer, NormalSequenceReplayBuffer)
        batch = self.replay_buffer.random_batch_bsd_format(
            batch_size=batch_size)

        assert isinstance(self.trainer.df, RnnVaeClassifierContSkills)
        pred_skill_dist = self.trainer.df(
            ptu.from_numpy(batch.next_obs)
        )

        df_accuracy = F.mse_loss(pred_skill_dist.loc, ptu.from_numpy(batch.mode))

        return df_accuracy

    @torch.no_grad()
    def _classfier_perf_eval(self):

        num_paths = 2
        eval_paths = self._get_paths_mode_influence_test(
            num_paths=num_paths,
            seq_len=self.seq_len,
        )

        assert type(eval_paths[0]) == td.TransitonModeMappingDiscreteSkills

        obs_dim = eval_paths[0].obs.shape[0]

        next_obs = []
        mode = []
        skill_id = []
        for path in eval_paths:
            next_obs.append(path.next_obs)
            mode.append(path.mode)
            skill_id.append(path.skill_id)

        next_obs = ptu.from_numpy(
            np.stack(next_obs, axis=0)
        ).transpose(-1, -2)
        mode = ptu.from_numpy(
            np.stack(mode, axis=0)
        ).transpose(-1, -2)
        skill_id = ptu.from_numpy(
            np.stack(skill_id, axis=0)
        ).transpose(-1, -2)

        # assert next_obs.shape \
        #    == torch.Size((num_paths * self.policy.skill_dim, self.seq_len, obs_dim))
        assert next_obs.shape \
               == torch.Size((len(eval_paths), self.seq_len, obs_dim))

        pred_skill_dist = self.trainer.df(
            next_obs,
        )

        df_accuracy = F.mse_loss(pred_skill_dist.loc, mode)

        posterior_fig = self._plot_posterior(
            post_dist=pred_skill_dist,
            skills_gt_seq=mode,
            skill_id_seq=skill_id
        )

        return df_accuracy, posterior_fig

    def _plot_posterior(self,
                        post_dist,
                        skills_gt_seq,
                        skill_id_seq):
        """
        Args:
            post_dist               : (N, S, skill_dim) distribution
            skills_gt_seq           : (N, S, skill_dim) skills
            skill_id_seq            : (N, S, 1)
        """
        batch_size = self.batch_size

        assert skills_gt_seq.size(-1) == 2
        assert isinstance(self.replay_buffer, NormalSequenceReplayBuffer)
        assert post_dist.batch_shape == skills_gt_seq.shape

        assert isinstance(self.trainer.df, RnnVaeClassifierContSkills)
        assert len(post_dist.batch_shape) == 3

        skill_ids = ptu.get_numpy(skill_id_seq[:, 0]).astype(np.int)
        skill_ids_unique = np.unique(ptu.get_numpy(skill_id_seq[:, 0])).astype(np.int)

        color_array = get_colors()
        plt.clf()
        plt.interactive(False)
        _, axes = plt.subplots()
        lim = [-3., 3.]
        axes.set_ylim(lim)
        axes.set_xlim(lim)
#        for idx, skill_gt_seq in enumerate(skills_gt_seq):
        for id in skill_ids_unique:
            id_idx = skill_ids == id
            id_idx = id_idx.squeeze()
            assert skills_gt_seq[id_idx, ...].shape[1:] == skills_gt_seq.shape[1:]

            plt.scatter(
                ptu.get_numpy(post_dist.loc[id_idx, :, 0].reshape(-1)),
                ptu.get_numpy(post_dist.loc[id_idx, :, 1].reshape(-1)),
                label="skill_{}_({})".format(
                    id, ptu.get_numpy(skills_gt_seq[id_idx, 0][0])),
                c=color_array[id]
            )

        axes.legend()
        axes.grid(True)
        fig = plt.gcf()

        return fig
