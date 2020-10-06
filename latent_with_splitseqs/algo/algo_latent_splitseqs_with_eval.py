from typing import List
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import self_supervised.utils.typed_dicts as td
from self_supervised.base.replay_buffer.env_replay_buffer import \
    NormalSequenceReplayBuffer
import self_supervised.utils.my_pytorch_util as my_ptu

from latent_with_splitseqs.algo.algo_latent_splitseqs \
    import SeqwiseAlgoRevisedSplitSeqs
from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq

from seqwise_cont_skillspace.algo.algo_cont_skillspace import  SeqwiseAlgoRevisedContSkills

import rlkit.torch.pytorch_util as ptu

from seqwise_cont_skillspace.utils.get_colors import get_colors


class SeqwiseAlgoRevisedSplitSeqsEval(SeqwiseAlgoRevisedSplitSeqs):

    def __init__(self,
                 *args,
                 seq_eval_len,
                 horizon_eval_len,
                 mode_influence_plotting=True,
                 **kwargs,
                 ):
        super(SeqwiseAlgoRevisedSplitSeqsEval, self).__init__(
            *args,
            mode_influence_plotting=mode_influence_plotting,
            **kwargs
        )
        self.seq_eval_len = seq_eval_len
        self.horizon_eval_len = horizon_eval_len

    def _get_paths_mode_influence_test(self,
                                       num_paths=1,
                                       rollout_seqlengths_dict=None) \
            -> List[td.TransitionModeMapping]:
        assert isinstance(self.seq_eval_collector, SeqCollectorSplitSeq)

        if rollout_seqlengths_dict is None:
            seq_len = 30
            horizon_len = 60

        else:
            seq_len = rollout_seqlengths_dict['seq_len']
            horizon_len = rollout_seqlengths_dict['horizon_len']

        for skill_id, skill in enumerate(
                self.seq_eval_collector.skill_selector.get_skill_grid()):

            self.seq_eval_collector.skill = skill
            self.seq_eval_collector.collect_new_paths(
                seq_len=seq_len,
                horizon_len=horizon_len,
                num_seqs=num_paths,
                skill_id=skill_id # Use the Option to assign skill id
            )

        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()
        assert type(mode_influence_eval_paths) is list
        assert type(mode_influence_eval_paths[0]) \
               is td.TransitonModeMappingDiscreteSkills
        assert len(mode_influence_eval_paths) < self.seq_eval_collector.maxlen

        return mode_influence_eval_paths

    def _log_net_param_hist(self, epoch):
        for k, net in self.trainer.network_dict.items():
            for name, weight in net.named_parameters():
                self.diagnostic_writer.writer.writer. \
                    add_histogram(k + name, weight, epoch)
                if weight.grad is not None:
                    self.diagnostic_writer.writer.writer. \
                        add_histogram(f'{k + name}.grad', weight.grad, epoch)

    def write_mode_influence_and_log(self, epoch):
        paths = self._get_paths_mode_influence_test(
            rollout_seqlengths_dict=dict(
                seq_len=self.horizon_eval_len,
                horizon_len=self.horizon_eval_len,
            )
        )
        self._write_mode_influence_and_log(
            paths=paths,
            epoch=epoch,
        )

    @torch.no_grad()
    def _end_epoch(self, epoch):
        super(SeqwiseAlgoRevisedSplitSeqsEval, self)._end_epoch(epoch)
        if self.diagnostic_writer.is_log(epoch):
            self._log_net_param_hist(epoch)
            self.classifier_perf_eval_log(epoch)

    def classifier_perf_eval_log(self, epoch):
        classifier_accuracy_eval_ret_dict = self.classifier_perf_eval()
        classifier_accuracy_eval = classifier_accuracy_eval_ret_dict['df_accuracy']
        post_figures = classifier_accuracy_eval_ret_dict['figs']
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Classifier Performance/Eval",
            scalar_value=classifier_accuracy_eval,
            global_step=epoch,
        )
        for key, fig in post_figures.items():
            self.diagnostic_writer.writer.writer.add_figure(
                tag="Rnn Debug/Mode Post Plot of evaluation "
                    "sequences from environment {}"
                    .format(key),
                figure=fig,
                global_step=epoch
            )

        classifier_accuracy_memory = self.classifier_perf_memory()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Classifier Performance/Memory",
            scalar_value=classifier_accuracy_memory,
            global_step=epoch,
        )

    def classifier_perf_memory(self):
        assert isinstance(self.replay_buffer, NormalSequenceReplayBuffer)
        batch = self.replay_buffer.random_batch_bsd_format(
            batch_size=self.batch_size)

        ret_dict = my_ptu.eval(self.trainer.df,
                               obs_seq=ptu.from_numpy(batch.next_obs))
        skill_recon_dist = ret_dict['skill_recon_dist']

        df_accuracy_eval = F.mse_loss(
            skill_recon_dist.loc,
            ptu.from_numpy(batch.mode[:, 0, :]),
        )

        return df_accuracy_eval

    def classifier_perf_eval(self):
        eval_paths = self._get_paths_mode_influence_test(
            num_paths=4,
            rollout_seqlengths_dict=dict(
                seq_len=self.seq_eval_len,
                horizon_len=self.horizon_eval_len,
            )
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

        assert next_obs.shape \
               == torch.Size((len(eval_paths), self.seq_eval_len, obs_dim))

        ret_dict = my_ptu.eval(self.trainer.df, obs_seq=next_obs)
        skill_recon_dist = ret_dict['skill_recon_dist']

        df_accuracy_eval = F.mse_loss(skill_recon_dist.loc, mode[:, 0, :])

        figs = self._plot_posterior(
            post_dist=skill_recon_dist,
            skills_gt_seq=mode,
            skill_id_seq=skill_id,
        )

        return dict(
            df_accuracy=df_accuracy_eval,
            figs=figs,
        )

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

        assert post_dist.batch_shape == skills_gt_seq[:, 0, :].shape
        assert len(post_dist.batch_shape) == 2

        skill_ids = ptu.get_numpy(skill_id_seq[:, 0]).astype(np.int)
        skill_ids_unique = np.unique(ptu.get_numpy(skill_id_seq[:, 0])).astype(np.int)

        color_array = get_colors()
        plt.clf()
        plt.interactive(False)
        _, axes = plt.subplots()
        #        for idx, skill_gt_seq in enumerate(skills_gt_seq):
        for id in skill_ids_unique:
            id_idx = skill_ids == id
            id_idx = id_idx.squeeze()
            assert skills_gt_seq[id_idx, ...].shape[1:] == skills_gt_seq.shape[1:]

            plt.scatter(
                ptu.get_numpy(post_dist.loc[id_idx, 0].reshape(-1)),
                ptu.get_numpy(post_dist.loc[id_idx, 1].reshape(-1)),
                label="skill_{}({})".format(
                    id, ptu.get_numpy(skills_gt_seq[id_idx, 0][0])),
                c=color_array[id]
            )
        axes.grid(True)
        axes.legend()
        fig_without_lim = plt.gcf()
        plt.close()

        _, axes = plt.subplots()
        for id in skill_ids_unique:
            id_idx = skill_ids == id
            id_idx = id_idx.squeeze()
            assert skills_gt_seq[id_idx, ...].shape[1:] == skills_gt_seq.shape[1:]

            plt.scatter(
                ptu.get_numpy(post_dist.loc[id_idx, 0].reshape(-1)),
                ptu.get_numpy(post_dist.loc[id_idx, 1].reshape(-1)),
                label="skill_{}_({})".format(
                    id, ptu.get_numpy(skills_gt_seq[id_idx, 0][0][:2])),
                c=color_array[id]
            )
        lim = [-3., 3.]
        axes.set_ylim(lim)
        axes.set_xlim(lim)
        axes.legend()

        fig_with_lim = plt.gcf()

        return dict(
            no_lim=fig_without_lim,
            lim=fig_with_lim
        )
