import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np

from seqwise_cont_skillspace.algo.algo_cont_skillspace import \
    SeqwiseAlgoRevisedContSkills

import self_supervised.utils.typed_dicts as td
from self_supervised.base.replay_buffer.env_replay_buffer import \
    NormalSequenceReplayBuffer

import rlkit.torch.pytorch_util as ptu

from seqwise_cont_skillspace.utils.get_colors import get_colors


class SeqwiseAlgoRevisedContSkillsHighdimusingvae(SeqwiseAlgoRevisedContSkills):

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

        ret_dict = self.trainer.df(
            next_obs, train=True
        )
        pred_skill_dist = ret_dict['skill_recon']['dist']
        pred_skill_dist_seq = ret_dict['classified_seqs']

        df_accuracy = F.mse_loss(
            pred_skill_dist.loc.reshape(*mode.shape),
            mode
        )
        df_accuracy_seq = F.mse_loss(pred_skill_dist_seq, mode[:, 0, :])

        figs = self._plot_posterior(
            post_dist=pred_skill_dist,
            skill_id_seq=skill_id
        )

        return df_accuracy, df_accuracy_seq, figs

    @torch.no_grad()
    def _classfier_perf_on_memory(self):
        batch_size = self.batch_size

        assert isinstance(self.replay_buffer, NormalSequenceReplayBuffer)
        batch = self.replay_buffer.random_batch_bsd_format(
            batch_size=batch_size)

        #assert isinstance(self.trainer.df, RnnVaeClassifierContSkills)
        ret_dict = self.trainer.df(
            ptu.from_numpy(batch.next_obs),
            train=True
        )

        df_accuracy = F.mse_loss(
            ret_dict['skill_recon']['dist'].loc.reshape(*batch.mode.shape),
            ptu.from_numpy(batch.mode)
        )

        return df_accuracy

    def _plot_posterior(self,
                        post_dist,
                        skill_id_seq: torch.Tensor,
                        skills_gt_seq=None,
                        ):
        """
        Args:
            post_dist               : (N * S, skill_dim) distribution
            skills_gt_seq           : (N, S, skill_dim) skills
            skill_id_seq            : (N, S, 1)
        """
        if skills_gt_seq is not None:
            raise ValueError

        post_dist.loc = post_dist.loc.reshape(
            *skill_id_seq.shape[:-1],
            post_dist.batch_shape[-1]
        )

        batch_size = self.batch_size

        assert isinstance(self.replay_buffer, NormalSequenceReplayBuffer)

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

            plt.scatter(
                ptu.get_numpy(post_dist.loc[id_idx, :, 0].reshape(-1)),
                ptu.get_numpy(post_dist.loc[id_idx, :, 1].reshape(-1)),
                label="skill_{}_".format(id),
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

            plt.scatter(
                ptu.get_numpy(post_dist.loc[id_idx, :, 0].reshape(-1)),
                ptu.get_numpy(post_dist.loc[id_idx, :, 1].reshape(-1)),
                label="skill_{}".format(id),
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
