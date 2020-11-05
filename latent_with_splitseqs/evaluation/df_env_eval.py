import torch
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from latent_with_splitseqs.base.df_env_evaluation_base import EnvEvaluationBase

from latent_with_splitseqs.data_collector.seq_collector_split \
    import SeqCollectorSplitSeq
from latent_with_splitseqs.base.classifier_base import SplitSeqClassifierBase
from latent_with_splitseqs.algo.post_epoch_func_gtstamp_wrapper \
    import post_epoch_func_wrapper

import self_supervised.utils.typed_dicts as td
import self_supervised.utils.my_pytorch_util as my_ptu

import rlkit.torch.pytorch_util as ptu

from seqwise_cont_skillspace.utils.get_colors import get_colors

from self_sup_combined.base.writer.is_log import is_log


class DfEnvEvaluationSplitSeq(EnvEvaluationBase):

    def __init__(
            self,
            *args,
            seq_eval_len: int,
            horizon_eval_len: int,
            seq_collector: SeqCollectorSplitSeq,
            df_to_evaluate: SplitSeqClassifierBase,
            num_paths_per_skill=1,
            **kwargs
    ):
        super(DfEnvEvaluationSplitSeq, self).__init__(
            *args,
            seq_collector=seq_collector,
            df_to_evaluate=df_to_evaluate,
            **kwargs
        )
        self.seq_eval_len = seq_eval_len
        self.horizon_eval_len = horizon_eval_len
        self.num_paths_per_skill = num_paths_per_skill

    @is_log()
    @post_epoch_func_wrapper(gt_stamp_name="df evaluation memory")
    def __call__(self, *args, **kwargs):
        super(DfEnvEvaluationSplitSeq, self).__call__(*args, **kwargs)

    def collect_skill_influence_paths(self) -> dict:
        """
        Collect Evaluation paths from eval environment
        Returns:
            skill_influence_paths       : td.TransitionModeMappingDiscrete
                                          consisting of (N, S, D) data
        """
        assert isinstance(self.seq_collector, SeqCollectorSplitSeq)

        for skill_id, skill in enumerate(
                self.seq_collector.skill_selector.get_skill_grid()):
            self.seq_collector.skill = skill
            self.seq_collector.collect_new_paths(
                seq_len=self.horizon_eval_len, # No splits
                horizon_len=self.horizon_eval_len,
                num_seqs=self.num_paths_per_skill,
                skill_id=skill_id,
            )
        skill_influence_paths = self.seq_collector.get_epoch_paths(transpose=False)
        self._check_skill_influence_paths(skill_influence_paths)
        
        skill_influence_paths_stacked_dict_bsd = self.prepare_paths(skill_influence_paths)
        return dict(
            **skill_influence_paths_stacked_dict_bsd
        )

    def prepare_paths(
            self,
            skill_influence_paths: List[td.TransitonModeMappingDiscreteSkills]
    ) -> dict:
        """
        Stack sequences to one numpy array and split paths
        Args:
            skill_influence_paths   : list of (D, S) paths
        Return:
            stacked_paths           : numpy array of dimension (B, S, D),
                                      where B denotes the number of paths
        """
        # Stack whole paths
        whole_paths_stacked = self._stack_paths(skill_influence_paths)
        whole_paths_stacked = td.TransitonModeMappingDiscreteSkills(
            **whole_paths_stacked
        )

        # Split paths and stack them
        split_paths = self.seq_collector.split_paths(
            split_seq_len=self.seq_eval_len,
            horizon_len=self.horizon_eval_len,
            paths_to_split=skill_influence_paths,
        )
        split_paths_stacked = self._stack_paths(split_paths)
        split_paths_stacked = td.TransitonModeMappingDiscreteSkills(
            **split_paths_stacked
        )

        return dict(
            skill_influence_whole_paths=whole_paths_stacked,
            skill_influence_split_paths=split_paths_stacked,
        )

    def plot_mode_influence_paths(
            self,
            *args,
            epoch,
            skill_influence_whole_paths,
            **kwargs
    ):
        seq_dim = 1
        data_dim = -1
        assert isinstance(skill_influence_whole_paths,
                          td.TransitonModeMappingDiscreteSkills)
        assert skill_influence_whole_paths.obs.shape[seq_dim] \
               == self.horizon_eval_len
        assert skill_influence_whole_paths.obs.shape[data_dim] \
               == self.seq_collector.obs_dim

        # Extract relevant dimensions and cat split seqs back to whole seqs
        obs = skill_influence_whole_paths.next_obs[..., self.obs_dims_to_log]
        action = skill_influence_whole_paths.action[..., self.action_dims_to_log]
        skill_id = skill_influence_whole_paths.skill_id.squeeze()[..., 0]

        # Observations
        if self.plot_skill_influence['obs']:
            self._plot_observation_mode_influence(
                epoch=epoch,
                obs=obs,
                skill_id=skill_id,
            )

        # Observations in one plot
        if self.plot_skill_influence['obs_one_plot']:
            self._plot_observation_mode_influence_all_in_one(
                epoch=epoch,
                obs=obs,
                skill_id=skill_id,
            )

        # Actions
        if self.plot_skill_influence['action']:
            self._plot_action_mode_influence(
                epoch=epoch,
                action=action,
                skill_id=skill_id,
            )

    @torch.no_grad()
    def apply_df(self, *args, skill_influence_split_paths, **kwargs) -> dict:
        next_obs = ptu.from_numpy(skill_influence_split_paths.next_obs)
        skill = ptu.from_numpy(skill_influence_split_paths.mode)
        skill_id = ptu.from_numpy(skill_influence_split_paths.skill_id)

        data_dim = -1
        assert next_obs.shape[:data_dim] \
               == skill_id.shape[:data_dim] \
               == skill.shape[:data_dim]

        ret_dict = my_ptu.eval(self.df_to_evaluate, obs_seq=next_obs)

        return ret_dict

    def plot_posterior(
            self,
            *args,
            epoch,
            skill_recon_dist,
            skill_influence_split_paths,
            **kwargs):
        assert isinstance(
            skill_influence_split_paths,
            td.TransitonModeMappingDiscreteSkills
        )

        skills = skill_influence_split_paths.mode
        skill_id = skill_influence_split_paths.skill_id
        assert skill_recon_dist.batch_shape == skills[:, 0, :].shape

        skill_id_squeezed = skill_id.squeeze().astype(np.int)
        skill_id_unique_np = np.unique(skill_id_squeezed).astype(np.int)
        color_array = get_colors()

        # Without Limits
        plt.clf()
        plt.interactive(False)
        _, axes = plt.subplots()
        for _id in skill_id_unique_np:
            self._scatter_post_skill(
                skill_id_np=skill_id_squeezed,
                id=_id,
                skill_recon_dist=skill_recon_dist,
                color_array=color_array,
                skills=skills,
            )
        axes.grid(True)
        axes.legend()
        fig_without_lim = plt.gcf()
        plt.close()

        # With Limits
        _, axes = plt.subplots()
        for _id in skill_id_unique_np:
            self._scatter_post_skill(
                skill_id_np=skill_id,
                id=_id,
                skill_recon_dist=skill_recon_dist,
                color_array=color_array,
                skills=skills,
            )
        axes.grid(True)
        lim = [-3., 3.]
        axes.set_ylim(lim)
        axes.set_xlim(lim)
        axes.legend()
        fig_with_lim = plt.gcf()
        plt.close()

        # Write
        figs = dict(
            no_lim=fig_without_lim,
            lim=fig_with_lim,
        )
        for key, fig in figs.items():
            self.diagno_writer.writer.writer.add_figure(
                tag="Skill Posterior Plot Eval/{}".format(key),
                figure=fig,
                global_step=epoch,
            )

    def classifier_evaluation(
            self,
            *args,
            epoch,
            skill_recon_dist,
            skill_influence_split_paths,
            **kwargs
    ):
        skill = ptu.from_numpy(skill_influence_split_paths.mode)
        df_accuracy_eval = F.mse_loss(skill_recon_dist.loc, skill[:, 0, :])

        self.diagno_writer.writer.writer.add_scalar(
            tag="Classifier Performance/Eval",
            scalar_value=df_accuracy_eval,
            global_step=epoch,
        )

    def _stack_paths(self, paths: List[td.TransitionMapping]):
        # Stack to one numpy array
        stacked_paths = {}
        for key, el in paths[0].items():
            stacked_paths[key] = np.stack(
                [path[key] for path in paths],
                axis=0
            )
        stacked_paths = td.TransitonModeMappingDiscreteSkills(**stacked_paths)

        return stacked_paths

    def _check_skill_influence_paths(
            self,
            skill_influence_paths: List[td.TransitonModeMappingDiscreteSkills]
    ):
        assert type(skill_influence_paths) is list
        assert type(skill_influence_paths[0]) \
               is td.TransitonModeMappingDiscreteSkills
        assert len(skill_influence_paths) < self.seq_collector.maxlen

    def _plot_observation_mode_influence(
            self,
            epoch,
            obs,
            skill_id,
            obs_lim=(-3., 3.),
    ):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        assert len(skill_id) == obs.shape[batch_dim]
        assert len(obs.shape) == 3
        num_obs_dims_used = obs.shape[data_dim]
        assert num_obs_dims_used == len(self.obs_dims_to_log)

        for obs_seq in obs:
            obs_seq_data_dim_first = np.transpose(obs_seq, axes=(1, 0))
            assert obs_seq_data_dim_first.shape[0] == num_obs_dims_used
            self.diagno_writer.writer.plot_lines(
                legend_str=["dim {}".format(i) for i in range(num_obs_dims_used)],
                tb_str="Mode Influence Test: Obs/Skill {}".format(skill_id),
                arrays_to_plot=obs_seq_data_dim_first,
                step=epoch,
                y_lim=obs_lim,
            )

    def _plot_observation_mode_influence_all_in_one(
            self,
            epoch,
            obs,
            skill_id,
            lim=None,
    ):
        if lim is None:
            lim = dict(
                x=[-2.2, 2.2],
                y=[-2.2, 2.2],
            )
        self.diagno_writer.writer.plot(
            obs[..., 0], obs[..., 1],
            tb_str="State Space Behaviour/Skill {}".format(skill_id),
            step=epoch,
            x_lim=lim['x'],
            y_lim=lim['y'],
        )

    def _plot_action_mode_influence(
            self,
            epoch,
            action,
            skill_id,
            lim=(-1.2, 1.2),
    ):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        assert action.shape[batch_dim] == skill_id.shape[0]
        assert len(action.shape) == 3
        assert len(skill_id.shape) == 1
        num_action_dims_used = action.shape[data_dim]
        assert num_action_dims_used == len(self.action_dims_to_log)

        for idx, action_seq in enumerate(action):
            action_seq_data_dim_first = np.transpose(action_seq, axes=(1, 0))
            assert action_seq_data_dim_first.shape[0] == num_action_dims_used
            self.diagno_writer.writer.plot_lines(
                legend_str=["dim {}".format(i) for i in range(num_action_dims_used)],
                tb_str="Mode Influence Test: Action/Skill {}".format(skill_id[idx]),
                arrays_to_plot=action_seq_data_dim_first,
                step=epoch,
                y_lim=lim,
            )

    def _scatter_post_skill(
            self,
            skill_id_np,
            id,
            skill_recon_dist,
            color_array,
            skills,
    ):
        """
        skill_id_np         : (N, S) array of skill_id
        """
        id_idx = np.unique(skill_id_np, axis=1) == id
        assert isinstance(id_idx, np.ndarray)
        id_idx = id_idx.squeeze()

        plt.scatter(
            ptu.get_numpy(skill_recon_dist.loc[id_idx, 0].reshape(-1)),
            ptu.get_numpy(skill_recon_dist.loc[id_idx, 1].reshape(-1)),
            label="skill_{}({})".format(id, skills[id_idx, 0][0]),
            c=color_array[id]
        )
