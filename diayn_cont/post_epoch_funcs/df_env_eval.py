import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as torch_dist
import matplotlib.pyplot as plt

from latent_with_splitseqs.base.df_env_evaluation_base import EnvEvaluationBase

import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector.path_collector import MdpPathCollector

from code_slac.network.latent import Gaussian

import self_supervised.utils.my_pytorch_util as my_ptu

from seqwise_cont_skillspace.utils.get_colors import get_colors


class DfEnvEvaluationDIAYNCont(EnvEvaluationBase):

    def __init__(
            self,
            *args,
            seq_len,
            seq_collector: MdpPathCollector,
            df_to_evaluate: Gaussian,
            num_paths_per_skill=1,
            **kwargs,
    ):
        super().__init__(
            *args,
            seq_collector=seq_collector,
            df_to_evaluate=df_to_evaluate,
            **kwargs
        )
        self.seq_len = seq_len
        self.num_paths_per_skill = num_paths_per_skill

    def collect_skill_influence_paths(self) -> dict:
        assert isinstance(self.seq_collector, MdpPathCollector)

        skill_ids = []
        skills = []
        for skill_id, skill in enumerate(
            self.seq_collector._policy.skill_selector.get_skill_grid()
        ):
            self.seq_collector._policy.skill = skill
            self.seq_collector.collect_new_paths(
                max_path_length=self.seq_len,
                num_steps=self.seq_len,
                discard_incomplete_paths=False
            )
            skills.append(skill)
            skill_ids.append(skill_id)
        skill_influence_paths = self.seq_collector.get_epoch_paths()
        assert isinstance(skill_influence_paths, list)

        for skill_id, skill, path in zip(skill_ids, skills, skill_influence_paths):
            path['skill_id'] = skill_id
            path['skill'] = skill
        self._check_skill_influence_paths(skill_influence_paths)

        skill_influence_paths = self._stack_paths(skill_influence_paths)
        return skill_influence_paths

    def _check_skill_influence_paths(self, skill_influence_paths):
        assert isinstance(skill_influence_paths, list)
        assert isinstance(skill_influence_paths[0], dict)
        assert isinstance(skill_influence_paths[0]['observations'], np.ndarray)
        seq_dim = 0
        assert skill_influence_paths[0]['observations'].shape[seq_dim] == self.seq_len

    def _stack_paths(self, paths: list) -> dict:
        stacked_paths = {}
        for key, el in paths[0].items():
            if isinstance(el, np.ndarray):
                stacked_paths[key] = np.stack(
                    [path[key] for path in paths],
                    axis=0
                )
            else:
                stacked_paths[key] = [path[key] for path in paths]

        return stacked_paths

    def plot_mode_influence_paths(
            self,
            *args,
            epoch,
            observations,
            actions,
            skill_id,
            **kwargs
    ):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        assert len(observations.shape) == 3
        assert observations.shape[seq_dim] == self.seq_len
        assert actions.shape[seq_dim] == self.seq_len
        assert isinstance(skill_id, list)
        assert len(skill_id) == observations.shape[batch_dim]

        # Extract relevant dimensions
        obs = observations[..., self.obs_dims_to_log]
        action = actions[..., self.action_dims_to_log]
        skill = np.array(skill_id)


        if self.plot_skill_influence['obs']:
            self._plot_observation_mode_influence(
                epoch=epoch,
                obs=obs,
                skill_id=skill_id,
            )

        if self.plot_skill_influence['obs_one_plot']:
            self._plot_observation_mode_influence_all_in_one(
                epoch=epoch,
                obs=obs,
                skill_id=skill_id,
                skill=skill,
            )

        if self.plot_skill_influence['action']:
            self._plot_action_mode_influence(
                epoch=epoch,
                action=action,
                skill_id=skill_id,
            )

    def _plot_observation_mode_influence(
            self,
            epoch,
            obs,
            skill_id,
            obs_lim=(-3., 3.)
    ):
        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        for _skill_id, obs_seq in zip(skill_id, obs):
            obs_seq_data_dim_first = np.transpose(obs_seq, axes=(1, 0))
            assert obs_seq_data_dim_first.shape[0] == obs.shape[data_dim]
            self.diagno_writer.writer.plot_lines(
                legend_str=["dim {}".format(i) for i in range(obs.shape[data_dim])],
                tb_str=self.get_log_string(
                    "Mode Influence Test: Obs/Skill {}".format(_skill_id)),
                arrays_to_plot=obs_seq_data_dim_first,
                step=epoch,
                y_lim=obs_lim,
            )

    def _plot_observation_mode_influence_all_in_one(
            self,
            epoch,
            obs,
            skill_id,
            skill,
            lim=None,
    ):
        if lim is None:
            lim = dict(
                x=[-2.2, 2.2],
                y=[-2.2, 2.2]
            )

        batch_dim = 0
        seq_dim = 1
        data_dim = -1

        obs_dsb = np.transpose(obs, axes=(data_dim, seq_dim, batch_dim))
        labels = ["skill {} ({})".format(_skill_id, _skill)
                  for _skill_id, _skill in zip(skill_id, skill)]

        self.diagno_writer.writer.plot(
            obs_dsb[0], obs_dsb[1],
            tb_str=self.get_log_string(
                "Mode-Influence all skills in one plot/With limits"
            ),
            step=epoch,
            labels=labels,
            x_lim=lim['x'],
            y_lim=lim['y'],
        )

        self.diagno_writer.writer.plot(
            obs_dsb[0], obs_dsb[1],
            tb_str=self.get_log_string(
                "Mode-Influence all skills in one plot/Without Limits",
            ),
            step=epoch,
            labels=labels,
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
        assert action.shape[batch_dim] == len(skill_id)
        assert len(action.shape) == 3
        num_action_dims_used = action.shape[data_dim]
        assert num_action_dims_used == len(self.action_dims_to_log)

        for idx, action_seq in enumerate(action):
            action_seq_data_dim_first = np.transpose(action_seq, axes=(1, 0))
            assert action_seq_data_dim_first.shape[0] == num_action_dims_used
            self.diagno_writer.writer.plot_lines(
                legend_str=["dim {}".format(i) for i in range(num_action_dims_used)],
                tb_str=self.get_log_string(
                    "Mode Influence Test: Action/Skill {}".format(skill_id[idx])
                ),
                arrays_to_plot=action_seq_data_dim_first,
                step=epoch,
                y_lim=lim,
            )

    @torch.no_grad()
    def apply_df(
            self,
            *args,
            next_observations,
            **kwargs
    ) -> dict:
        batch_dim = 0
        seq_dim = 1
        data_dim = -1
        assert next_observations.shape[seq_dim] == self.seq_len
        num_seqs = next_observations.shape[batch_dim]

        next_observations = ptu.from_numpy(next_observations)
        skill_recon_dist = my_ptu.eval(self.df_to_evaluate, obs_seq=next_observations)
        ret_dict = dict(skill_recon_dist=skill_recon_dist)

        return ret_dict

    def plot_posterior(
            self,
            *args,
            epoch,
            skill_recon_dist,
            skill_id,
            skill,
            **kwargs
    ):
        assert skill_recon_dist.batch_shape \
               == torch.Size((self.seq_len, len(skill), skill[0].shape[-1]))
        assert isinstance(skill_recon_dist, torch_dist.Normal)
        colors = get_colors()

        # Without Limits
        plt.clf()
        _, axes = plt.subplots()
        for _skill_id, _skill, recon_dist_loc \
                in zip(skill_id, skill, skill_recon_dist.loc):
            plt.scatter(
                ptu.get_numpy(recon_dist_loc[:, 0].reshape(-1)),
                ptu.get_numpy(recon_dist_loc[:, 1].reshape(-1)),
                label="skill_{}({})".format(_skill_id, _skill),
                c=colors[_skill_id]
            )
        axes.grid(True)
        axes.legend()
        fig_without_lim = plt.gcf()

        # With Limits
        lim = [-3., 3.]
        axes.set_ylim(lim)
        axes.set_xlim(lim)
        fig_with_lim = plt.gcf()
        plt.close()

        # Write
        figs = dict(
            no_lim=fig_without_lim,
            lim=fig_with_lim,
        )
        for key, fig in figs.items():
            self.diagno_writer.writer.writer.add_figure(
                tag=self.get_log_string(
                    "Skill Posterior Plot Eval/{}".format(key)
                ),
                figure=fig,
                global_step=epoch,
            )

    def classifier_evaluation(
            self,
            *args,
            epoch,
            skill_recon_dist,
            skill,
            **kwargs
    ):
        skills_np = np.array([np.array([_skill] * self.seq_len) for _skill in skill])
        assert skills_np.shape == (len(skill), self.seq_len, skill[0].shape[-1])
        skills = ptu.from_numpy(skills_np)
        assert skill_recon_dist.batch_shape == skills.shape

        df_accuracy_eval = F.mse_loss(skill_recon_dist.loc, skills)

        self.diagno_writer.writer.writer.add_scalar(
            tag=self.get_log_string("Classifier Performance/Eval"),
            scalar_value=df_accuracy_eval,
            global_step=epoch,
        )

