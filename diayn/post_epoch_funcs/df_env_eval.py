import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from diayn_cont.post_epoch_funcs.df_env_eval import DfEnvEvaluationDIAYNCont

import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector.path_collector import MdpPathCollector

import self_supervised.utils.my_pytorch_util as my_ptu

from seqwise_cont_skillspace.utils.get_colors import get_colors


class DfEnvEvaluationDIAYN(DfEnvEvaluationDIAYNCont):

    def __init__(self,
                 *args,
                 skill_dim,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.skill_dim = skill_dim

    def collect_skill_influence_paths(self) -> dict:
        assert isinstance(self.seq_collector, MdpPathCollector)

        skill_array = np.eye(self.skill_dim)
        skill_ids = []
        skills = []
        for skill_id, skill in enumerate(skill_array):
            self.seq_collector._policy.skill = skill
            self.seq_collector.collect_new_paths(
                max_path_length=self.seq_len,
                num_steps=self.seq_len,
                discard_incomplete_paths=False
            )
            skills.append(skill)
            skill_ids.append(skill_id)
        skill_influence_paths = self.seq_collector.get_epoch_paths()
        skill_influence_paths = list(skill_influence_paths)
        assert isinstance(skill_influence_paths, list)

        for skill_id, skill, path in zip(skill_ids, skills, skill_influence_paths):
            path['skill_id'] = skill_id
            path['skill'] = skill
        self._check_skill_influence_paths(skill_influence_paths)

        skill_influence_paths = self._stack_paths(skill_influence_paths)
        return skill_influence_paths

    @torch.no_grad()
    def apply_df(
            self,
            *args,
            next_observations,
            **kwargs
    ) -> dict:
        next_observations = ptu.from_numpy(next_observations)
        skill_recon = my_ptu.eval(self.df_to_evaluate, next_observations)
        ret_dict = dict(skill_recon=skill_recon)

        return ret_dict

    def plot_posterior(
            self,
            *args,
            epoch,
            skill_recon,
            skill_id,
            skill,
            **kwargs
    ):
        assert skill_recon.shape \
               == torch.Size((len(skill), self.seq_len, skill[0].shape[-1]))
        assert isinstance(skill_recon, torch.Tensor)
        colors = get_colors()

        # Without Limits
        plt.clf()
        _, axes = plt.subplots()
        for _skill_id, _skill, recon\
                in zip(skill_id, skill, skill_recon):
            plt.scatter(
                ptu.get_numpy(recon[:, 0].reshape(-1)),
                ptu.get_numpy(recon[:, 1].reshape(-1)),
                label="skill {}({})".format(
                    _skill_id,
                    np.array2string(ptu.get_numpy(_skill))
                ),
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
            skill_recon,
            skill,
            **kwargs
    ):
        #skills_np = np.array([np.array([_skill] * self.seq_len) for _skill in skill])
        assert isinstance(skill, list)
        assert isinstance(skill[0], torch.Tensor)
        skill = torch.stack(skill)
        assert skill_recon.shape == torch.Size((skill.shape[0], self.seq_len, 2))
        skills = torch.stack([skill] * self.seq_len, dim=1)
        assert skill_recon.shape == skills.shape

        df_accuracy_eval = F.mse_loss(skill_recon, skills)

        self.diagno_writer.writer.writer.add_scalar(
            tag=self.get_log_string("Classifier Performance/Eval"),
            scalar_value=df_accuracy_eval,
            global_step=epoch,
        )
