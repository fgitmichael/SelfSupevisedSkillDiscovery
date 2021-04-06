import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from diayn_cont.post_epoch_funcs.df_env_eval import DfEnvEvaluationDIAYNCont

import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector.path_collector import MdpPathCollector

import self_supervised.utils.my_pytorch_util as my_ptu

from seqwise_cont_skillspace.utils.get_colors import get_colors

import self_supervised.utils.my_pytorch_util as my_ptu


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

        skill_array = [number for number in range(self.seq_collector._policy.skill_dim)]
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
        pass

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
        assert skill_recon.shape[:-1] == torch.Size((len(skill), self.seq_len))
        skill_recon_reshaped = skill_recon.reshape(len(skill) * self.seq_len, -1)
        assert my_ptu.tensor_equality(skill_recon_reshaped[:self.seq_len], skill_recon[0,])
        skills = torch.stack([torch.tensor(skill)] * self.seq_len, dim=-1).reshape(len(skill) * self.seq_len)

        df_accuracy_eval = F.cross_entropy(skill_recon_reshaped.cpu(), skills.cpu())

        self.diagno_writer.writer.writer.add_scalar(
            tag=self.get_log_string("Classifier Performance/Eval"),
            scalar_value=df_accuracy_eval,
            global_step=epoch,
        )
