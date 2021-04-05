import torch
import torch.nn.functional as F

from diayn_cont.post_epoch_funcs.df_memory_eval import DfMemoryEvalDIAYNCont

import rlkit.torch.pytorch_util as ptu

import self_supervised.utils.my_pytorch_util as my_ptu


class DfMemoryEvalDIAYN(DfMemoryEvalDIAYNCont):

    @torch.no_grad()
    def apply_df(
            self,
            *args,
            sampled_transitions,
            **kwargs
    ) -> dict:
        next_obs = ptu.from_numpy(sampled_transitions['next_observations'])
        skill_recon= my_ptu.eval(self.df_to_evaluate, next_obs)
        return dict(
            skill_recon=skill_recon
        )

    def classifier_evaluation(
            self,
            *args,
            epoch,
            skill_recon,
            sampled_transitions,
            **kwargs
    ):
        df_accuracy_memory = F.mse_loss(
            skill_recon.loc,
            ptu.from_numpy(sampled_transitions['skills'])
        )

        self.diagno_writer.writer.writer.add_scalar(
            tag=self.get_log_string("Classifier Performance/Memory"),
            scalar_value=df_accuracy_memory,
            global_step=epoch
        )
