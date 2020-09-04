import torch
from torch.nn import functional as F
import numpy as np

from diayn_original_cont.algo.algo_cont import DIAYNContAlgo
from diayn_original_cont.utils.mode_post_plotter_highdimusingvae import \
    ModepostPlotterHighdimusingvae

import rlkit.torch.pytorch_util as ptu

from diayn_original_cont.policy.policies import MakeDeterministicCont

import self_supervised.utils.typed_dicts as td



class DIAYNContAlgoHighdimusingvae(DIAYNContAlgo):

    def _classifier_perf_eval(self, eval_paths):
        obs_dim = eval_paths[0].obs.shape[0]
        seq_len = eval_paths[0].obs.shape[-1]

        next_obs = []
        skills = []
        for path in eval_paths:
            next_obs.append(
                path.next_obs.transpose((1, 0))
            ) # data_dim x seq_len
            skills.append(
                path.mode.transpose((1, 0))
            ) # data_dim x seq_len

        next_obs = ptu.from_numpy(
            np.concatenate(next_obs, axis=0)
        )
        skills = ptu.from_numpy(
            np.concatenate(skills, axis=0)
        )
        assert isinstance(self.seq_eval_collector._rollouter.policy,
                          MakeDeterministicCont)
        assert next_obs.shape[0] % (self.policy.eval_grid.size(0) * seq_len) \
               == skills.shape[0] % (self.policy.eval_grid.size(0) * seq_len) \
               == 0
        assert next_obs.shape[-1] == obs_dim

        ret_dict = self.trainer.df(
            next_obs,
            train=True,
        )
        assert next_obs.shape \
               == torch.Size((self.policy.eval_grid.size(0) * seq_len, obs_dim))

        assert ret_dict['recon']['dist'].batch_shape == skills.shape
        df_accuracy = F.mse_loss(ret_dict['recon']['dist'].loc, skills)

        return df_accuracy

    def _classfier_perf_on_memory(self):
        len_memory = self.batch_size

        batch_size = len_memory
        batch = self.replay_buffer.random_batch(
            batch_size=batch_size)
        skills = batch['skills']
        skills = ptu.from_numpy(skills)
        next_obs = batch['next_observations']
        next_obs = ptu.from_numpy(next_obs)

        ret_dict = self.trainer.df(
            next_obs,
            train=True,
        )
        skill_post = ret_dict['recon']

        df_accuracy = F.mse_loss(skills, skill_post['dist'].loc)

        return df_accuracy

    def write_mode_map(self, paths, epoch):
        assert isinstance(paths[0], td.TransitonModeMappingDiscreteSkills)
        next_obs = []
        skills = []
        skill_ids = []
        for path in paths:
            next_obs.append(
                path.next_obs.transpose((1, 0))
            ) # data_dim x seq_len
            skills.append(
                path.mode.transpose((1, 0))
            ) # data_dim x seq_len
            skill_ids.append(
                path.skill_id.transpose((1, 0))
            )

        next_obs = np.concatenate(next_obs, axis=0)
        skills = np.concatenate(skills, axis=0)
        skill_ids = np.concatenate(skill_ids, axis=0)

        ret_dict = self.trainer.df(
            ptu.from_numpy(next_obs),
            train=True,
        )
        latent_post = ret_dict['post']

        fig = ModepostPlotterHighdimusingvae().plot_mode_post(
            mu_post=ptu.get_numpy(latent_post['dist'].loc),
            ids=skill_ids,
        )

        self.diagnostic_writer.writer.writer.add_figure(
            tag="Mode Post Plot",
            figure=fig,
            global_step=epoch
        )

