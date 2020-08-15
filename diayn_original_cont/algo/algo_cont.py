import torch
from torch.nn import functional as F
import numpy as np
from typing import List

import rlkit.torch.pytorch_util as ptu

from diayn_original_tb.algo.algo_diayn_tb_perf_logging import \
    DIAYNTorchOnlineRLAlgorithmTbPerfLoggingEffiently

from diayn_original_cont.data_collector.seq_collector_optionally_id import \
    SeqCollectorRevisedOptionalId

from diayn_original_cont.policy.policies import MakeDeterministicCont
from diayn_original_cont.utils.mode_post_plotter import ModepostPlotter

import self_supervised.utils.typed_dicts as td


class DIAYNContAlgo(DIAYNTorchOnlineRLAlgorithmTbPerfLoggingEffiently):

    def _get_paths_mode_influence_test(self,
                                       num_paths=1,
                                       seq_len=200
    ) -> List[td.TransitonModeMappingDiscreteSkills]:
        for _ in range(num_paths):
            for id, skill in enumerate(self.policy.eval_grid):
                assert isinstance(self.seq_eval_collector, SeqCollectorRevisedOptionalId)
                self.seq_eval_collector.set_skill(skill)
                self.seq_eval_collector.collect_new_paths(
                    seq_len=seq_len,
                    num_seqs=1,
                    id_to_add=id,
                )

        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()
        return mode_influence_eval_paths

    @torch.no_grad()
    def _write_mode_influence_and_log(self, paths, epoch):
        """
        Main logging function

        Args:
            eval_paths              : (data_dim, seq_dim) evaluation paths sampled directly
                                      from the environment
            epoch                   : int
        """
        super()._write_mode_influence_and_log(paths, epoch)
        self.write_mode_map(
            paths=paths,
            epoch=epoch
        )

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

        skill_post = self.trainer.df(
            ptu.from_numpy(next_obs)
        )
        fig = ModepostPlotter().plot_mode_post(
            skills=skills,
            mu_post=ptu.get_numpy(skill_post['dist'].loc),
            ids=skill_ids,
        )

        self.diagnostic_writer.writer.writer.add_figure(
            tag="Mode Post Plot",
            figure=fig,
            global_step=epoch
        )

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

        skill_post = self.trainer.df(
            next_obs,
        )
        assert next_obs.shape \
               == torch.Size((self.policy.eval_grid.size(0) * seq_len, obs_dim))

        assert skill_post['dist'].batch_shape == skills.shape
        df_accuracy = F.mse_loss(skill_post['dist'].loc, skills)

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

        skill_post = self.trainer.df(
            next_obs
        )

        df_accuracy = F.mse_loss(skills, skill_post['dist'].loc)

        return df_accuracy
