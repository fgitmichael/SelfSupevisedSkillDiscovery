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

import rlkit.torch.pytorch_util as ptu


class SeqwiseAlgoRevisedContSkills(SeqwiseAlgoRevised):

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

        obs_dim = eval_paths[0].obs.shape[0]

        next_obs = []
        mode = []
        for path in eval_paths:
            next_obs.append(path.next_obs)
            mode.append(path.mode)

        next_obs = ptu.from_numpy(
            np.stack(next_obs, axis=0)
        ).transpose(-1, -2)
        mode = ptu.from_numpy(
            np.stack(mode, axis=0)
        ).transpose(-1, -2)
        # assert next_obs.shape \
        #    == torch.Size((num_paths * self.policy.skill_dim, self.seq_len, obs_dim))
        assert next_obs.shape \
               == torch.Size((len(eval_paths), self.seq_len, obs_dim))

        pred_skill_dist = self.trainer.df(
            next_obs,
        )

        df_accuracy = F.mse_loss(pred_skill_dist.loc, mode)

        return df_accuracy
