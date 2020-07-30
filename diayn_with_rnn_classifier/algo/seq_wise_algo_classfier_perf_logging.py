import torch
from torch.nn import functional as F
import numpy as np


from diayn_original_tb.algo.algo_diayn_tb_own_fun import \
    DIAYNTorchOnlineRLAlgorithmOwnFun

import rlkit.torch.pytorch_util as ptu


class SeqWiseAlgoClassfierPerfLogging(DIAYNTorchOnlineRLAlgorithmOwnFun):

    @torch.no_grad()
    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)

        classfier_accuracy_memory = self._classfier_perf_on_memory()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Rnn Debug/Classfier accuracy replay buffer",
            scalar_value=classfier_accuracy_memory,
            global_step=epoch
        )

        classfier_accuracy_eval = self._classfier_perf_eval()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Rnn Debug/Classfier accuracy eval",
            scalar_value=classfier_accuracy_eval,
            global_step=epoch
        )

    @torch.no_grad()
    def _classfier_perf_eval(self):

        num_paths = 10
        eval_paths = self._get_paths_mode_influence_test(
            num_paths=num_paths,
            seq_len=self.seq_len,
        )
        
        obs_dim = eval_paths[0].obs.shape[0]
        action_dim = eval_paths[0].action.shape[0]

        next_obs = []
        mode = []
        for path in eval_paths:
            next_obs.append(path.next_obs)
            mode.append(path.mode)
        next_obs = np.stack(next_obs, dim=0)
        mode = np.stack(mode, dim=0)
        assert next_obs.shape == (num_paths * self.policy.skill_dim, self.seq_len, obs_dim)

        pred = self.trainer.seq_classifier_mod.encoder(
            state_rep_seq=next_obs,
        )
        pred_log_softmax = F.log_softmax(pred, dim=1)

        labels = torch.argmax(ptu.from_numpy(mode), dim=-1, keepdim=True)[:, 0, :]
        pred_z = torch.argmax(pred_log_softmax, dim=-1, keepdim=True)
        assert labels.shape == pred_z.shape
        df_accuracy = torch.sum(
            torch.eq(
                labels,
                pred_z
            )).float()/pred_z.size(0)

        return df_accuracy

    @torch.no_grad()
    def _classfier_perf_on_memory(self):
        len_memory = len(self.replay_buffer)

        batch_size = len_memory
        batch = self.replay_buffer.random_batch_bsd_format(
            batch_size=batch_size)

        pred = self.trainer.seq_classifier_mod.encoder(
            state_rep_seq=batch.next_obs)
        pred_log_softmax = F.log_softmax(pred, dim=1)

        labels = torch.argmax(batch.mode, dim=-1, keepdim=True)[:, 0, :]
        pred_z = torch.argmax(pred_log_softmax, dim=-1, keepdim=True)
        assert labels.shape == pred_z.shape
        df_accuracy = torch.sum(
            torch.eq(
                labels,
                pred_z
            )).float()/pred_z.size(0)

        return df_accuracy





