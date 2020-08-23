import torch
from torch.nn import functional as F
import numpy as np
import gtimer as gt

from self_supervised.base.replay_buffer.env_replay_buffer import NormalSequenceReplayBuffer

from diayn_original_tb.algo.algo_diayn_tb_own_fun import \
    DIAYNTorchOnlineRLAlgorithmOwnFun

import rlkit.torch.pytorch_util as ptu


class SeqWiseAlgoClassfierPerfLogging(DIAYNTorchOnlineRLAlgorithmOwnFun):

    @torch.no_grad()
    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)

        self._log_perf_memory(epoch)
        self._log_perf_eval(epoch)
        gt.stamp("own classfier perf logging")

        self._log_net_param_hist(epoch)
        gt.stamp("net parameter histogram logging")

    def _log_perf_eval(self, epoch):
        classfier_accuracy_eval = self._classfier_perf_eval()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Rnn Debug/Classfier accuracy eval",
            scalar_value=classfier_accuracy_eval,
            global_step=epoch
        )

    def _log_perf_memory(self, epoch):
        classfier_accuracy_memory = self._classfier_perf_on_memory()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Rnn Debug/Classfier accuracy replay buffer",
            scalar_value=classfier_accuracy_memory,
            global_step=epoch
        )

    @torch.no_grad()
    def _classfier_perf_eval(self):

        num_paths = 2
        eval_paths = self._get_paths_mode_influence_test(
            num_paths=num_paths,
            seq_len=self.seq_len,
        )
        
        obs_dim = eval_paths[0].obs.shape[0]

        next_obs = []
        z_hat = []
        for path in eval_paths:
            next_obs.append(path.next_obs)
            z_hat.append(path.skill_id)

        next_obs = ptu.from_numpy(
            np.stack(next_obs, axis=0)
        ).transpose(-1, -2)
        z_hat = ptu.from_numpy(
            np.stack(z_hat, axis=0)
        ).transpose(-1, -2)
        #assert next_obs.shape \
        #    == torch.Size((num_paths * self.policy.skill_dim, self.seq_len, obs_dim))
        assert next_obs.shape \
            == torch.Size((len(eval_paths), self.seq_len, obs_dim))
        assert z_hat.shape \
            == torch.Size((len(eval_paths), self.seq_len, 1))

        d_pred = self.trainer.df(
            next_obs,
        )
        d_pred_log_softmax = F.log_softmax(d_pred, dim=-1)

        pred_z = torch.argmax(d_pred_log_softmax, dim=-1, keepdim=True)
        assert z_hat.shape == pred_z.shape
        pred_z = pred_z.view(-1, 1)
        z_hat = z_hat.view(-1, 1)
        df_accuracy = torch.sum(
            torch.eq(
                z_hat,
                pred_z
            )).float()/pred_z.size(0)

        return df_accuracy

    @torch.no_grad()
    def _classfier_perf_on_memory(self):
        len_memory = self.batch_size

        batch_size = len_memory
        assert isinstance(self.replay_buffer, NormalSequenceReplayBuffer)
        batch = self.replay_buffer.random_batch_bsd_format(
            batch_size=batch_size)

        z_hat = ptu.from_numpy(batch.skill_id)
        d_pred = self.trainer.df(
            ptu.from_numpy(batch.next_obs))
        pred_log_softmax = F.log_softmax(d_pred, dim=-1)
        pred_z = torch.argmax(pred_log_softmax, dim=-1, keepdim=True)
        assert z_hat.shape == pred_z.shape

        pred_z = pred_z.view(-1, 1)
        z_hat = z_hat.view(-1, 1).int()
        df_accuracy = torch.sum(
            torch.eq(
                z_hat,
                pred_z,
            )).float()/pred_z.size(0)

        return df_accuracy

    def _log_net_param_hist(self, epoch):
        for k, net in self.trainer.network_dict.items():
            for name, weight in net.named_parameters():
                self.diagnostic_writer.writer.writer.\
                    add_histogram(k + name, weight, epoch)
                if weight.grad is not None:
                    self.diagnostic_writer.writer.writer.\
                        add_histogram(f'{k + name}.grad', weight.grad, epoch)

