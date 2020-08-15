import torch
from torch.nn import functional as F
import numpy as np
import gtimer as gt

import rlkit.torch.pytorch_util as ptu

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

class DIAYNTorchOnlineRLAlgorithmTbPerfLoggingEffiently(DIAYNTorchOnlineRLAlgorithmTb):
    # Saves one evaluation sampling loop from env compared to
    # DIAYNTorchOnlineRLAlgorithmTbPerfLogging (can be found below for comparison)
    def _classifier_perf_eval(self, eval_paths):
        obs_dim = eval_paths[0].obs.shape[0]
        seq_len = eval_paths[0].obs.shape[-1]

        next_obs = []
        z_hat = []
        for path in eval_paths:
            next_obs.append(path.next_obs.transpose((1, 0))) # data_dim x seq_len
            z_hat.append(path.mode.transpose((1, 0))) # data_dim x seq_len

        next_obs = ptu.from_numpy(
            np.concatenate(next_obs, axis=0)
        )
        z_hat = ptu.from_numpy(
            np.concatenate(z_hat, axis=0)
        )
        z_hat = torch.argmax(z_hat, dim=-1)
        assert next_obs.shape[0] % (self.policy.skill_dim * seq_len) == 0
        assert next_obs.shape[-1] == obs_dim

        d_pred = self.trainer.df(
            next_obs,
        )
        d_pred_log_softmax = F.log_softmax(d_pred, dim=-1)

        pred_z = torch.argmax(d_pred_log_softmax, dim=-1, keepdim=True)
        assert z_hat.shape == pred_z.squeeze().shape
        df_accuracy = torch.sum(
            torch.eq(
                z_hat,
                pred_z.squeeze()
            )).float() / pred_z.size(0)

        return df_accuracy

    def log_classifier_perf_eval(self, eval_paths, epoch):
        classfier_accuracy_eval = self._classifier_perf_eval(eval_paths=eval_paths)
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Debug/Classfier accuracy eval",
            scalar_value=classfier_accuracy_eval,
            global_step=epoch
        )

    def _classfier_perf_on_memory(self):
        len_memory = self.batch_size

        batch_size = len_memory
        batch = self.replay_buffer.random_batch(
            batch_size=batch_size)
        skills = batch['skills']
        next_obs = batch['next_observations']

        z_hat = ptu.from_numpy(np.argmax(skills, axis=-1))
        d_pred = self.trainer.df(
            ptu.from_numpy(next_obs))
        pred_log_softmax = F.log_softmax(d_pred, dim=-1)
        pred_z = torch.argmax(pred_log_softmax, dim=-1, keepdim=True)
        assert z_hat.shape == pred_z.squeeze().shape

        df_accuracy = torch.sum(
            torch.eq(
                z_hat,
                pred_z.squeeze(),
            )).float()/pred_z.size(0)

        return df_accuracy

    def log_classifier_perf_on_memory(self, epoch):
        classfier_accuracy_memory = self._classfier_perf_on_memory()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Debug/Classfier accuracy replay buffer",
            scalar_value=classfier_accuracy_memory,
            global_step=epoch
        )

    def _write_mode_influence_and_log(self, paths, epoch):
        """
        Main logging function

        Args:
            eval_paths              : (data_dim, seq_dim) evaluation paths sampled directly
                                      from the environment
            epoch                   : int
        """
        super()._write_mode_influence_and_log(
            paths=paths,
            epoch=epoch,
        )

        self.log_classifier_perf_eval(
            eval_paths=paths,
            epoch=epoch,
        )

        self.log_classifier_perf_on_memory(
            epoch=epoch,
        )

class DIAYNTorchOnlineRLAlgorithmTbPerfLogging(DIAYNTorchOnlineRLAlgorithmTb):

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)

        classfier_accuracy_memory = self._classfier_perf_on_memory()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Debug/Classfier accuracy replay buffer",
            scalar_value=classfier_accuracy_memory,
            global_step=epoch
        )

        classfier_accuracy_eval = self._classfier_perf_eval()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Debug/Classfier accuracy eval",
            scalar_value=classfier_accuracy_eval,
            global_step=epoch
        )

        gt.stamp('own logging')

    def _classfier_perf_eval(self):
        num_paths = 2
        seq_len = 100
        eval_paths = self._get_paths_mode_influence_test(
            num_paths=num_paths,
            seq_len=seq_len,
        )

        obs_dim = eval_paths[0].obs.shape[0]

        next_obs = []
        z_hat = []
        for path in eval_paths:
            next_obs.append(path.next_obs.transpose((1, 0))) # data_dim x seq_len
            z_hat.append(path.mode.transpose((1, 0))) # data_dim x seq_len

        next_obs = ptu.from_numpy(
            np.concatenate(next_obs, axis=0)
        )
        z_hat = ptu.from_numpy(
            np.concatenate(z_hat, axis=0)
        )
        z_hat = torch.argmax(z_hat, dim=-1)
        assert next_obs.shape \
               == torch.Size((num_paths * self.policy.skill_dim * seq_len, obs_dim))

        d_pred = self.trainer.df(
            next_obs,
        )
        d_pred_log_softmax = F.log_softmax(d_pred, dim=-1)

        pred_z = torch.argmax(d_pred_log_softmax, dim=-1, keepdim=True)
        assert z_hat.shape == pred_z.squeeze().shape
        df_accuracy = torch.sum(
            torch.eq(
                z_hat,
                pred_z.squeeze()
            )).float() / pred_z.size(0)

        return df_accuracy

    def _classfier_perf_on_memory(self):
        len_memory = self.batch_size

        batch_size = len_memory
        batch = self.replay_buffer.random_batch(
            batch_size=batch_size)
        skills = batch['skills']
        next_obs = batch['next_observations']

        z_hat = ptu.from_numpy(np.argmax(skills, axis=-1))
        d_pred = self.trainer.df(
            ptu.from_numpy(next_obs))
        pred_log_softmax = F.log_softmax(d_pred, dim=-1)
        pred_z = torch.argmax(pred_log_softmax, dim=-1, keepdim=True)
        assert z_hat.shape == pred_z.squeeze().shape

        df_accuracy = torch.sum(
            torch.eq(
                z_hat,
                pred_z.squeeze(),
            )).float()/pred_z.size(0)

        return df_accuracy




