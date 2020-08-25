import torch
from torch.nn import functional as F
import numpy as np

from diayn_seq_code_revised.algo.seqwise_algo_revised import \
    SeqwiseAlgoRevisedDiscreteSkills

import rlkit.torch.pytorch_util as ptu


class AlgoStepwiseOnlyDiscreteSkills(SeqwiseAlgoRevisedDiscreteSkills):

    def _log_perf_eval(self, epoch):
        classifier_accuracy_step = self._classfier_perf_eval()
        self.diagnostic_writer.writer.writer.add_scalar(
            tag="Rnn Debug/Classfier accuracy eval step",
            scalar_value=classifier_accuracy_step,
            global_step=epoch
        )

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
        assert next_obs.shape == torch.Size((len(eval_paths), self.seq_len, obs_dim))
        assert z_hat.shape == torch.Size((len(eval_paths), self.seq_len, 1))

        d_pred = self.trainer.df(next_obs)
        d_pred_logsoftmax = F.log_softmax(d_pred, dim=-1)
        pred_z = torch.argmax(d_pred_logsoftmax, dim=-1, keepdim=True)
        assert z_hat.shape == pred_z.shape
        pred_z = pred_z.view(-1, 1)
        z_hat = z_hat.view(-1, 1)
        df_accuracy = torch.sum(
            torch.eq(
                z_hat,
                pred_z
            )
        ).float()/pred_z.size(0)

        return df_accuracy
