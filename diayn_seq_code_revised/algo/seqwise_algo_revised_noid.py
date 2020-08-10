import torch
import numpy as np
import torch.nn.functional as F

from diayn_seq_code_revised.algo.seqwise_algo_revised import \
    SeqwiseAlgoRevisedDiscreteSkills
import rlkit.torch.pytorch_util as ptu

class SeqwiseAlgoRevisedDiscreteSkillsNoid(SeqwiseAlgoRevisedDiscreteSkills):

    @torch.no_grad()
    def _classfier_perf_on_memory(self):
        len_memory = self.batch_size

        batch_size = len_memory
        batch = self.replay_buffer.random_batch_bsd_format(
            batch_size=batch_size
        )

        skill_gt = ptu.from_numpy(batch.skill)
        pred_skill_dist = self.trainer.df(ptu.from_numpy(batch.next_obs))
        assert pred_skill_dist.batch_shape[:-1] == torch.Size(
            (batch_size, batch.next_obs.size(1))
        )
        assert pred_skill_dist.batch_shape[-1] == self.trainer.df.classifier.output_size
        assert pred_skill_dist.batch_shape == skill_gt.shape

        accuracy = F.mse_loss(skill_gt, pred_skill_dist.loc)

        return accuracy

    @torch.no_grad()
    def _classfier_perf_eval(self):
        num_paths = 2
        eval_paths = self._get_paths_mode_influence_test(
            num_paths=num_paths,
            seq_len=self.seq_len,
        )

        next_obs = []
        skill_gt = []
        for path in eval_paths:
            next_obs.append(path.next_obs)
            skill_gt.append(path.mode)

        next_obs = ptu.from_numpy(
            np.stack(next_obs, axis=0)
        ).transpose(-1, -2)
        skill_gt = ptu.from_numpy(
            np.stack(skill_gt, axis=0)
        ).transpose(-1, -2)

        pred_skill_dist = self.trainer.df(
            next_obs
        )
        assert pred_skill_dist.batch_shape == skill_gt.shape

        accuracy = F.mse_loss(skill_gt, pred_skill_dist.loc)

        return accuracy





