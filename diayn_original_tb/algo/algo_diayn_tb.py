import torch
import torch.nn.functional as F
from typing import List


from rlkit.torch.sac.diayn.diayn_torch_online_rl_algorithm import \
    DIAYNTorchOnlineRLAlgorithm
import rlkit.torch.pytorch_util as ptu

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from self_sup_comb_discrete_skills.data_collector.path_collector_discrete_skills import \
    PathCollectorSelfSupervisedDiscreteSkills, TransitonModeMappingDiscreteSkills



class DIAYNTorchOnlineRLAlgorithmTb(DIAYNTorchOnlineRLAlgorithm):

    def __init__(self,
                 *args,
                 diagnostic_writer: DiagnosticsWriter,
                 sequence_eval_collector: PathCollectorSelfSupervisedDiscreteSkills,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.diagnostic_writer = diagnostic_writer
        self.seq_eval_collector = sequence_eval_collector

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)

        if self.diagnostic_writer.is_log(epoch):
            self.write_mode_influence(epoch)

    def write_mode_influence(self, epoch):
        paths = self._get_paths_mode_influence_test()

        obs_dim = paths[0].obs.shape[0]
        action_dim = paths[0].action.shape[0]
        for path in paths:

            skill_id = path.skill_id.squeeze()[0]

            # Observations
            self.diagnostic_writer.writer.plot_lines(
                legend_str=["dim {}".format(i) for i in range(obs_dim)],
                tb_str="Mode Influence Test: Obs/Skill {}".format(skill_id),
                arrays_to_plot=path.obs,
                step=epoch,
                y_lim=[-3, 3]
            )

            # Actions
            self.diagnostic_writer.writer.plot_lines(
                legend_str=["dim {}".format(i) for i in range(action_dim)],
                tb_str="Mode Influence Test: Action/Skill {}".format(skill_id),
                arrays_to_plot=path.action,
                step=epoch,
                y_lim=[-3, 3]
            )

            # TODO: write rewards

    def _get_paths_mode_influence_test(self, seq_len=100) \
            -> List[TransitonModeMappingDiscreteSkills]:

        for skill in range(self.policy.skill_dim):
            # Set skill
            skill_oh = F.one_hot(
                skill, num_classes=self.policy.skill_dim).float().to(ptu.device)
            self.seq_eval_collector.set_discrete_skill(
                skill_vec=skill_oh,
                skill_id=skill
            )

            self.seq_eval_collector.collect_new_paths(
                seq_len=seq_len,
                num_seqs=1,
                discard_incomplete_paths=False
            )

        mode_influence_eval_paths = self.eval_data_collector.get_epoch_paths()

        return mode_influence_eval_paths








