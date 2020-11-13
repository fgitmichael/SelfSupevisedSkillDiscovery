from typing import List
import gtimer as gt
import numpy as np
from torch.nn import functional as F

from rlkit.torch import pytorch_util as ptu

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

from self_supervised.utils import typed_dicts as td


class DIAYNTorchOnlineRLAlgorithmTbEval(DIAYNTorchOnlineRLAlgorithmTb):

    def __init__(self, *args, **kwargs):
        super(DIAYNTorchOnlineRLAlgorithmTbEval, self).__init__(*args, **kwargs)

        self.post_epoch_funcs.append(self.save_objects)

    def save_objects(self, *args, epoch):
        self.diagnostic_writer.save_object_islog(
            obj=self.policy,
            save_name="policy_net",
            epoch=epoch,
            log_interval=20,
        )
        if epoch == 0:
            self.diagnostic_writer.save_env(self.expl_env)

    def _end_epoch(self, epoch):
        super()._end_epoch(epoch)
        if self.diagnostic_writer.is_log(epoch):
            self.write_mode_influence_and_log(epoch)

    def write_mode_influence_and_log(self, epoch):
        if self.diagnostic_writer.is_log(epoch):
            paths = self._get_paths_mode_influence_test()
            self._write_mode_influence_and_log(
                paths=paths,
                epoch=epoch,
            )
        # self.write_skill_hist(epoch)
        gt.stamp('mode influence logging')

    def _write_mode_influence_and_log(self, paths, epoch):
        """
        Main logging function

        Args:
            eval_paths              : (data_dim, seq_dim) evaluation paths
                                      sampled directly
                                      from the environment
            epoch                   : int
        """
        obs_dim = paths[0].obs.shape[0]
        action_dim = paths[0].action.shape[0]

        # Plot influence in different plot
        for path in paths:
            self._write_mode_influence(
                path,
                obs_dim=obs_dim,
                action_dim=action_dim,
                epoch=epoch,
                obs_lim=self.mode_influence_path_obs_lim,
            )

        # Plot influence in one plot
        self._write_mode_influence_one_plot(
            paths=paths,
            epoch=epoch,
        )

    def _write_mode_influence_one_plot(self, paths, epoch):
        obs_dim = paths[0].obs.shape[0]
        action_dim = paths[0].action.shape[0]

        # Plot influence in one plot
        if obs_dim == 2:
            obs = np.stack([path.obs for path in paths], axis=2)
            self.diagnostic_writer.writer.plot(
                obs[0], obs[1],
                tb_str="ModeInfluence All Skills in One Plot/With Limits",
                step=epoch,
                labels=["skill {}".format(path.skill_id.squeeze()[0]) for path in paths],
                x_lim=[-2, 2],
                y_lim=[-2, 2]
            )

            obs = np.stack([path.obs for path in paths], axis=2)
            self.diagnostic_writer.writer.plot(
                obs[0], obs[1],
                tb_str="ModeInfluence All Skills in One Plot/Without Limits",
                step=epoch,
                labels=["skill {}".format(path.skill_id.squeeze()[0]) for path in paths],
            )

            if self.mode_influence_one_plot_scatter:
                self.diagnostic_writer.writer.scatter(
                    obs[0], obs[1],
                    tb_str="ModeInfluence All Skills in One Plot Scatter/With Limits",
                    step=epoch,
                    labels=["skill {}".format(
                        path.skill_id.squeeze()[0]) for path in paths],
                    x_lim=[-2, 2],
                    y_lim=[-2, 2]
                )
                obs = np.stack([path.obs for path in paths], axis=2)
                self.diagnostic_writer.writer.scatter(
                    obs[0], obs[1],
                    tb_str="ModeInfluence All Skills in One Plot Scatter/Without Limits",
                    step=epoch,
                    labels=["skill {}".format(
                        path.skill_id.squeeze()[0]) for path in paths],
                )

    def _write_mode_influence(self,
                              path,
                              obs_dim,
                              action_dim,
                              epoch,
                              obs_lim=None,
                              ):
        skill_id = path.skill_id.squeeze()[0]

        # Observations
        self.diagnostic_writer.writer.plot_lines(
            legend_str=["dim {}".format(i) for i in range(obs_dim)],
            tb_str="Mode Influence Test: Obs/Skill {}".format(skill_id),
            arrays_to_plot=path.obs,
            step=epoch,
            y_lim=[-3, 3] if obs_lim is None else obs_lim,
        )

        if obs_dim == 2:
            # State Space
            self.diagnostic_writer.writer.plot(
                *[obs_dim_array for obs_dim_array in path.obs],
                tb_str="State Space Behaviour/Skill {}".format(skill_id),
                step=epoch,
                x_lim=[-2.2, 2.2],
                y_lim=[-2.2, 2.2]
            )

        # Actions
        self.diagnostic_writer.writer.plot_lines(
            legend_str=["dim {}".format(i) for i in range(action_dim)],
            tb_str="Mode Influence Test: Action/Skill {}".format(skill_id),
            arrays_to_plot=path.action,
            step=epoch,
            y_lim=[-1.2, 1.2]
        )
        # TODO: write rewards

    def _get_paths_mode_influence_test(self, num_paths=1, seq_len=200) \
            -> List[td.TransitonModeMappingDiscreteSkills]:

        for _ in range(num_paths):
            for skill in range(self.policy.skill_dim):
                # Set skill
                skill_oh = F.one_hot(
                    ptu.tensor(skill), num_classes=self.policy.skill_dim)
                self.seq_eval_collector.set_skill(skill)

                self.seq_eval_collector.collect_new_paths(
                    seq_len=seq_len,
                    num_seqs=1,
                )

        mode_influence_eval_paths = self.seq_eval_collector.get_epoch_paths()

        return mode_influence_eval_paths

    def write_skill_hist(self, epoch):
        buffer_size = self.replay_buffer._size
        buffer_skills = np.argmax(self.replay_buffer._skill[:buffer_size], axis=-1)

        self.diagnostic_writer.writer.writer.add_histogram(
            tag="Buffer Skill Distribution",
            values=buffer_skills,
            global_step=epoch,
            bins=self.policy.skill_dim-1,
        )