import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from thes_graphics.base.grid_rollout_processor import GridRolloutProcessor

from my_utils.rollout.grid_rollouter import GridRollouterBase
from my_utils.rollout.point_rollouter import PointRollouter
from my_utils.rollout.frame_plus_obs_rollout import rollout as rollout_frame_plus_obs


class RelevantTrajectoryVideoSaver(GridRolloutProcessor):

    def __init__(self,
                 test_rollouter: GridRollouterBase,
                 extract_relevant_rollouts_fun,
                 num_relevant_skills: int,
                 path,
                 save_name_prefix,
                 ):
        assert isinstance(test_rollouter, GridRollouterBase)
        super().__init__(
            test_rollouter=test_rollouter,
        )

        self.extract_relevant_rollouts_fun = extract_relevant_rollouts_fun
        self.num_relevant_skills = num_relevant_skills

        self.path_name_grid_rollouts = path
        self.save_name_prefix = save_name_prefix

        if not os.path.exists(self.path_name_grid_rollouts):
            os.makedirs(self.path_name_grid_rollouts)

    def plot_relevant_rollouts(self, grid_rollout_relevant):
        for idx, rollout in enumerate(grid_rollout_relevant):
            obs_plot_dims = rollout['observations'][..., :2]
            plt.plot(
                obs_plot_dims[..., 0],
                obs_plot_dims[..., 1],
                label='skill {} {}'.format(idx, rollout['skill']),
            )
        save_name = os.path.join(self.path_name_grid_rollouts, 'skills_plotted.png')
        plt.legend()
        plt.savefig(save_name)
        plt.close()

    def plot_x_dist_heatmap(self, grid_rollout):
        assert np.sqrt(len(grid_rollout)).is_integer()
        covered_dists = [rollout['observations'][-1, 0]
                         for rollout in grid_rollout]
        covered_dists_mat = np.reshape(
            np.expand_dims(np.array(covered_dists), 0),
            self.test_rollouter.skill_grid.shape[:-1]
        )

        plt.imshow(covered_dists_mat)
        plt.colorbar(orientation='vertical')

        save_name = os.path.join(self.path_name_grid_rollouts, 'covered_dist_heatmap.png')
        plt.savefig(save_name)

        plt.close()

    def __call__(self, 
                 *args,
                 render=False,
                 **kwargs):
        grid_rollout = super().__call__(
            *args,
            render=render,
            **kwargs
        )

        # Clear all figures
        plt.close()

        # Extract relevant rollouts
        grid_rollout_relevant = self.extract_relevant_rollouts_fun(
            grid_rollout,
            num_to_extract=self.num_relevant_skills,
        )
        assert len(grid_rollout_relevant[0]['frames']) == 0

        # Plots
        self.plot_x_dist_heatmap(grid_rollout)
        self.plot_relevant_rollouts(grid_rollout_relevant)

        if render:
            grid_rollout_relevant_with_frames = grid_rollout_relevant

        else:
            skills = [el['skill'] for el in grid_rollout_relevant]
            relevant_skill_rollouter = PointRollouter(
                env=self.test_rollouter.env,
                policy=self.test_rollouter.policy,
                rollout_fun=rollout_frame_plus_obs,
                horizon_len=self.test_rollouter.horizon_len,
                skill_points=skills,
            )
            grid_rollout_relevant_with_frames = relevant_skill_rollouter(
                render=True,
                render_kwargs=dict(
                    mode='rgb_array'
                )
            )

        # Save Videos
        _, h, w, _ = grid_rollout_relevant_with_frames[0]['frames'].shape
        for idx, rollout in enumerate(grid_rollout_relevant_with_frames):
            # Destination
            save_name = os.path.join(
                self.path_name_grid_rollouts,
                self.save_name_prefix + "_skill_{}".format(idx) + '.avi'
            )
            out = cv2.VideoWriter(
                save_name,
                cv2.VideoWriter_fourcc(*'DIVX'),
                60,
                (w, h),
            )

            # Write images to video
            frames = rollout['frames']
            for frame in frames:
                out.write(frame)
            out.release()
