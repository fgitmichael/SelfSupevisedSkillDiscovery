import os
import cv2
import matplotlib.pyplot as plt

from thes_graphics.base.grid_rollout_processor import GridRolloutProcessor


class RelevantTrajectoryVideoSaver(GridRolloutProcessor):

    def __init__(self,
                 *args,
                 extract_relevant_rollouts_fun,
                 num_relevant_skills: int,
                 path,
                 save_name_prefix,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.extract_relevant_rollouts_fun = extract_relevant_rollouts_fun
        self.num_relevant_skills = num_relevant_skills

        self.path_name_grid_rollouts = path
        self.save_name_prefix = save_name_prefix

        if not os.path.exists(self.path_name_grid_rollouts):
            os.makedirs(self.path_name_grid_rollouts)
            
    def __call__(self, 
                 *args, 
                 **kwargs):
        grid_rollout = super().__call__(*args, **kwargs)

        # Clear all figures
        plt.close()

        # Extract relevant rollouts
        grid_rollout_relevant = self.extract_relevant_rollouts_fun(
            grid_rollout,
            num_to_extract=self.num_relevant_skills,
        )

        # Save Videos
        frame_dims = [-2, -1]
        frame_size = grid_rollout_relevant[0]['frames'].shape[frame_dims]
        for idx, rollout in enumerate(grid_rollout_relevant):
            # Destination
            save_name = os.path.join(
                self.path_name_grid_rollouts,
                self.save_name_prefix + "_skill_{}".format(idx) + '.avi'
            )
            out = cv2.VideoWriter(
                save_name,
                cv2.VideoWriter_fourcc(*'DIVX'), 60, frame_size
            )

            # Write images to video
            frames = rollout['frames']
            for frame in frames:
                out.write(frame)
            out.release()
