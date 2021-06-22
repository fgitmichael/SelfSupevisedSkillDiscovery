import tqdm

from my_utils.rollout.frame_plus_obs_rollout import rollout as rollout_function
from my_utils.rollout.grid_rollouter import GridRollouter

import rlkit.torch.pytorch_util as ptu


class VideoGridRollouter(GridRollouter):

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.render_kwargs = dict(
            mode='rgb_array'
        )

    def rollout_trajectories(self) -> list:
        rollouts = []

        for skill in tqdm.tqdm(self.skills_to_rollout):
            self.policy.skill = ptu.from_numpy(skill)
            rollout = rollout_function(
                env=self.env,
                agent=self.policy,
                max_path_length=self.horizon_len,
                render=True,
                render_kwargs=self.render_kwargs,
            )
            rollout['skill'] = ptu.get_numpy(self.policy.skill)
            rollouts.append(rollout)

        return rollouts
