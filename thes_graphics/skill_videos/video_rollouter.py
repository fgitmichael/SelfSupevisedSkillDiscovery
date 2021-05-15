import tqdm

from cont_skillspace_test.grid_rollout.grid_rollouter import GridRollouter

from my_utils.rollout.frame_rollout import rollout as rollout_function

import rlkit.torch.pytorch_util as ptu


class VideoGridRollouter(GridRollouter):

    def __init__(self,
                 *args,
                 render_kwargs: dict,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.render_kwargs = render_kwargs

    def rollout_trajectories(self) -> list:
        rollouts = []

        for skill in tqdm.tqdm(self.skills_to_rollout):
            self.policy.skill = ptu.from_numpy(skill)
            rollout = rollout_function(
                env=self.env,
                agent=self.policy,
                max_path_length=self.horizon_len,
                render=True,
            )
            rollout['skill'] = ptu.get_numpy(self.policy.skill)
            rollouts.append(rollout)

        return rollouts
