import abc
import tqdm
import numpy as np

from my_utils.rollout.frame_plus_obs_rollout import rollout as rollout_function
import rlkit.torch.pytorch_util as ptu

from diayn_seq_code_revised.policies.skill_policy \
    import SkillTanhGaussianPolicyRevised


class TestRollouter(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            env,
            policy: SkillTanhGaussianPolicyRevised,
            rollout_fun = rollout_function,
            horizon_len: int = 300,
    ):
        self.skill_grid = None

        self.rollout_fun = rollout_fun
        self.env = env
        self.policy = policy
        self.horizon_len = horizon_len

        self.rollouts = None

    def __call__(self, *args, **kwargs):
        if self.rollouts is None:
            rollouts = self.rollout_trajectories(*args, **kwargs)
            self.rollouts = rollouts
            return rollouts
        else:
            return self.rollouts

    @abc.abstractmethod
    def skills_to_rollout(self) -> np.ndarray:
        raise NotImplementedError

    def rollout_trajectories(
            self,
            render=False,
            render_kwargs=None,
    ):
        if render and render_kwargs is None:
            render_kwargs = dict(
                mode='rgb_array'
            )

        rollouts = []
        for skill in tqdm.tqdm(self.skills_to_rollout):
            self.policy.skill = ptu.from_numpy(skill)
            rollout = rollout_function(
                env=self.env,
                agent=self.policy,
                max_path_length=self.horizon_len,
                render=render,
                render_kwargs=render_kwargs,
            )
            rollout['skill'] = ptu.get_numpy(self.policy.skill)
            rollouts.append(rollout)

        return rollouts
