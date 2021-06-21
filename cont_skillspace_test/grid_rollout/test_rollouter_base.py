import abc

from rlkit.samplers.rollout_functions import rollout as rollout_function


class TestRollouter(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            env,
            policy,
            rollout_fun = rollout_function,
            horizon_len: int = 300,
    ):
        self.skill_grid = None

        self.rollout_fun = rollout_fun
        self.env = env
        self.policy = policy
        self.horizon_len = horizon_len

        self.rollouts = None

    def __call__(self,):
        if self.rollouts is None:
            rollouts = self.rollout_trajectories()
            self.rollouts = rollouts
            return rollouts
        else:
            return self.rollouts

    @abc.abstractmethod
    def rollout_trajectories(self):
        raise NotImplementedError
