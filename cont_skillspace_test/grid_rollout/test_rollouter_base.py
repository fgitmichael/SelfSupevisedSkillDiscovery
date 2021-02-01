import abc


class TestRollouter(object, metaclass=abc.ABCMeta):

    def __init__(self):
        self.skill_grid = None

    @abc.abstractmethod
    def __call__(self,):
        raise NotImplementedError

    @abc.abstractmethod
    def create_skills_to_rollout(self, **kwargs):
        raise NotImplementedError