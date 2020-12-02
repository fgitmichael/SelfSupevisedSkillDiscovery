import abc


class TestRollouter(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self,):
        raise NotImplementedError

    @abc.abstractmethod
    def create_skills_to_rollout(self, **kwargs):
        raise NotImplementedError