from rlkit.torch.sac.diayn.policies import MakeDeterministic

class DiaynSkillPolicyWrapper():

    def __init__(self,
                 skill_policy: MakeDeterministic,
                 ):

        self.policy = skill_policy

        self.skill_dim = self.policy.stochastic_policy.skill_dim

    def set_skill(self,
                  skill: int
                  ):
        if skill <= self.skill_dim:
            self.policy.stochastic_policy.skill = skill
        else:
            raise ValueError('skill has to be <' + str(self.skill_dim))

    @property
    def num_skills(self):
        return self.skill_dim - 1

    def get_action(self,
                   obs_denormalized,
                   ):
        """
        Returns action of the skill policy dependant on DE-NORMALIZED observation
        Args:
            obs_denormalized   : (*observation_shape)-ndarray If environment is
                                 normalizing observation has to be denormalized
        Return:
            action             : action as trained with Diayn
        """
        action, info = self.policy.get_action(obs_denormalized)
        return action



