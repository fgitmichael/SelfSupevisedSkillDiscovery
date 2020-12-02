from latent_with_splitseqs.base.tb_logging_base import PostEpochDiagnoWritingBase

from self_supervised.memory.self_sup_replay_buffer \
    import SelfSupervisedEnvSequenceReplayBuffer


class ReplayBufferSkillDistributionPlotter(PostEpochDiagnoWritingBase):

    def __init__(self,
                 *args,
                 replay_buffer: SelfSupervisedEnvSequenceReplayBuffer,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.replay_buffer = replay_buffer

    def __call__(self, *args, epoch, **kwargs):
        self.plot_skill_dist(epoch=epoch)

    def plot_skill_dist(self, epoch):
        batch_dim = 0
        data_dim = 1
        saved_skills = self.replay_buffer.get_saved_skills(unique=True)
        assert saved_skills.shape[data_dim] == 2

        self.diagno_writer.writer.scatter(
            saved_skills[:, 0],
            saved_skills[:, 1],
            tb_str="Replay-Buffer_Eval_Stats/Saved Skill Distribution",
            step=epoch,
        )
