from latent_with_splitseqs.base.tb_logging_base import PostEpochDiagnoWritingBase

from diayn.memory.replay_buffer_discrete import DIAYNEnvReplayBufferOptDiscrete


class NumTimesSkillUsedForTrainingLogger(PostEpochDiagnoWritingBase):

    def __init__(self,
                 *args,
                 replay_buffer: DIAYNEnvReplayBufferOptDiscrete,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer = replay_buffer

    def __call__(self, *args, epoch, **kwargs):
        self.plot_num_times_skills_used(epoch)

    def plot_num_times_skills_used(self, epoch):
        num_times_skills_used = self.replay_buffer.num_times_skill_used_for_training

        self.diagno_writer.writer.plot(
            num_times_skills_used,
            marker='o',
            tb_str="Replay-Buffer_Eval_Stats/Num Times skills used for training",
            step=epoch,
        )
